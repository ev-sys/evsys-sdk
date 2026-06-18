"""Benchmark — load a harbor-format eval suite and score a model against it.

A benchmark on disk is a directory:

    data/benchmark/<name>/
        tasks.jsonl       # required — one HarborTask per line
        metadata.yaml     # optional — description, version, source, splits, ...
        images/           # optional — referenced by relative path from tasks
        raw/              # optional — pre-harbor source for traceability

`Benchmark.from_dir(path)` loads the suite. `bench.score(client)` runs each
task's `instruction` through an `InferenceClient`, then scores the completion
via the SDK's in-process verifier-fn registry (`verifiers/fns.py`). The result
collects per-task rows, top-level aggregates (`mean_reward`, `pass_rate`,
`n_tasks`), and optional breakdown buckets (e.g. per-toolkit pass rate) keyed
by an attribute path into `metadata`.

E2B and LLM-judge verifiers are recognized but not executed here — they need
network / sandboxes and live behind their own runners. Tasks carrying those
verifier kinds raise a clear error so callers don't silently mis-score.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

from .data_types import (
    E2BVerifier,
    HarborTask,
    InProcessVerifier,
    LLMJudgeVerifier,
    harbor_task_from_dict,
)
from .protocols import InferenceClient
from .registry import get_metric
from .verifiers import fns as verifier_fns

if TYPE_CHECKING:
    from .training.trajectory import TrajectoryGroup


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkTaskResult:
    """One scored task."""

    task_id: str
    instruction: str
    model_output: str
    expected: Any
    reward: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkScore:
    """Aggregate output of `Benchmark.score`."""

    metrics: dict[str, float]
    """Top-level aggregates: `mean_reward`, `pass_rate`, `n_tasks`."""
    per_task: list[BenchmarkTaskResult]
    """One entry per task in the same order as `Benchmark.tasks`."""
    breakdowns: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    """`{bucket_field: {bucket_value: {n, mean_reward, pass_rate}}}`.

    Populated when `score(..., breakdown_keys=[...])` is passed. Each bucket
    field is an attribute path into a task's `metadata` (e.g. `"toolkit"`)."""
    rollouts: list["TrajectoryGroup"] = field(default_factory=list)
    """Raw per-(task, sample) harbor rollouts (token ids + reward + usage), in
    task order; populated by `score_via_harbor`, empty for in-process `score()`.
    Lets callers upload per-sample eval predictions without re-running."""


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@dataclass
class Benchmark:
    """A harbor-format eval suite loaded into memory.

    Two builders:
      * `from_dir(path)`   — `data/benchmark/<name>/{tasks.jsonl,metadata.yaml}`
      * `from_iterable(name, rows, metadata=...)` — for tests / programmatic.

    `score(client)` runs inference + verification and returns a `BenchmarkScore`.
    """

    name: str
    tasks: list[HarborTask]
    metadata: dict = field(default_factory=dict)
    root: Path | None = None
    """Filesystem dir the benchmark was loaded from (None if in-memory)."""

    # -- constructors -----------------------------------------------------

    @classmethod
    def from_dir(cls, path: str | Path) -> Benchmark:
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"benchmark dir not found: {root}")

        tasks_path = root / "tasks.jsonl"
        if not tasks_path.is_file():
            raise FileNotFoundError(f"benchmark missing tasks.jsonl: {tasks_path}")

        tasks: list[HarborTask] = []
        for lineno, line in enumerate(tasks_path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{tasks_path}:{lineno}: malformed json: {e}") from e
            try:
                tasks.append(harbor_task_from_dict(row))
            except (KeyError, ValueError) as e:
                raise ValueError(f"{tasks_path}:{lineno}: {e}") from e

        metadata = _read_metadata_yaml(root / "metadata.yaml")
        name = str(metadata.get("name") or root.name)
        return cls(name=name, tasks=tasks, metadata=metadata, root=root)

    @classmethod
    def from_iterable(
        cls,
        name: str,
        rows: list[dict] | list[HarborTask],
        *,
        metadata: dict | None = None,
    ) -> Benchmark:
        tasks: list[HarborTask] = []
        for row in rows:
            if isinstance(row, HarborTask):
                tasks.append(row)
            else:
                tasks.append(harbor_task_from_dict(row))
        return cls(name=name, tasks=tasks, metadata=dict(metadata or {}), root=None)

    @classmethod
    def load(cls, spec: dict[str, Any], *, store: Any = None) -> Benchmark | None:
        """Resolve a benchmark spec ``{path | id | name}`` to a ``Benchmark``.

        The single resolver shared by the experiment config, the standalone
        ``run_benchmark``, and the CLI — so all accept the same references:

        * ``path`` → local harbor dir (delegates to ``from_dir``); offline / dev.
        * ``id``   → dashboard benchmark id, pulled into the local ``.evsys/``.
        * ``name`` → resolved to the latest version's id, then pulled.

        Returns ``None`` when the spec carries none of those. ``store`` is
        needed only for the id / name paths.
        """
        path = spec.get("path")
        if path:
            return cls.from_dir(path)
        bid, dashboard_name = spec.get("id"), spec.get("name")
        if not (bid or dashboard_name):
            return None
        from .workspace import Workspace, read_jsonl_rows

        ws = Workspace(store) if store is not None else Workspace()
        resolved = str(bid) if bid else ws.benchmark_id_for_name(str(dashboard_name))
        mat = ws.pull_benchmark(resolved)
        tasks = [harbor_task_from_dict(r) for r in read_jsonl_rows(mat.path)]
        return cls.from_iterable(str(dashboard_name or bid or "benchmark"), tasks)

    # -- scoring ----------------------------------------------------------

    def score(
        self,
        client: InferenceClient,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        prompt_builder: "callable | None" = None,
        breakdown_keys: list[str] | None = None,
        limit: int | None = None,
        metrics: list[str] | None = None,
        num_samples: int = 1,
    ) -> BenchmarkScore:
        """Run each task through `client` and score the completion.

        Sequential — wrap in a thread/process pool externally if you need
        concurrency. (Most local clients are GPU-bound and don't benefit.)

        `prompt_builder(task) -> str` lets callers shape the model input;
        default is `task.instruction` verbatim.

        `breakdown_keys` are dotted attribute paths into `task.metadata`. Each
        key produces `{value -> {n, mean_reward, pass_rate}}` in the result.

        `limit` caps how many tasks are scored — the first `limit` in
        `self.tasks` (deterministic, in benchmark order). Useful for fast
        smoke-runs on large benchmarks. ``None`` means score everything.
        """
        if breakdown_keys is None:
            breakdown_keys = []
        tasks_iter = self.tasks if limit is None else self.tasks[: max(0, int(limit))]
        n_samples = max(1, int(num_samples))

        per_task: list[BenchmarkTaskResult] = []
        task_rewards: list[list[float]] = []
        for task in tasks_iter:
            prompt = prompt_builder(task) if prompt_builder else task.instruction
            sample_rewards: list[float] = []
            last_completion, last_expected = "", None
            for _ in range(n_samples):
                completion = client.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
                reward, expected = _score_task(task, completion)
                sample_rewards.append(reward)
                last_completion, last_expected = completion, expected
            task_rewards.append(sample_rewards)
            per_task.append(
                BenchmarkTaskResult(
                    task_id=task.task_id,
                    instruction=task.instruction,
                    model_output=last_completion,
                    expected=last_expected,
                    # Per-task mean reward — drives the breakdown buckets.
                    reward=sum(sample_rewards) / len(sample_rewards),
                    metadata=dict(task.metadata),
                )
            )

        n = len(per_task)
        names = list(metrics) if metrics else ["mean_reward", "pass_rate"]
        score_metrics: dict[str, float] = {"n_tasks": float(n)}
        for name in names:
            try:
                score_metrics[name] = float(get_metric(name)().compute(task_rewards))
            except Exception:
                logger.warning("benchmark metric %r failed; skipping", name, exc_info=True)
        metrics = score_metrics
        breakdowns = _compute_breakdowns(per_task, breakdown_keys)
        return BenchmarkScore(metrics=metrics, per_task=per_task, breakdowns=breakdowns)

    async def score_via_harbor(
        self,
        *,
        model_name: str,
        model_path: str | None = None,
        model_client: str = "tinker",
        workspace_dir: Path,
        renderer_name: str | None = None,
        num_samples: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        limit: int | None = None,
        breakdown_keys: list[str] | None = None,
        metrics: list[str] | None = None,
        n_concurrent: int = 8,
        agent_import_path: str | None = None,
        max_retries: int = 2,
        _job_factory: Any | None = None,
    ) -> BenchmarkScore:
        """Score this benchmark through harbor's rollout engine — the harbor
        counterpart of :meth:`score`, returning the same :class:`BenchmarkScore`.

        Each task is rolled out ``num_samples`` times (one verifier reward per
        sample); ``metrics`` (registry names like ``pass@3``) reduce the per-task
        sample rewards, and ``time/tokens/cost_per_task`` come from harbor usage.
        ``model_client`` is ``"tinker"`` (on-policy checkpoint, needs
        ``model_path``) or ``"litellm"`` (closed/API model; ``model_name`` a
        litellm string). The result's :attr:`BenchmarkScore.rollouts` carries the
        raw per-(task, sample) rollouts so callers can upload eval predictions
        without re-running.
        """
        from .training.harbor_engine import run_harbor_rollouts
        from .training.harbor_eval import eval_metrics

        tasks = self.tasks if limit is None else self.tasks[: max(0, int(limit))]
        groups = await run_harbor_rollouts(
            tasks,                       # HarborTasks → fmt="harbor_task" (default), scored
            model_name=model_name,
            model_path=model_path,
            model_client=model_client,
            workspace_dir=workspace_dir,
            renderer_name=renderer_name,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            n_concurrent=n_concurrent,
            agent_import_path=agent_import_path,
            max_retries=max_retries,
            _job_factory=_job_factory,
        )
        score_metrics = eval_metrics(groups, metrics=list(metrics) if metrics else None)
        per_task: list[BenchmarkTaskResult] = []
        for task, group in zip(tasks, groups):
            rewards = list(group.rewards)
            per_task.append(
                BenchmarkTaskResult(
                    task_id=task.task_id,
                    instruction=task.instruction,
                    # harbor harvests token ids, not completion text.
                    model_output="",
                    expected=getattr(task.verifier, "expected", None),
                    # Per-task mean reward — drives the breakdown buckets.
                    reward=(sum(rewards) / len(rewards)) if rewards else 0.0,
                    metadata=dict(task.metadata),
                )
            )
        breakdowns = _compute_breakdowns(per_task, breakdown_keys or [])
        return BenchmarkScore(
            metrics=score_metrics,
            per_task=per_task,
            breakdowns=breakdowns,
            rollouts=list(groups),
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _score_task(task: HarborTask, completion: str) -> tuple[float, Any]:
    verifier = task.verifier
    if isinstance(verifier, InProcessVerifier):
        fn = verifier_fns.get(verifier.fn_name)
        reward = float(fn(completion, verifier.expected, dict(verifier.params or {})))
        return reward, verifier.expected
    if isinstance(verifier, E2BVerifier):
        raise NotImplementedError(
            f"task {task.task_id!r} uses E2BVerifier; run it through an E2B-aware "
            "scorer rather than Benchmark.score"
        )
    if isinstance(verifier, LLMJudgeVerifier):
        raise NotImplementedError(
            f"task {task.task_id!r} uses LLMJudgeVerifier; run it through an "
            "LLM-judge scorer rather than Benchmark.score"
        )
    raise TypeError(f"task {task.task_id!r} has unknown verifier type: {type(verifier).__name__}")


def _read_metadata_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    import yaml  # local — pyyaml is a required dep

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: metadata.yaml must be a mapping at the top level")
    return data


def _compute_breakdowns(
    per_task: list[BenchmarkTaskResult], breakdown_keys: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Bucket per-task rewards by each dotted ``metadata`` key →
    ``{key: {value: {n, mean_reward, pass_rate}}}``. Shared by the in-process
    and harbor scoring paths."""
    breakdowns: dict[str, dict[str, dict[str, float]]] = {}
    for key in breakdown_keys:
        buckets: dict[str, list[float]] = {}
        for r in per_task:
            bucket = str(_dotted_get(r.metadata, key, "__missing__"))
            buckets.setdefault(bucket, []).append(r.reward)
        breakdowns[key] = {
            bucket: {
                "n": float(len(rewards)),
                "mean_reward": sum(rewards) / len(rewards),
                "pass_rate": sum(1 for x in rewards if x >= 1.0) / len(rewards),
            }
            for bucket, rewards in buckets.items()
        }
    return breakdowns


def _dotted_get(d: dict, dotted_key: str, default: Any) -> Any:
    cur: Any = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


__all__ = [
    "Benchmark",
    "BenchmarkScore",
    "BenchmarkTaskResult",
]

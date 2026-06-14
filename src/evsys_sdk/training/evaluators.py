"""In-loop evaluator adapters — wrap an :class:`evsys_sdk.Benchmark`
so the training loop can score it every N steps against the live
sampler.

The post-training scoring path (``Experiment._eval_arm`` →
``Benchmark.score(client)``) takes a *synchronous* ``InferenceClient``.
The training-loop eval slot hands evaluators a live *async*
``SamplingClient``. :class:`_AsyncToSyncSampler` bridges the two with
``asyncio.run_coroutine_threadsafe`` + ``asyncio.to_thread`` so the
Benchmark iteration doesn't block the event loop and the existing
``ChatTemplatedInference`` wrapper (which inspects ``_tokenizer``) keeps
working unchanged.

The other half of the wiring is :func:`build_in_loop_evaluators` — a
metadata-aware factory that the three native algorithm composers call to
turn ``metadata.benchmark`` list entries with a ``run_every`` field into
:class:`BenchmarkEvaluator` instances ready for the
:class:`~evsys_sdk.training.loop.TrainingLoop` evaluators list.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

import tinker

from ..benchmark import Benchmark
from ..inference.chat_templated import ChatTemplatedInference

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async → sync sampler bridge so Benchmark.score keeps working
# ---------------------------------------------------------------------------


class _AsyncToSyncSampler:
    """Sync :class:`~evsys_sdk.protocols.InferenceClient` over an async
    :class:`~evsys_sdk.training.backend.SamplingClient`.

    ``.generate(prompt=...)`` is called sequentially by
    :meth:`Benchmark.score` from a worker thread; we route the awaited
    ``sample_async`` call back to the main event loop via
    :func:`asyncio.run_coroutine_threadsafe` so the inner async work
    runs on the loop that owns the live training client.

    Exposes ``_tokenizer`` so :class:`ChatTemplatedInference` can wrap
    this client without further plumbing — same shape
    :class:`TinkerInference` exposes.
    """

    name: ClassVar[str] = "in_loop_sampler"

    def __init__(self, sampler: Any, tokenizer: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._sampler = sampler
        self._tokenizer = tokenizer
        self._loop = loop

    def generate(
        self,
        *,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        prompt_mi = tinker.ModelInput.from_ints(ids)
        params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )
        coro = self._sampler.sample_async(prompt=prompt_mi, params=params)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        response = fut.result()
        return self._decode(response)

    def _decode(self, response: Any) -> str:
        seqs = getattr(response, "sequences", None)
        if not seqs:
            return ""
        first = seqs[0]
        tokens = (
            getattr(first, "tokens", None) or getattr(first, "token_ids", None) or []
        )
        if not tokens:
            return ""
        return self._tokenizer.decode(list(tokens))


# ---------------------------------------------------------------------------
# BenchmarkEvaluator — what the training loop iterates
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkEvaluator:
    """Score a :class:`~evsys_sdk.Benchmark` against the live sampler.

    The :class:`~evsys_sdk.training.loop.TrainingLoop` checks
    ``run_every`` per evaluator (see
    :meth:`TrainingLoop._is_due`); a value of ``0`` means "inherit the
    loop's ``eval_every``", so a single shared cadence still works.

    ``chat_template`` mirrors the post-training eval spec
    (``system_prompt`` + ``user_template`` + ``enable_thinking``) so the
    same YAML knob configures both the in-loop val and the final test set.
    """

    name: str
    benchmark: Benchmark
    tokenizer: Any
    run_every: int = 0
    max_tokens: int = 256
    temperature: float = 0.0
    breakdown_keys: list[str] = field(default_factory=list)
    chat_template: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    """Cap the number of tasks scored per eval — useful when the benchmark
    is large and you want quick in-loop snapshots."""
    engine: str = ""
    """``"harbor"`` → score through harbor's rollout engine (off the eval
    checkpoint). Anything else → the live-sampler InferenceClient path."""
    model_name: str | None = None
    workspace_dir: Any = None
    num_samples: int = 1

    async def evaluate(
        self, sampler: Any, *,
        model_path: str | None = None, step: int | None = None,
    ) -> dict[str, float]:
        if self.engine.lower() == "harbor" and model_path and self.model_name:
            return await self._evaluate_harbor(model_path)
        loop = asyncio.get_running_loop()
        client: Any = _AsyncToSyncSampler(sampler, self.tokenizer, loop)
        if self.chat_template:
            client = ChatTemplatedInference(client, **self.chat_template)
        score = await asyncio.to_thread(
            self.benchmark.score,
            client,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            breakdown_keys=list(self.breakdown_keys),
            limit=self.limit,
        )
        return dict(score.metrics)

    async def _evaluate_harbor(self, model_path: str) -> dict[str, float]:
        """Score the validation benchmark through harbor (same engine as
        training); reward = each task's verifier. Returns the metric dict."""
        import tempfile
        from pathlib import Path

        from .harbor_eval import eval_metrics, score_via_harbor

        tasks = (self.benchmark.tasks if self.limit is None
                 else self.benchmark.tasks[: max(0, self.limit)])
        ws = Path(self.workspace_dir) if self.workspace_dir else Path(
            tempfile.mkdtemp(prefix="evsys_val_")
        )
        groups = await score_via_harbor(
            tasks,
            model_name=self.model_name,
            model_path=model_path,
            workspace_dir=ws,
            num_samples=self.num_samples,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system_prompt=(self.chat_template or {}).get("system_prompt"),
        )
        return eval_metrics(groups)


# ---------------------------------------------------------------------------
# Factory — translate metadata.benchmark list entries → evaluators
# ---------------------------------------------------------------------------


def build_in_loop_evaluators(
    metadata: dict[str, Any] | None,
    *,
    tokenizer: Any,
    store: Any = None,
    model_name: str | None = None,
    workspace_dir: Any = None,
) -> list[BenchmarkEvaluator]:
    """Read ``metadata.benchmark`` and return one
    :class:`BenchmarkEvaluator` per entry whose ``run_every`` > 0.

    Single-dict ``benchmark`` and list form are both accepted (matches the
    parser in :meth:`evsys_sdk.experiment.Experiment._resolve_benchmarks`).
    Entries without ``run_every`` are post-training-only and silently
    skipped here — they're handled by ``Experiment._eval_arm``.

    ``tokenizer`` is required (used by the async→sync sampler bridge);
    ``store`` is required when an entry resolves a benchmark by ``id`` or
    ``name`` rather than ``path``.
    """
    if not metadata:
        return []
    raw = metadata.get("benchmark")
    if not raw:
        return []
    specs: list[dict[str, Any]] = (
        [raw] if isinstance(raw, dict) else list(raw)
    )
    out: list[BenchmarkEvaluator] = []
    for i, spec in enumerate(specs):
        if not isinstance(spec, dict):
            logger.warning(
                "build_in_loop_evaluators: benchmark[%d] not a dict — skipping",
                i,
            )
            continue
        run_every = int(spec.get("run_every") or 0)
        if run_every <= 0:
            continue
        bench = _materialize_benchmark(spec, store)
        if bench is None:
            logger.warning(
                "build_in_loop_evaluators: benchmark[%d] (%r) didn't resolve — skipping",
                i, spec.get("name"),
            )
            continue
        out.append(BenchmarkEvaluator(
            name=str(spec.get("name", f"benchmark_{i}")),
            benchmark=bench,
            tokenizer=tokenizer,
            run_every=run_every,
            max_tokens=int(spec.get("max_tokens", 256)),
            temperature=float(spec.get("temperature", 0.0)),
            breakdown_keys=list(spec.get("breakdown_keys") or []),
            chat_template=dict(spec.get("chat_template") or {}),
            limit=int(spec["limit"]) if spec.get("limit") is not None else None,
            engine=str(spec.get("engine", "")),
            model_name=model_name,
            workspace_dir=workspace_dir,
            num_samples=int(spec.get("num_samples", 1)),
        ))
    return out


def _materialize_benchmark(spec: dict[str, Any], store: Any) -> Benchmark | None:
    """Same logic as ``Experiment._materialize_benchmark`` — resolve a spec
    to a :class:`Benchmark`. Duplicated here so the training/ package stays
    independent of ``Experiment``; the two paths must stay in sync."""
    path = spec.get("path")
    if path:
        return Benchmark.from_dir(path)
    bid, dashboard_name = spec.get("id"), spec.get("name")
    if not (bid or dashboard_name):
        return None
    from ..data_types import harbor_task_from_dict
    from ..workspace import Workspace, read_jsonl_rows
    ws = Workspace(store) if store is not None else Workspace()
    resolved = str(bid) if bid else ws.benchmark_id_for_name(str(dashboard_name))
    mat = ws.pull_benchmark(resolved)
    tasks = [harbor_task_from_dict(r) for r in read_jsonl_rows(mat.path)]
    return Benchmark.from_iterable(dashboard_name or "benchmark", tasks)


__all__ = ["BenchmarkEvaluator", "build_in_loop_evaluators"]

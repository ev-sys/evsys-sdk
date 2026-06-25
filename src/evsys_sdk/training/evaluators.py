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
    ``run_every`` per evaluator (see :meth:`TrainingLoop._is_due`);
    a value of ``0`` disables the evaluator (it never fires).

    ``chat_template`` mirrors the post-training eval spec
    (``system_prompt`` + ``user_template`` + ``enable_thinking``) so the
    same YAML knob configures both the in-loop val and the final test set.
    """

    name: str
    benchmark: Benchmark
    tokenizer: Any
    run_every: int = 0
    split: str = "val"
    """Metric namespace + log split tag (e.g. ``val`` / ``test``). Comes from the
    benchmark spec's ``split`` field — the only thing that tells validation and
    test metrics apart in the log store."""
    max_tokens: int = 256
    temperature: float = 0.0
    breakdown_keys: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    """Registered metric names to compute (harbor engine), e.g.
    ``["pass@3", "pass^3", "avg"]``. Empty → ``mean_reward`` + ``pass_rate``."""
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
    n_concurrent: int = 8
    """Concurrent harbor trials (harbor engine only). Higher = more rollouts in
    flight against the sampler; all share one cached sampling client."""
    # Dashboard upload wiring. When ``store`` + ``run_id`` are present, the
    # harbor branch records one ``eval`` per invocation (tagged with ``step``,
    # so the many validations across a run stay distinct) and uploads its
    # per-task rollouts as ``kind='eval'`` predictions.
    store: Any = None
    run_id: str | None = None
    benchmark_id: str | None = None

    async def evaluate(
        self, sampler: Any, *,
        model_path: str | None = None, step: int | None = None,
    ) -> dict[str, float]:
        if self.engine.lower() == "harbor" and model_path and self.model_name:
            return await self._evaluate_harbor(model_path, step=step)
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
            metrics=list(self.metrics),
            num_samples=self.num_samples,
        )
        return dict(score.metrics)

    async def _evaluate_harbor(
        self, model_path: str, *, step: int | None = None,
    ) -> dict[str, float]:
        """Score the validation benchmark through harbor (same engine as
        training); reward = each task's verifier. Returns the metric dict and,
        when ``store`` + ``run_id`` are set, uploads the eval rollouts."""
        import tempfile
        from pathlib import Path

        ws = Path(self.workspace_dir) if self.workspace_dir else Path(
            tempfile.mkdtemp(prefix="evsys_val_")
        )
        if step is not None:
            ws = ws / f"step_{step}"
        score = await self.benchmark.score_via_harbor(
            model_name=self.model_name,
            model_path=model_path,
            workspace_dir=ws,
            num_samples=self.num_samples,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system_prompt=(self.chat_template or {}).get("system_prompt"),
            limit=self.limit,
            breakdown_keys=list(self.breakdown_keys),
            metrics=list(self.metrics) or None,
            n_concurrent=self.n_concurrent,
        )
        if self.store is not None and self.run_id:
            tasks = (self.benchmark.tasks if self.limit is None
                     else self.benchmark.tasks[: max(0, self.limit)])
            self._upload(tasks, score.rollouts, dict(score.metrics), step)
        return dict(score.metrics)

    def _upload(
        self, tasks: list[Any], groups: list[Any],
        metrics: dict[str, float], step: int | None,
    ) -> None:
        """Record one ``eval`` (per step) + its per-task rollout predictions on
        the dashboard. Best-effort: a dashboard hiccup must not kill training."""
        from .harbor_eval import eval_predictions, upload_eval_rollouts

        eval_id: str | None = None
        create_eval = getattr(self.store, "create_eval", None)
        if callable(create_eval):
            try:
                rec = create_eval(
                    run_id=self.run_id,
                    benchmark_id=self.benchmark_id,
                    step=step,
                    metrics=metrics,
                )
                if isinstance(rec, dict):
                    eval_id = rec.get("id") or rec.get("eval_id")
                else:
                    eval_id = getattr(rec, "id", None)
            except Exception:  # pragma: no cover — defensive
                logger.exception("create_eval failed for run %s", self.run_id)
        # Each validation mints its own eval (tagged with `step`); the per-task
        # predictions hang off that eval_id so step-5 / step-10 / final evals
        # stay distinguishable. No eval_id → don't upload orphan predictions.
        if eval_id is None:
            logger.warning(
                "skipping val rollout upload for run %s step %s: no eval_id",
                self.run_id, step,
            )
            return
        try:
            preds = eval_predictions(tasks, groups, eval_id=eval_id, step=step)
            upload_eval_rollouts(self.store, self.run_id, preds)
        except Exception:  # pragma: no cover — defensive
            logger.exception("eval rollout upload failed for run %s", self.run_id)


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
    run_id: str | None = None,
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
        bench = Benchmark.load(spec, store=store)
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
            split=str(spec.get("split", "val")),
            max_tokens=int(spec.get("max_tokens", 256)),
            temperature=float(spec.get("temperature", 0.0)),
            breakdown_keys=list(spec.get("breakdown_keys") or []),
            metrics=list(spec.get("metrics") or []),
            chat_template=dict(spec.get("chat_template") or {}),
            limit=int(spec["limit"]) if spec.get("limit") is not None else None,
            engine=str(spec.get("engine", "")),
            model_name=model_name,
            workspace_dir=workspace_dir,
            num_samples=int(spec.get("num_samples", 1)),
            n_concurrent=int(spec.get("n_concurrent", 8)),
            store=store,
            run_id=run_id,
            benchmark_id=(str(spec["id"]) if spec.get("id") is not None else None),
        ))
    return out


__all__ = ["BenchmarkEvaluator", "build_in_loop_evaluators"]

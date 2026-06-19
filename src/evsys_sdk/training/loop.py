"""TrainingLoop — algorithm-agnostic driver around a tinker
training client.

The loop owns:

* the ``for step in range(start, num_steps)`` iteration,
* dispatching ``forward_backward_async`` then ``optim_step_async``,
* routing a callable ``loss_fn`` through ``forward_backward_custom_async``,
* periodic checkpoint saves via :class:`~evsys_sdk.training.checkpoints.CheckpointManager`,
* periodic in-loop evaluation via :class:`Evaluator` objects,
* writing one row per step into ``ctx.log_store`` (so the existing
  ``forward_step_metrics`` forwarder picks them up unchanged),
* a final "final" checkpoint at the end of training.

What the loop does NOT own: data shaping. A :class:`StepBuilder` decides
what each batch contains and computes algorithm-specific metrics. That's
the seam SFT / SDFT / RL plug into.

Designed to be exercised against
:class:`~evsys_sdk.training.backend.MockBackend` end-to-end so tests don't
need a real tinker session.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

import tinker

from .backend import Backend, LossCallable, SamplingClient
from .callbacks import Callback, LoopState
from .checkpoints import CheckpointManager, ManifestRow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch / StepBuilder contracts — what an algorithm implements
# ---------------------------------------------------------------------------


@dataclass
class TrainingBatch:
    """One step's worth of training data + loss spec.

    Constructed by a :class:`StepBuilder` per step. The loop hands ``data``
    to the backend along with ``loss_fn`` and ``loss_fn_config``.
    """

    data: list[tinker.Datum]
    loss_fn: tinker.types.LossFnType | LossCallable
    """Either a tinker-recognised string name (``"cross_entropy"`` /
    ``"importance_sampling"`` / ...) for a server-side loss, OR a Python
    callable for a client-side custom loss. The loop dispatches the right
    backend method based on the type."""
    loss_fn_config: dict[str, Any] | None = None
    """Only used when ``loss_fn`` is a string."""
    metrics: dict[str, float] = field(default_factory=dict)
    """Algorithm-precomputed per-step metrics (e.g. teacher entropy,
    reward stats). Merged into the per-step log row."""


@runtime_checkable
class StepBuilder(Protocol):
    """The only thing each algorithm overrides.

    Owns data shaping + per-step metrics. Owns nothing about
    checkpointing / logging / eval / resume.
    """

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        """Produce the batch for ``step_idx`` (0-based)."""
        ...

    def step_metrics(
        self, step_idx: int, batch: TrainingBatch, fb_result: Any
    ) -> dict[str, float]:
        """Compute per-step metrics from the forward-backward result.

        ``fb_result`` is what the backend's
        ``forward_backward_*_async()`` future resolved to. For
        ``MockBackend`` it's a duck-typed
        :class:`~evsys_sdk.training.backend.ForwardBackwardResult`; for
        ``TinkerBackend`` it's the native ``tinker.ForwardBackwardOutput``.
        The StepBuilder is free to inspect ``fb_result.loss_fn_outputs``
        (one entry per Datum).
        """
        ...


# ---------------------------------------------------------------------------
# Evaluator contract — what the in-loop eval slot dispatches to
# ---------------------------------------------------------------------------


@runtime_checkable
class Evaluator(Protocol):
    """Periodic in-loop evaluator.

    Each evaluator owns one metric source (a Benchmark, a custom probe,
    etc.) and carries its own ``run_every`` cadence. This lets a single
    training run mix a fast val benchmark (``run_every: 50``) with a
    heavier test benchmark (``run_every: 500``).
    """

    name: str
    """Short name; used to prefix the metric keys (``val/<name>/<key>``)."""

    run_every: int
    """Per-evaluator step cadence. ``0`` → disabled (never fires).
    Positive → fire when ``(step + 1) % run_every == 0``."""

    async def evaluate(
        self, sampler: SamplingClient, *,
        model_path: str | None = None, step: int | None = None,
    ) -> dict[str, float]:
        ...


# ---------------------------------------------------------------------------
# Loop artifacts — what `.run()` returns to the algorithm composer
# ---------------------------------------------------------------------------


@dataclass
class LoopArtifacts:
    """Per-run filesystem + state pointers, used to populate `RunResult`."""

    run_dir: Path
    """The output directory written to. Algorithms surface this as
    ``RunResult.artifacts["run_dir"]`` for downstream consumers
    (e.g. ``TinkerInference.from_run_result``)."""
    manifest_path: Path
    """Path to the ``checkpoints.jsonl`` the loop wrote."""
    checkpoints: list[ManifestRow]
    """Manifest rows the loop wrote, in order. The last one is the
    final sampler — the URI eval consumes."""
    total_requested_steps: int
    """The step horizon passed to :meth:`TrainingLoop.run` (``num_steps``) —
    NOT the number of steps actually executed. These differ when a callback
    early-stops the loop via ``state.request_stop()``: this field still reads
    the requested ceiling. For "how many steps actually ran", read
    ``state.step + 1`` inside ``on_train_end`` (``state.step`` is the last
    executed index)."""
    train_seconds: float

    def as_dict(self) -> dict[str, str]:
        """Flatten to the ``RunResult.artifacts`` shape that downstream
        consumers (e.g. ``TinkerInference.from_run_result``) read."""
        out: dict[str, str] = {"run_dir": str(self.run_dir)}
        for row in self.checkpoints:
            if row.sampler_path:
                out[f"checkpoint-{row.name}"] = row.sampler_path
            # Full training-state path (weights + optimizer). Surfaced so
            # continual learning can chain a stage's final weights into the
            # next stage (see Experiment continual mode).
            if row.state_path:
                out[f"state-{row.name}"] = row.state_path
        return out


# ---------------------------------------------------------------------------
# Loop driver
# ---------------------------------------------------------------------------


@dataclass
class _LoopMetricKeys:
    """Per-step metric keys the loop always emits (independent of algorithm)."""

    step: str = "progress/step"
    done_frac: str = "progress/done_frac"
    epoch: str = "progress/epoch"
    finish_batch: str = "time/finish_batch"
    optim_lr: str = "optim/lr"


class TrainingLoop:
    """Drive a training run against a :class:`Backend`.

    Construction is intentionally explicit — no global state, no env-var
    knobs, no auto-resume. Algorithms compose the loop with their own
    StepBuilder + Backend + optional Evaluators.
    """

    def __init__(
        self,
        *,
        backend: Backend,
        step_builder: StepBuilder,
        log_store: Any,
        output_dir: str | Path,
        adam_params: tinker.AdamParams,
        save_every: int,
        evaluators: list[Evaluator] | None = None,
        callbacks: list[Callback] | None = None,
        log_context: Any = None,
        log_prefix: str = "",
        metric_keys: _LoopMetricKeys | None = None,
    ) -> None:
        self.backend = backend
        self.step_builder = step_builder
        self.log_store = log_store
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adam_params = adam_params
        self.save_every = save_every
        self.evaluators: list[Evaluator] = list(evaluators or [])
        self.callbacks: list[Callback] = list(callbacks or [])
        # The experiment-wide LogContext (shared with experiment-scope hooks),
        # threaded onto LoopState so loop-scope logger hooks reach ctx.ids/store.
        self.log_context = log_context
        self.log_prefix = log_prefix
        self._keys = metric_keys or _LoopMetricKeys()
        self.checkpoint_mgr = CheckpointManager(
            log_path=self.output_dir, save_every=save_every
        )

    # --- public surface -----------------------------------------------------

    async def run(self, *, num_steps: int, start_step: int = 0) -> LoopArtifacts:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0 (got {num_steps})")
        if start_step < 0 or start_step >= num_steps:
            raise ValueError(
                f"start_step={start_step} must be in [0, {num_steps})"
            )

        state = LoopState(
            step=start_step,
            num_steps=num_steps,
            output_dir=self.output_dir,
            backend=self.backend,
            log_store=self.log_store,
            checkpoint_mgr=self.checkpoint_mgr,
            ctx=self.log_context,
        )
        self._dispatch("on_train_start", state)

        t_start = time.time()
        last_step = start_step
        for step in range(start_step, num_steps):
            state.step = step
            last_step = step
            await self._run_one_step(step, num_steps, state)
            if state.stop_requested:
                logger.info(
                    "TrainingLoop: early-stopped at step %d (callback request)", step,
                )
                break

        # Always record a final checkpoint, even if `save_every` didn't land
        # on `num_steps - 1`. Downstream eval reads sampler_path off this row.
        await self._save_checkpoint("final", batch=last_step, state=state)

        artifacts = LoopArtifacts(
            run_dir=self.output_dir,
            manifest_path=self.checkpoint_mgr.manifest_path,
            checkpoints=self.checkpoint_mgr.rows,
            total_requested_steps=num_steps,
            train_seconds=time.time() - t_start,
        )
        self._dispatch("on_train_end", state, artifacts)
        return artifacts

    # --- per-step machinery -------------------------------------------------

    async def _run_one_step(self, step: int, num_steps: int,
                            state: LoopState | None = None) -> None:
        t0 = time.time()

        batch = await self.step_builder.build_batch(step)

        # Dispatch loss based on whether it's a name (server-side) or a
        # callable (client-side custom). Custom losses don't take
        # ``loss_fn_config`` — they're closures.
        if callable(batch.loss_fn) and not isinstance(batch.loss_fn, str):
            fb_future = self.backend.forward_backward_custom_async(
                batch.data, batch.loss_fn
            )
        else:
            fb_future = self.backend.forward_backward_async(
                batch.data,
                loss_fn=batch.loss_fn,
                loss_fn_config=batch.loss_fn_config,
            )
        optim_future = self.backend.optim_step_async(self.adam_params)

        fb_result = await fb_future.result_async()
        optim_result = await optim_future.result_async()

        # Build the per-step metric row. Layering, lowest precedence first:
        #   loop-emitted (progress/*, time/*, optim/lr)
        #   <- algorithm step_metrics (train_mean_nll, ...)
        #   <- optimizer-emitted metrics
        #   <- batch.metrics (algorithm-precomputed, e.g. teacher entropy)
        metrics: dict[str, float] = {
            self._keys.step: float(step),
            self._keys.done_frac: float(step + 1) / float(num_steps),
            self._keys.optim_lr: float(self.adam_params.learning_rate),
        }
        try:
            metrics.update(self.step_builder.step_metrics(step, batch, fb_result))
        except Exception:  # pragma: no cover  — never block the loop
            logger.exception("step_metrics raised on step %d; continuing", step)
        metrics.update(getattr(optim_result, "metrics", None) or {})
        metrics.update(batch.metrics)
        metrics[self._keys.finish_batch] = time.time() - t0

        self.log_store.log_metrics(metrics, step=step)
        if state is not None:
            self._dispatch("on_step_end", state, step, batch, metrics)

        if self.checkpoint_mgr.should_save(step):
            await self._save_checkpoint(f"step_{step + 1}", batch=step, state=state)

        due = [ev for ev in self.evaluators if self._is_due(ev, step)]
        if due:
            await self._run_eval(step, due, state)

    async def _save_checkpoint(self, name: str, *, batch: int,
                                state: LoopState | None = None) -> None:
        """Snapshot both training state (for resume) and sampler weights
        (for eval), then record one manifest row."""
        state_path = await self.backend.save_full_state(name)
        sampler_path = await self.backend.save_for_sampler(name)
        row = ManifestRow(
            name=name,
            batch=batch,
            state_path=state_path,
            sampler_path=sampler_path,
        )
        self.checkpoint_mgr.record(row)
        if state is not None:
            self._dispatch("on_checkpoint", state, row)

    def _is_due(self, ev: Evaluator, step: int) -> bool:
        """Per-evaluator cadence check. ``run_every <= 0`` → disabled;
        positive → fire when ``(step + 1) % run_every == 0``."""
        cadence = int(getattr(ev, "run_every", 0) or 0)
        if cadence <= 0:
            return False
        return (step + 1) % cadence == 0

    async def _run_eval(
        self, step: int, due: list[Evaluator], state: LoopState | None = None,
    ) -> None:
        """Take ONE sampler snapshot for the step, run each due evaluator,
        log results under ``val/<eval_name>/<metric>``."""
        sampler = await self.backend.snapshot_sampling_client(
            name=f"eval_{step + 1}"
        )
        model_path = getattr(sampler, "model_path", None)
        for ev in due:
            try:
                ev_metrics = await ev.evaluate(sampler, model_path=model_path, step=step + 1)
            except Exception:
                logger.exception(
                    "evaluator %r raised at step %d; continuing", ev.name, step
                )
                continue
            self.log_store.log_metrics(
                {f"val/{ev.name}/{k}": float(v) for k, v in ev_metrics.items()},
                step=step + 1,
                split="val",
            )
            if state is not None:
                self._dispatch("on_eval", state, step, ev.name, dict(ev_metrics))

    # --- callback dispatch -------------------------------------------------

    def _dispatch(self, hook: str, *args: Any) -> None:
        """Call ``hook`` on every callback (error-isolated). Thin wrapper over
        the shared :func:`~evsys_sdk.training.callbacks.dispatch` so the loop
        and the Experiment fan out identically."""
        from .callbacks import dispatch
        dispatch(self.callbacks, hook, *args)


__all__ = [
    "Evaluator",
    "LoopArtifacts",
    "StepBuilder",
    "TrainingBatch",
    "TrainingLoop",
]

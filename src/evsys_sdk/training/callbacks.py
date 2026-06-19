"""Callbacks — hook points inside :class:`~evsys_sdk.training.loop.TrainingLoop`
for logging, debugging, visualization, and early stopping.

The :class:`Callback` base has no-op defaults for every hook. Override only
the ones you need:

    class MyDebug(Callback):
        def on_step_end(self, state, step_idx, batch, metrics):
            if step_idx % 100 == 0:
                print(f"[{step_idx}] {metrics}")

Then hand to the loop:

    loop = TrainingLoop(..., callbacks=[MyDebug(), CsvMetricsCallback(...)])

A failing callback never kills the loop — the exception is logged at
WARNING and the loop continues. Same pattern as failing evaluators.

For early stopping (e.g. on a plateau in val pass_rate), call
``state.request_stop()`` from inside a hook and the loop breaks after the
current step. Researchers writing a new "stop after N evals without
improvement" policy subclass :class:`EarlyStoppingCallback`.

Three built-ins ship today:

* :class:`PrintProgressCallback` — tqdm-style stdout one-liner per step.
* :class:`CsvMetricsCallback` — mirror metrics.jsonl to a per-step CSV for
  pandas-friendly inspection.
* :class:`EarlyStoppingCallback` — request_stop after N evals with no
  improvement on a named metric.
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import get_callback, register_callback

if TYPE_CHECKING:
    from ..config import ExperimentConfig, RunConfig
    from ..experiment import ArmResult, EvalResult, ExperimentResult
    from ..protocols import RunResult
    from .backend import Backend, SamplingClient
    from .checkpoints import CheckpointManager, ManifestRow
    from .loop import LoopArtifacts, TrainingBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LogContext — experiment-wide context shared across ALL hooks (both scopes)
# ---------------------------------------------------------------------------


@dataclass
class LogContext:
    """Mutable context threaded to every callback hook, in BOTH the
    Experiment scope (lifecycle / benchmark eval) and the TrainingLoop scope
    (per-step / checkpoint). Logger callbacks *create* the dashboard ids and
    write them onto :attr:`ids`; later hooks read them back.

    The framework owns only :attr:`run_key` (the local output-dir name). All
    dashboard ids (``experiment_id`` / ``group:<name>`` / ``run_id``) are
    callback-owned and accumulate on :attr:`ids` as the ``create_*`` hooks
    fire. Arms run sequentially, so ``ids["run_id"]`` is the current arm's.
    """

    output_dir: Path
    config: "ExperimentConfig | None" = None
    store: Any = None
    """Resolved store handle (EvsysStore / LocalStore / DashboardClient) or
    None when no store is configured."""
    run_key: str | None = None
    run_config: "RunConfig | None" = None
    group_name: str | None = None
    ids: dict[str, str] = field(default_factory=dict)
    """Callback-populated dashboard ids: ``experiment_id``, ``group:<name>``,
    ``run_id``. Read parents here to link children (e.g. create_run reads
    ``ids['experiment_id']``)."""
    extras: dict[str, Any] = field(default_factory=dict)
    """Scratch for passing values between callbacks within a run (e.g.
    ``wandb_url`` set by wandb_logger, read by evsys_logger's create_run)."""


# ---------------------------------------------------------------------------
# LoopState — what loop-scope callbacks see
# ---------------------------------------------------------------------------


@dataclass
class LoopState:
    """Live training-loop context handed to every callback hook.

    The fields are mutated in-place by the loop as the run progresses;
    ``step`` advances every iteration, ``stop_requested`` flips when a
    callback calls :meth:`request_stop`. Other fields are immutable for
    the run's lifetime.
    """

    step: int
    """Current step index (0-based). Advances per iteration."""
    num_steps: int
    output_dir: Path
    backend: "Backend"
    log_store: Any
    """The same log_store the loop writes to. Callbacks can also write
    auxiliary rows (e.g. ``log_metrics({"debug/x": 1}, step=...)``)."""
    checkpoint_mgr: "CheckpointManager"
    stop_requested: bool = False
    ctx: "LogContext | None" = None
    """The experiment-wide :class:`LogContext` (shared with the Experiment-scope
    hooks). ``None`` for a bare ``run_experiment`` with no Experiment driving it;
    populated when the Experiment threads callbacks into the loop."""

    def request_stop(self) -> None:
        """Signal the loop to break after the current step completes.

        Used by EarlyStoppingCallback and similar policies. The loop honours
        it at the top of the next iteration so the current step's writes
        flush cleanly first.
        """
        self.stop_requested = True


# ---------------------------------------------------------------------------
# Callback base class
# ---------------------------------------------------------------------------


class Callback:
    """Override any of these. Defaults are no-ops so callbacks don't have to
    implement every hook.

    Hook signatures are positional + typed (not a generic event-bag) so
    IDE autocomplete and static type checking still work. Each hook fires
    at exactly one moment in the loop — see the docstrings.

    All hooks are sync (``def``, not ``async``). If you need to schedule
    async work, spawn ``asyncio.create_task(...)`` from inside the hook.
    """

    # --- lifecycle ---------------------------------------------------------

    def on_train_start(self, state: LoopState) -> None:
        """Fires once before the for-loop starts. Open log files, register
        a wandb run, snapshot the config — anything that should happen
        before the first step."""

    def on_train_end(self, state: LoopState, artifacts: "LoopArtifacts") -> None:
        """Fires once after the loop completes (including the final
        checkpoint save). Flush summary writes, close files."""

    # --- per-step ----------------------------------------------------------

    def on_step_end(
        self,
        state: LoopState,
        step_idx: int,
        batch: "TrainingBatch",
        metrics: dict[str, float],
    ) -> None:
        """Fires after every train step's metric row is written. The
        universal "do something per step" hook (printing, plotting,
        custom metric derivations, gradient debugging)."""

    # --- side events -------------------------------------------------------

    def on_checkpoint(self, state: LoopState, row: "ManifestRow") -> None:
        """Fires after each checkpoint manifest row is recorded. Useful
        for shipping to S3, pruning old checkpoints, kicking a side eval."""

    def on_eval(
        self,
        state: LoopState,
        step_idx: int,
        eval_name: str,
        metrics: dict[str, float],
    ) -> None:
        """Fires per evaluator after each in-loop eval completes. Useful
        for pushing to a dashboard, plotting val curves, driving
        early-stopping decisions. (Logger callbacks that also need the
        rollout predictions should use :meth:`on_benchmark_eval`, which the
        loop fires for benchmark evaluators with the full payload.)"""

    # --- experiment scope (dispatched by Experiment, not the loop) ---------
    # These let ONE logger callback instance own the full lifecycle: create
    # the experiment/run records, persist benchmark predictions, close the run.
    # The shared LogContext carries the dashboard ids between them.

    def on_experiment_start(self, ctx: LogContext) -> None:
        """Fires once at the start of an experiment, before any arm. A logger
        creates the experiment record here (``ctx.ids['experiment_id'] = ...``)."""

    def on_group_start(self, ctx: LogContext, group_name: str) -> None:
        """Fires when a new run-group is needed (n_repeats replicates or
        continual stages). A logger creates the group
        (``ctx.ids[f'group:{group_name}'] = ...``)."""

    def on_run_start(self, ctx: LogContext) -> None:
        """Fires per arm, before training. ``ctx.run_config`` is set. A logger
        opens its run-scoped sink (wandb.init / create_run →
        ``ctx.ids['run_id']``), reading ``ctx.ids['experiment_id']`` /
        ``ctx.ids[f'group:{ctx.group_name}']`` to parent it."""

    def on_benchmark_eval(
        self,
        ctx: LogContext,
        eval_result: "EvalResult",
        predictions: list[dict],
        *,
        step: int | None = None,
    ) -> None:
        """Fires per benchmark scored — in-loop (``step`` = the train step) or
        post-training (``step=None``). Carries metrics + breakdowns + tags
        (on ``eval_result``) AND the per-task prediction rows. A logger
        creates one eval row per ``(eval_result.name, step)`` and persists the
        predictions."""

    def on_run_end(
        self, ctx: LogContext, run_result: "RunResult", arm: "ArmResult",
    ) -> None:
        """Fires per arm, after eval, before the run is marked completed. A
        logger flushes/closes its run-scoped sink (wandb.finish) and records
        the final status (update_run)."""

    def on_experiment_end(
        self, ctx: LogContext, result: "ExperimentResult",
    ) -> None:
        """Fires once at the end of the experiment. Final summary / flush."""


# ---------------------------------------------------------------------------
# dispatch — error-isolated fan-out, shared by the loop AND the Experiment
# ---------------------------------------------------------------------------


def dispatch(callbacks: list[Callback], hook: str, *args: Any, **kwargs: Any) -> None:
    """Call ``hook`` on every callback. A raising callback NEVER propagates —
    the exception is logged at WARNING and the next callback runs. Used by both
    the TrainingLoop (loop-scope hooks) and the Experiment (experiment-scope
    hooks) so error isolation is identical everywhere."""
    for cb in callbacks or []:
        fn = getattr(cb, hook, None)
        if fn is None:
            continue
        try:
            fn(*args, **kwargs)
        except Exception:
            logger.exception(
                "callback %s.%s raised; continuing", type(cb).__name__, hook,
            )


# ---------------------------------------------------------------------------
# Built-in callbacks
# ---------------------------------------------------------------------------


class PrintProgressConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    every: int = 1
    keys: list[str] | None = None


@register_callback("print_progress")
@dataclass
class PrintProgressCallback(Callback):
    """Compact one-liner per step to stdout. Useful when you're not on a
    dashboard.

    Parameters
    ----------
    every:
        Only print every Nth step (default 1 = every step).
    keys:
        Metric keys to include in the printed dict. ``None`` (default)
        prints all keys; pass an explicit list to keep the line short.
    stream:
        Where to write (default ``sys.stdout``). Not YAML-configurable.
    """

    name: ClassVar[str] = "print_progress"
    Config: ClassVar[type] = PrintProgressConfig

    every: int = 1
    keys: list[str] | None = None
    stream: Any = field(default_factory=lambda: sys.stdout)

    def on_step_end(self, state, step_idx, batch, metrics):
        if self.every > 1 and (step_idx % self.every) != 0:
            return
        view = (
            {k: metrics[k] for k in self.keys if k in metrics}
            if self.keys is not None else metrics
        )
        line = f"[{step_idx + 1}/{state.num_steps}] " + " ".join(
            f"{k}={_fmt_value(v)}" for k, v in view.items()
        )
        print(line, file=self.stream, flush=True)


class CsvMetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    out_path: str
    delimiter: str = ","


@register_callback("csv_metrics")
@dataclass
class CsvMetricsCallback(Callback):
    """Mirror the loop's per-step metric writes into a CSV alongside
    ``metrics.jsonl``. Friendly for pandas / spreadsheet inspection.

    Parameters
    ----------
    out_path:
        Where to write. Parent directory is created if missing.
    delimiter:
        Field separator (default ``,``).
    """

    name: ClassVar[str] = "csv_metrics"
    Config: ClassVar[type] = CsvMetricsConfig

    out_path: Path
    delimiter: str = ","
    _header_written: bool = field(default=False, init=False, repr=False)
    _keys: list[str] = field(default_factory=list, init=False, repr=False)
    _fh: Any = field(default=None, init=False, repr=False)
    _writer: Any = field(default=None, init=False, repr=False)

    def on_train_start(self, state):
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh, delimiter=self.delimiter)

    def on_step_end(self, state, step_idx, batch, metrics):
        if self._writer is None:
            return
        if not self._header_written:
            self._keys = sorted(metrics.keys())
            self._writer.writerow(["step", *self._keys])
            self._header_written = True
        # When new keys appear later, append columns at the end (no rewrite).
        for k in metrics:
            if k not in self._keys:
                self._keys.append(k)
        self._writer.writerow(
            [step_idx, *[metrics.get(k, "") for k in self._keys]],
        )
        self._fh.flush()

    def on_train_end(self, state, artifacts):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                logger.exception("CsvMetricsCallback: close failed")


class EarlyStoppingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: str
    eval_name: str | None = None
    patience: int = 3
    mode: str = "max"
    min_delta: float = 0.0


@register_callback("early_stopping")
@dataclass
class EarlyStoppingCallback(Callback):
    """Watch a metric emitted on eval; request_stop after N evals without
    improvement.

    Parameters
    ----------
    metric:
        Metric key as it lands in :meth:`Callback.on_eval`'s ``metrics``
        dict (e.g. ``"pass_rate"``).
    eval_name:
        Restrict matching to one evaluator's output (e.g. ``"val"``). ``None``
        = match any evaluator.
    patience:
        Evaluations without improvement before stopping. Default 3.
    mode:
        ``"max"`` (default) for metrics where higher is better, ``"min"`` for
        loss-like metrics.
    min_delta:
        Treat improvements smaller than this as no improvement (default 0).
    """

    name: ClassVar[str] = "early_stopping"
    Config: ClassVar[type] = EarlyStoppingConfig

    metric: str
    eval_name: str | None = None
    patience: int = 3
    mode: str = "max"
    min_delta: float = 0.0
    _best: float | None = field(default=None, init=False, repr=False)
    _staleness: int = field(default=0, init=False, repr=False)

    def on_eval(self, state, step_idx, eval_name, metrics):
        if self.eval_name is not None and eval_name != self.eval_name:
            return
        if self.metric not in metrics:
            return
        v = float(metrics[self.metric])
        if self._best is None:
            self._best = v
            return
        improved = (
            v > self._best + self.min_delta if self.mode == "max"
            else v < self._best - self.min_delta
        )
        if improved:
            self._best = v
            self._staleness = 0
        else:
            self._staleness += 1
            if self._staleness >= self.patience:
                logger.info(
                    "EarlyStoppingCallback: stop after %d evals without improvement on %r",
                    self._staleness, self.metric,
                )
                state.request_stop()


# ---------------------------------------------------------------------------
# Logger callbacks — one per backend, each owning its sink across the full
# lifecycle (on_run_start → on_step_end/on_eval/on_benchmark_eval → on_run_end).
# ---------------------------------------------------------------------------


class WandbLoggerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    mode: str = "online"        # online | offline | disabled
    log_every: int = 1          # log per-step metrics every N steps
    max_pred_rows: int = 100    # cap rows in the predictions wandb.Table


@register_callback("wandb_logger")
class WandbLoggerCallback(Callback):
    """Log metrics + benchmark predictions to Weights & Biases.

    One W&B run per arm: opened in ``on_run_start`` (so it exists before the
    first step) and closed in ``on_run_end``. wandb is imported lazily — if it
    isn't installed the callback warns once and every hook no-ops (training is
    never affected; :func:`dispatch` also isolates errors). The run URL is
    surfaced on ``ctx.extras['wandb_url']`` so a downstream logger
    (evsys_logger) can record it on the dashboard run.
    """

    name: ClassVar[str] = "wandb_logger"
    Config: ClassVar[type] = WandbLoggerConfig

    def __init__(
        self,
        *,
        project: str | None = None,
        entity: str | None = None,
        name: str | None = None,
        mode: str = "online",
        log_every: int = 1,
        max_pred_rows: int = 100,
    ) -> None:
        self.project = project
        self.entity = entity
        self.name = name
        self.mode = mode
        self.log_every = max(1, int(log_every))
        self.max_pred_rows = max(0, int(max_pred_rows))
        self._wandb: Any = None
        self._run: Any = None
        self._disabled = False

    def _lazy_wandb(self) -> Any:
        if self._wandb is None and not self._disabled:
            try:
                import wandb  # noqa: PLC0415
                self._wandb = wandb
            except Exception:
                self._disabled = True
                logger.warning("wandb_logger: wandb not installed; disabling W&B logging")
        return self._wandb

    def on_run_start(self, ctx: LogContext) -> None:
        wb = self._lazy_wandb()
        if wb is None:
            return
        cfg = ctx.run_config.model_dump() if ctx.run_config is not None else {}
        project = self.project or (getattr(ctx.config, "name", None) if ctx.config else None) or "evsys"
        run_name = self.name or (getattr(ctx.run_config, "name", None) if ctx.run_config else None) or ctx.run_key
        try:
            self._run = wb.init(
                project=project, entity=self.entity, name=run_name,
                config=cfg, mode=self.mode, reinit=True,
            )
            url = getattr(self._run, "url", None)
            if url:
                ctx.extras["wandb_url"] = url
        except Exception:
            logger.exception("wandb_logger: wandb.init failed; disabling")
            self._run = None
            self._disabled = True

    def on_step_end(self, state: LoopState, step_idx, batch, metrics) -> None:
        if self._run is None or (self.log_every > 1 and (step_idx + 1) % self.log_every):
            return
        self._run.log({k: float(v) for k, v in metrics.items()}, step=step_idx)

    def on_eval(self, state: LoopState, step_idx, eval_name, metrics) -> None:
        if self._run is None:
            return
        self._run.log(
            {f"val/{eval_name}/{k}": float(v) for k, v in metrics.items()}, step=step_idx
        )

    def on_benchmark_eval(self, ctx, eval_result, predictions, *, step=None) -> None:
        if self._run is None:
            return
        ename = getattr(eval_result, "name", "benchmark")
        metrics = getattr(eval_result, "metrics", {}) or {}
        payload: dict[str, Any] = {f"eval/{ename}/{k}": float(v) for k, v in metrics.items()}
        if predictions and self.max_pred_rows:
            tbl = self._wandb.Table(columns=["task_id", "expected", "reward"])
            for p in predictions[: self.max_pred_rows]:
                tbl.add_data(p.get("task_id"), str(p.get("expected")), p.get("reward"))
            payload[f"eval/{ename}/predictions"] = tbl
        self._run.log(payload, **({"step": step} if step is not None else {}))

    def on_run_end(self, ctx, run_result, arm) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            finally:
                self._run = None


# ---------------------------------------------------------------------------
# Factory — build callbacks from {kind, params} specs (YAML surface)
# ---------------------------------------------------------------------------


def build_callbacks(specs: Any) -> list[Callback]:
    """Materialize callbacks from a list of ``{kind, params}`` specs.

    Each ``spec`` may be a :class:`~evsys_sdk.config.CallbackSpec` (or any
    object/dict with ``kind`` + ``params``). The ``kind`` is resolved through
    the callback registry; ``params`` are validated against the callback's
    ``Config`` (so a YAML typo fails loudly) before construction. Users
    register their own callbacks with ``@register_callback("my_name")`` — see
    the built-ins above for the contract (``name`` + ``Config`` ClassVars).
    """
    out: list[Callback] = []
    for spec in specs or []:
        kind = spec.kind if hasattr(spec, "kind") else spec["kind"]
        raw = (spec.params if hasattr(spec, "params") else spec.get("params")) or {}
        cls = get_callback(kind)
        cfg = getattr(cls, "Config", None)
        params = cfg(**raw).model_dump() if cfg is not None else dict(raw)
        out.append(cls(**params))
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _fmt_value(v: Any) -> str:
    try:
        f = float(v)
        if abs(f) < 1e-3 or abs(f) >= 1e6:
            return f"{f:.2e}"
        return f"{f:.4f}"
    except Exception:
        return str(v)


__all__ = [
    "Callback",
    "CsvMetricsCallback",
    "EarlyStoppingCallback",
    "LogContext",
    "LoopState",
    "PrintProgressCallback",
    "WandbLoggerCallback",
    "build_callbacks",
    "dispatch",
]

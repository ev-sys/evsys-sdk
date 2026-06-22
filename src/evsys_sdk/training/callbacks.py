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

    def on_train_data(self, ctx: "LogContext", rows: list[dict[str, Any]]) -> None:
        """Fires once in setup with the FINAL examples fed to the model (after
        the chat template / rendering). Lets a logger persist exactly what went
        into training."""

    def on_rollout(self, state: LoopState, step_idx: int, rollouts: list[Any]) -> None:
        """Fires per step with the algorithm's on-policy rollouts — only when
        ``log_rollouts`` is on (e.g. a ``--dry`` run) and the algorithm set
        ``batch.rollouts`` (RL/SDFT do; SFT never does)."""

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


class TensorBoardLoggerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    log_dir: str | None = None   # default: <output_dir>/tb/<run_key>
    flush_secs: int = 30


@register_callback("tensorboard_logger")
class TensorBoardLoggerCallback(Callback):
    """Log scalar metrics + eval scalars to TensorBoard, one event dir per arm.

    Opens a ``SummaryWriter`` in ``on_run_start`` (lazy import so the callback
    registers without torch installed) and closes it in ``on_run_end``."""

    name: ClassVar[str] = "tensorboard_logger"
    Config: ClassVar[type] = TensorBoardLoggerConfig

    def __init__(self, *, log_dir: str | None = None, flush_secs: int = 30) -> None:
        self._log_dir = log_dir
        self._flush_secs = int(flush_secs)
        self._writer: Any = None
        self._disabled = False

    def on_run_start(self, ctx: LogContext) -> None:
        if self._disabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415
        except Exception:
            self._disabled = True
            logger.warning("tensorboard_logger: tensorboard/torch missing; disabling")
            return
        path = self._log_dir or str(Path(ctx.output_dir) / "tb" / (ctx.run_key or "run"))
        self._writer = SummaryWriter(log_dir=path, flush_secs=self._flush_secs)

    def on_step_end(self, state: LoopState, step_idx, batch, metrics) -> None:
        if self._writer is None:
            return
        for k, v in metrics.items():
            self._writer.add_scalar(k, float(v), global_step=step_idx)

    def on_eval(self, state: LoopState, step_idx, eval_name, metrics) -> None:
        if self._writer is None:
            return
        for k, v in metrics.items():
            self._writer.add_scalar(f"val/{eval_name}/{k}", float(v), global_step=step_idx)

    def on_benchmark_eval(self, ctx, eval_result, predictions, *, step=None) -> None:
        if self._writer is None:
            return
        ename = getattr(eval_result, "name", "benchmark")
        gs = step if step is not None else 0
        for k, v in (getattr(eval_result, "metrics", {}) or {}).items():
            self._writer.add_scalar(f"eval/{ename}/{k}", float(v), global_step=gs)

    def on_run_end(self, ctx, run_result, arm) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None


class LocalLoggerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    print_every: int = 1         # stdout one-liner cadence (0 = silent)
    keys: list[str] | None = None  # restrict printed keys


@register_callback("local_logger")
class LocalLoggerCallback(Callback):
    """Human-readable local logging: prints what's happening per step AND
    persists everything **per run** under ``<output_dir>/<run_key>/logs/``,
    organised by concern::

        logs/data/training_data.jsonl   final examples fed to the model
        logs/training/metrics.jsonl     per-step train metrics
        logs/training/rollouts.jsonl    on-policy training rollouts (--dry only)
        logs/validation/metrics.jsonl   in-loop validation scores
        logs/validation/rollouts.jsonl  validation predictions/rollouts
        logs/test/metrics.jsonl         final benchmark scores
        logs/test/rollouts.jsonl        test predictions/rollouts
        logs/hypothesis.md              the experiment hypothesis (this run)
        logs/conclusion.md              the experiment conclusion + run status

    harbor's raw rollout workspace lives OUTSIDE ``logs/`` under
    ``<run_key>/.harbor/<phase>/`` so ``logs/`` stays clean. As the loop is
    handed a no-op store, this callback is the single local writer — no
    duplicate ``metrics.jsonl``. The experiment-scope hypothesis/conclusion are
    also mirrored to ``<output_dir>/experiment.md``."""

    name: ClassVar[str] = "local_logger"
    Config: ClassVar[type] = LocalLoggerConfig

    # split label -> per-concern folder
    _FOLDER: ClassVar[dict[str, str]] = {
        "train": "training", "val": "validation",
        "validation": "validation", "test": "test",
    }

    def __init__(self, *, print_every: int = 1, keys: list[str] | None = None) -> None:
        self.print_every = int(print_every)
        self.keys = keys
        self._dir: Path | None = None
        self._metrics_fps: dict[str, Any] = {}
        self._evals: list[dict] = []
        self._hypothesis: str | None = None
        self._runs: list[dict] = []   # one record per arm: {dir, run_key, status, evals}

    # --- experiment scope -------------------------------------------------
    def on_experiment_start(self, ctx: LogContext) -> None:
        meta = (getattr(ctx.config, "metadata", None) or {}) if ctx.config else {}
        self._hypothesis = meta.get("hypothesis")
        self._write_experiment_md(ctx, conclusion=None)

    def _write_experiment_md(self, ctx: LogContext, *, conclusion: str | None) -> None:
        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = getattr(ctx.config, "name", None) if ctx.config else None
        lines = [f"# {name or 'experiment'}", ""]
        lines.append(f"- hypothesis: {self._hypothesis or '(none)'}")
        if conclusion is not None:
            lines.append(f"- conclusion: {conclusion}")
        (out / "experiment.md").write_text("\n".join(lines) + "\n")

    # --- per run ----------------------------------------------------------
    def on_run_start(self, ctx: LogContext) -> None:
        self._dir = Path(ctx.output_dir) / (ctx.run_key or "run") / "logs"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._metrics_fps = {}
        self._evals = []
        (self._dir / "hypothesis.md").write_text(
            f"# {ctx.run_key or 'run'} — hypothesis\n\n{self._hypothesis or '(none)'}\n"
        )
        if self.print_every:
            print(f"[local_logger] run {ctx.run_key} → {self._dir}", flush=True)

    def _ensure_dir(self, ctx: LogContext) -> Path | None:
        if self._dir is None and ctx is not None:
            self._dir = Path(ctx.output_dir) / (ctx.run_key or "run") / "logs"
            self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir

    def _phase_dir(self, folder: str) -> Path:
        d = self._dir / folder        # type: ignore[operator]
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- metrics (one file per concern) -----------------------------------
    def _write_metrics(self, step: int, metrics: dict, split: str) -> None:
        if self._dir is None:
            return
        import json  # noqa: PLC0415
        folder = self._FOLDER.get(split, split)
        fp = self._metrics_fps.get(folder)
        if fp is None:
            fp = (self._phase_dir(folder) / "metrics.jsonl").open("a")
            self._metrics_fps[folder] = fp
        fp.write(json.dumps({"step": step, "split": split,
                             "metrics": {k: float(v) for k, v in metrics.items()}}) + "\n")
        fp.flush()

    def on_step_end(self, state: LoopState, step_idx, batch, metrics) -> None:
        self._write_metrics(step_idx, metrics, "train")
        if self.print_every and (step_idx + 1) % self.print_every == 0:
            view = metrics if self.keys is None else {k: metrics[k] for k in self.keys if k in metrics}
            line = f"[{step_idx + 1}/{state.num_steps}] " + " ".join(
                f"{k}={_fmt_value(v)}" for k, v in view.items()
            )
            print(line, flush=True)

    def on_eval(self, state: LoopState, step_idx, eval_name, metrics) -> None:
        self._write_metrics(step_idx, metrics, "val")
        if self.print_every:
            cells = " ".join(f"{k}={_fmt_value(v)}" for k, v in metrics.items())
            print(f"  [eval {eval_name} @ {step_idx}] {cells}", flush=True)

    # --- data going in ----------------------------------------------------
    def on_train_data(self, ctx: LogContext, rows: list[dict]) -> None:
        d = self._ensure_dir(ctx)
        if d is None:
            return
        import json  # noqa: PLC0415
        fp = self._phase_dir("data") / "training_data.jsonl"
        with fp.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        if self.print_every:
            print(f"  [training_data] {len(rows)} rows → {fp}", flush=True)

    # --- training rollouts (--dry) ----------------------------------------
    def on_rollout(self, state: LoopState, step_idx, rollouts) -> None:
        if self._dir is None:
            return
        import json  # noqa: PLC0415
        texts = self._harbor_completion_texts("train")
        recs: list[dict] = []
        flat = 0
        for gi, group in enumerate(rollouts or []):
            for ti, traj in enumerate(getattr(group, "trajectories", []) or []):
                turns = getattr(traj, "turns", []) or []
                text = next((getattr(t, "text", "") for t in reversed(turns)
                             if getattr(t, "text", "")), "")
                if not text and flat < len(texts):
                    text = texts[flat]
                flat += 1
                recs.append({
                    "step": step_idx, "group": gi, "traj": ti,
                    "reward": getattr(traj, "reward", None),
                    "text": text,
                    "usage": (getattr(traj, "metadata", {}) or {}).get("usage"),
                })
        fp = self._phase_dir("training") / "rollouts.jsonl"
        with fp.open("a") as f:
            for rec in recs:
                f.write(json.dumps(rec, default=str) + "\n")
        if self.print_every:
            print(f"  [rollouts step {step_idx}] {len(recs)} trajectories → {fp}", flush=True)

    def _harbor_completion_texts(self, phase: str) -> list[str]:
        """Best-effort: read decoded completions harbor wrote to
        ``<run>/.harbor/<phase>/jobs/<newest>/<trial>/agent/completion.txt``.
        Token-level Trajectory turns carry only token ids, so this recovers the
        text for the clean rollouts.jsonl. Order-based, hence best-effort."""
        if self._dir is None:
            return []
        jobs = self._dir.parent / ".harbor" / phase / "jobs"
        if not jobs.is_dir():
            return []
        job_dirs = sorted(p for p in jobs.iterdir() if p.is_dir())
        if not job_dirs:
            return []
        texts: list[str] = []
        for comp in sorted(job_dirs[-1].glob("*/agent/completion.txt")):
            try:
                texts.append(comp.read_text())
            except OSError:
                texts.append("")
        return texts

    # --- benchmark predictions (val / test) -------------------------------
    def on_benchmark_eval(self, ctx, eval_result, predictions, *, step=None) -> None:
        if self._dir is None:
            return
        import json  # noqa: PLC0415
        ename = getattr(eval_result, "name", "benchmark")
        metrics = dict(getattr(eval_result, "metrics", {}) or {})
        split = str(getattr(eval_result, "split", None) or "test")
        self._evals.append({"name": ename, "step": step, "split": split, "metrics": metrics})
        if metrics:                       # aggregate scores -> <folder>/metrics.jsonl
            self._write_metrics(int(step or 0), metrics, split)
        if predictions:                   # per-example predictions -> <folder>/rollouts.jsonl
            folder = self._FOLDER.get(split, split)
            fp = self._phase_dir(folder) / "rollouts.jsonl"
            with fp.open("a") as f:
                for p in predictions:
                    row = dict(p) if isinstance(p, dict) else {"prediction": p}
                    row.setdefault("benchmark", ename)
                    f.write(json.dumps(row, default=str) + "\n")
        if self.print_every:
            cells = " ".join(f"{k}={_fmt_value(v)}" for k, v in metrics.items())
            print(f"  [benchmark {ename}/{split}] {cells}  n_pred={len(predictions)}", flush=True)

    # --- close out --------------------------------------------------------
    def on_run_end(self, ctx, run_result, arm) -> None:
        if self._dir is not None:
            self._runs.append({
                "dir": self._dir, "run_key": ctx.run_key,
                "status": getattr(run_result, "status", None),
                "evals": list(self._evals),
            })
        for fp in self._metrics_fps.values():
            try:
                fp.close()
            except Exception:  # noqa: BLE001
                pass
        self._metrics_fps = {}

    def on_experiment_end(self, ctx, result) -> None:
        if self._hypothesis is None:
            self._hypothesis = getattr(result, "hypothesis", None)
        conclusion = getattr(result, "conclusion", None)
        self._write_experiment_md(ctx, conclusion=conclusion)
        for rec in self._runs:            # per-run conclusion.md
            lines = [f"# {rec['run_key']} — conclusion", ""]
            lines.append(f"- hypothesis: {self._hypothesis or '(none)'}")
            lines.append(f"- status: {rec['status']}")
            if conclusion is not None:
                lines.append(f"- conclusion: {conclusion}")
            for ev in rec["evals"]:
                cells = ", ".join(f"{k}={v:.4f}" for k, v in ev["metrics"].items()
                                  if isinstance(v, (int, float)))
                lines.append(f"- eval **{ev['name']}** ({ev['split']}, step={ev['step']}): {cells}")
            try:
                (rec["dir"] / "conclusion.md").write_text("\n".join(lines) + "\n")
            except OSError:
                pass


class EvsysLoggerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str | None = None
    flush_every: int = 1   # batch per-step metric uploads every N steps


@register_callback("evsys_logger")
class EvsysLoggerCallback(Callback):
    """Persist EVERYTHING to the evsys dashboard store via callbacks — the
    experiment/group/run records, per-step metrics, checkpoints, and benchmark
    evals + predictions. Keyed off the shared ``ctx.ids`` it populates
    (``experiment_id`` → ``group:<name>`` → ``run_id``).

    This is the "callbacks own the store" mode: construct ``Experiment`` WITHOUT
    a ``store=`` (so the orchestrator makes no store calls) and add this
    callback instead. If ``Experiment`` already has a store (``ctx.store`` set),
    this callback disables itself to avoid double-writing. The store handle is
    built lazily from the environment (EvsysStore) unless one is injected
    (``cb._store = ...``) — the test seam."""

    name: ClassVar[str] = "evsys_logger"
    Config: ClassVar[type] = EvsysLoggerConfig

    def __init__(self, *, project_id: str | None = None, flush_every: int = 1) -> None:
        self.project_id = project_id
        self.flush_every = max(1, int(flush_every))
        self._store: Any = None
        self._disabled = False
        self._buf: list[tuple[int, dict, str]] = []

    # -- store lifecycle ----------------------------------------------------

    def _ensure_store(self, ctx: LogContext) -> Any:
        if self._disabled:
            return None
        if ctx.store is not None:
            self._disabled = True
            logger.warning(
                "evsys_logger: Experiment already has a store; disabling to avoid "
                "double-writes (drop store= from Experiment to use this callback)"
            )
            return None
        if self._store is None:
            try:
                from ..store import EvsysStore  # noqa: PLC0415
                self._store = EvsysStore(project_id=self.project_id)
            except Exception:
                self._disabled = True
                logger.warning("evsys_logger: no usable store (missing EVSYS_API_KEY?); disabling")
        return self._store

    @staticmethod
    def _id(resp: Any) -> str | None:
        return resp.get("id") if isinstance(resp, dict) else None

    # -- experiment scope ---------------------------------------------------

    def on_experiment_start(self, ctx: LogContext) -> None:
        store = self._ensure_store(ctx)
        if store is None:
            return
        meta = (getattr(ctx.config, "metadata", None) or {}) if ctx.config else {}
        resp = store.create_experiment(
            experiment_name=getattr(ctx.config, "name", "experiment"),
            hypothesis=meta.get("hypothesis"),
            tags=list(meta.get("tags") or []) or None,
        )
        eid = self._id(resp)
        if eid:
            ctx.ids["experiment_id"] = eid

    def on_group_start(self, ctx: LogContext, group_name: str) -> None:
        if self._disabled or self._store is None:
            return
        resp = self._store.create_group(ctx.ids.get("experiment_id"), group_name)
        gid = self._id(resp)
        if gid:
            ctx.ids[f"group:{group_name}"] = gid

    def on_run_start(self, ctx: LogContext) -> None:
        if self._disabled or self._store is None:
            return
        rc = ctx.run_config
        gid = ctx.ids.get(f"group:{ctx.group_name}") if ctx.group_name else None
        resp = self._store.create_run(
            experiment_id=ctx.ids.get("experiment_id"),
            group_id=gid,
            recipe_kind=getattr(getattr(rc, "algorithm", None), "kind", None),
            run_config=rc.model_dump() if rc is not None else None,
            seed=getattr(rc, "seed", None),
            status="running",
            wandb_run_url=ctx.extras.get("wandb_url"),
        )
        rid = self._id(resp)
        if rid:
            ctx.ids["run_id"] = rid

    def on_benchmark_eval(self, ctx, eval_result, predictions, *, step=None) -> None:
        if self._disabled or self._store is None:
            return
        run_id = ctx.ids.get("run_id")
        if not run_id:
            return
        resp = self._store.create_eval(
            run_id=run_id,
            benchmark_id=getattr(eval_result, "benchmark_id", None),
            metrics=dict(getattr(eval_result, "metrics", {}) or {}),
            breakdowns=dict(getattr(eval_result, "breakdowns", {}) or {}) or None,
            step=step,
        )
        eval_id = self._id(resp)
        if predictions:
            from .harbor_eval import upload_eval_rollouts  # noqa: PLC0415
            rows = [{**p, "eval_id": eval_id} for p in predictions]
            upload_eval_rollouts(self._store, run_id, rows)

    def on_run_end(self, ctx, run_result, arm) -> None:
        if self._disabled or self._store is None:
            return
        self._flush(ctx)
        run_id = ctx.ids.get("run_id")
        if run_id:
            status = getattr(run_result, "status", None) or getattr(arm, "status", None)
            patch = {"status": status}
            err = getattr(run_result, "error", None) or getattr(arm, "error", None)
            if err:
                patch["error_message"] = err
            self._store.update_run(run_id, **patch)

    # -- loop scope ---------------------------------------------------------

    def on_step_end(self, state: LoopState, step_idx, batch, metrics) -> None:
        if self._disabled or self._store is None:
            return
        self._buf.append((step_idx, {k: float(v) for k, v in metrics.items()}, "train"))
        if len(self._buf) >= self.flush_every:
            self._flush(state.ctx)

    def on_eval(self, state: LoopState, step_idx, eval_name, metrics) -> None:
        if self._disabled or self._store is None:
            return
        self._buf.append(
            (step_idx, {f"{eval_name}/{k}": float(v) for k, v in metrics.items()}, "val")
        )

    def on_checkpoint(self, state: LoopState, row) -> None:
        if self._disabled or self._store is None:
            return
        ctx = state.ctx
        run_id = ctx.ids.get("run_id") if ctx else None
        uri = getattr(row, "sampler_path", None) or getattr(row, "state_path", None)
        if run_id and uri:
            self._store.add_checkpoint(
                run_id, uri=uri, label=getattr(row, "name", None), step=getattr(row, "batch", None),
            )

    def _flush(self, ctx: LogContext | None) -> None:
        if self._store is None or not self._buf or ctx is None:
            return
        run_id = ctx.ids.get("run_id")
        if not run_id:
            self._buf.clear()
            return
        for step, metrics, split in self._buf:
            try:
                self._store.log_metrics(run_id=run_id, step=int(step), metrics=metrics, split=split)
            except Exception:
                logger.warning("evsys_logger: log_metrics failed at step %s", step, exc_info=True)
        self._buf.clear()


class DebugLoggerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_len: int = 400          # truncate each value's repr to this many chars
    max_pred_rows: int = 3      # how many prediction rows to show


@register_callback("debug_logger")
class DebugLoggerCallback(Callback):
    """Pretty-print EVERYTHING handed to each callback hook — every lifecycle
    and loop event, with its arguments summarized. Pure introspection (no
    persistence); drop it into ``callbacks:`` to see exactly what the logger
    callbacks receive and in what order."""

    name: ClassVar[str] = "debug_logger"
    Config: ClassVar[type] = DebugLoggerConfig

    def __init__(self, *, max_len: int = 400, max_pred_rows: int = 3) -> None:
        self.max_len = int(max_len)
        self.max_pred_rows = int(max_pred_rows)

    def _short(self, v: Any) -> str:
        s = repr(v)
        return s if len(s) <= self.max_len else s[: self.max_len] + f"… (+{len(s) - self.max_len} chars)"

    def _ctx(self, ctx: Any) -> dict:
        rc = getattr(ctx, "run_config", None)
        return {
            "run_key": getattr(ctx, "run_key", None),
            "group_name": getattr(ctx, "group_name", None),
            "ids": dict(getattr(ctx, "ids", {}) or {}),
            "store": type(getattr(ctx, "store", None)).__name__ if getattr(ctx, "store", None) else None,
            "extras": dict(getattr(ctx, "extras", {}) or {}),
            "run_config.name": getattr(rc, "name", None),
            "output_dir": str(getattr(ctx, "output_dir", "")),
        }

    def _state(self, state: Any) -> dict:
        return {
            "step": getattr(state, "step", None),
            "num_steps": getattr(state, "num_steps", None),
            "has_ctx": getattr(state, "ctx", None) is not None,
        }

    def _emit(self, hook: str, fields: dict) -> None:
        print(f"\n🔍 [debug_logger] {hook}", flush=True)
        for k, v in fields.items():
            print(f"      {k} = {self._short(v)}", flush=True)

    # -- experiment scope ---------------------------------------------------
    def on_experiment_start(self, ctx):
        self._emit("on_experiment_start", {"ctx": self._ctx(ctx)})

    def on_group_start(self, ctx, group_name):
        self._emit("on_group_start", {"group_name": group_name, "ctx": self._ctx(ctx)})

    def on_run_start(self, ctx):
        self._emit("on_run_start", {"ctx": self._ctx(ctx)})

    def on_benchmark_eval(self, ctx, eval_result, predictions, *, step=None):
        self._emit("on_benchmark_eval", {
            "step": step,
            "eval_result.name": getattr(eval_result, "name", None),
            "eval_result.metrics": getattr(eval_result, "metrics", None),
            "eval_result.breakdowns": getattr(eval_result, "breakdowns", None),
            "eval_result.tags": getattr(eval_result, "tags", None),
            "n_predictions": len(predictions),
            "predictions[:n]": predictions[: self.max_pred_rows],
            "ctx.ids": dict(getattr(ctx, "ids", {}) or {}),
        })

    def on_run_end(self, ctx, run_result, arm):
        self._emit("on_run_end", {
            "run_result.status": getattr(run_result, "status", None),
            "run_result.metrics": getattr(run_result, "metrics", None),
            "arm.name": getattr(arm, "name", None),
            "arm.status": getattr(arm, "status", None),
            "ctx.ids": dict(getattr(ctx, "ids", {}) or {}),
        })

    def on_experiment_end(self, ctx, result):
        self._emit("on_experiment_end", {
            "result.status": getattr(result, "status", None),
            "result.best_arm": getattr(getattr(result, "best_arm", None), "name", None),
            "result.best_score": getattr(result, "best_score", None),
            "result.conclusion": getattr(result, "conclusion", None),
        })

    # -- loop scope ---------------------------------------------------------
    def on_train_start(self, state):
        self._emit("on_train_start", {"state": self._state(state)})

    def on_step_end(self, state, step_idx, batch, metrics):
        self._emit("on_step_end", {
            "step_idx": step_idx,
            "metrics": metrics,
            "batch.loss_fn": getattr(batch, "loss_fn", None),
            "batch.n_data": len(getattr(batch, "data", []) or []),
            "batch.metrics": getattr(batch, "metrics", None),
            "state": self._state(state),
        })

    def on_eval(self, state, step_idx, eval_name, metrics):
        self._emit("on_eval", {"step_idx": step_idx, "eval_name": eval_name, "metrics": metrics})

    def on_checkpoint(self, state, row):
        self._emit("on_checkpoint", {
            "row.name": getattr(row, "name", None),
            "row.batch": getattr(row, "batch", None),
            "row.sampler_path": getattr(row, "sampler_path", None),
            "row.state_path": getattr(row, "state_path", None),
        })

    def on_train_end(self, state, artifacts):
        self._emit("on_train_end", {
            "artifacts.total_requested_steps": getattr(artifacts, "total_requested_steps", None),
            "artifacts.train_seconds": getattr(artifacts, "train_seconds", None),
            "artifacts.run_dir": str(getattr(artifacts, "run_dir", "")),
            "n_checkpoints": len(getattr(artifacts, "checkpoints", []) or []),
        })


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
    "DebugLoggerCallback",
    "EvsysLoggerCallback",
    "LocalLoggerCallback",
    "LogContext",
    "LoopState",
    "PrintProgressCallback",
    "TensorBoardLoggerCallback",
    "WandbLoggerCallback",
    "build_callbacks",
    "dispatch",
]

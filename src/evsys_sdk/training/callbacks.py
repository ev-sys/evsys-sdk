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
    from .backend import Backend
    from .checkpoints import CheckpointManager, ManifestRow
    from .loop import LoopArtifacts, TrainingBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoopState — what callbacks see
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
    backend: Backend
    log_store: Any
    """The same log_store the loop writes to. Callbacks can also write
    auxiliary rows (e.g. ``log_metrics({"debug/x": 1}, step=...)``)."""
    checkpoint_mgr: CheckpointManager
    stop_requested: bool = False

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

    def on_train_end(self, state: LoopState, artifacts: LoopArtifacts) -> None:
        """Fires once after the loop completes (including the final
        checkpoint save). Flush summary writes, close files."""

    # --- per-step ----------------------------------------------------------

    def on_step_end(
        self,
        state: LoopState,
        step_idx: int,
        batch: TrainingBatch,
        metrics: dict[str, float],
    ) -> None:
        """Fires after every train step's metric row is written. The
        universal "do something per step" hook (printing, plotting,
        custom metric derivations, gradient debugging)."""

    # --- side events -------------------------------------------------------

    def on_checkpoint(self, state: LoopState, row: ManifestRow) -> None:
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
        early-stopping decisions."""


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
# DeltaSnapshotCallback — portable checkpoints for router-managed jobs
# ---------------------------------------------------------------------------


class DeltaSnapshotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    store_dir: str = ""
    """Node store directory (the persistent volume mount). Empty -> the
    ``EVSYS_STORE_DIR`` env var, then ``/data/store``."""
    events_url: str = ""
    """Topic to report checkpoints to. Empty -> ``EVSYS_EVENTS_URL``; still
    empty -> events are skipped (local runs work without a router)."""
    job_id: str = ""
    """Router job id. Empty -> ``EVSYS_JOB_ID``."""
    provider: str = ""
    """Cloud holding the volume. Empty -> ``EVSYS_PROVIDER``, then 'verda'."""
    volume: str = ""
    """Volume id for the StoreRef. Empty -> ``EVSYS_VOLUME``."""


@register_callback("delta_snapshot")
class DeltaSnapshotCallback(Callback):
    """Ship every loop checkpoint to the node's persistent volume as an
    XOR-delta, and report it so the router's CheckpointMap stays current.

    This is the node half of the JobRouter contract: the loop saves its
    checkpoint as usual; ``on_checkpoint`` re-encodes the saved files as
    base + compressed delta (:mod:`evsys_sdk.checkpoint_delta` over the raw
    bytes — file sizes are stable within a run, and a size change just
    re-bases), writes them to the volume store, and posts a ``checkpoint``
    event. A preempted job restarts from exactly the last row the router
    heard about.

    Enable from config like every callback::

        callbacks:
          - {kind: delta_snapshot, params: {}}

    On a router-provisioned node all params arrive via the agent's env
    (EVSYS_JOB_ID, EVSYS_STORE_DIR, EVSYS_EVENTS_URL, EVSYS_VOLUME), so the
    empty default config is the normal one.
    """

    Config: ClassVar[type] = DeltaSnapshotConfig

    def __init__(self, store_dir: str = "", events_url: str = "",
                 job_id: str = "", provider: str = "", volume: str = ""):
        import os as _os
        self.store_dir = store_dir or _os.environ.get("EVSYS_STORE_DIR",
                                                      "/data/store")
        self.events_url = events_url or _os.environ.get("EVSYS_EVENTS_URL", "")
        self.job_id = job_id or _os.environ.get("EVSYS_JOB_ID", "")
        self.provider = provider or _os.environ.get("EVSYS_PROVIDER", "verda")
        self.volume = volume or _os.environ.get("EVSYS_VOLUME", "")
        self._ck = None          # DeltaCheckpointer, created on first row
        self._base_key = "base.evd"

    @staticmethod
    def _as_state(paths: list[str]) -> dict:
        """Checkpoint files as uint8 arrays — byte-exact, format-agnostic."""
        import numpy as _np
        state = {}
        for p in paths:
            state[Path(p).name] = _np.frombuffer(
                Path(p).read_bytes(), dtype=_np.uint8).copy()
        return state

    def on_checkpoint(self, state: LoopState, row) -> None:
        from ..checkpoint_delta import DeltaCheckpointer
        from ..compute.checkpoint_store import LocalDirStore, put_file, sha256_of
        from ..compute.events_topic import post_event

        paths = [p for p in (row.state_path, row.sampler_path) if p]
        if not paths:
            return
        step = row.batch if row.batch is not None else state.step
        weights = self._as_state(paths)
        work = Path(state.output_dir) / "_delta_snapshots"
        try:
            if self._ck is None:
                self._ck = DeltaCheckpointer(weights, str(work), keep_last=3)
            self._ck.save(step, weights)
        except (ValueError, KeyError):
            # A file changed size/name mid-run: re-base rather than fail.
            self._base_key = f"base-{step}.evd"
            self._ck = DeltaCheckpointer(weights, str(work), keep_last=3)
            self._ck.save(step, weights)
        store = LocalDirStore(self.store_dir)
        delta_key = f"step-{step}.evd"
        put_file(store, self._base_key, work / "base.evd")
        put_file(store, delta_key, work / f"step-{step}.evd")
        digests = {k: sha256_of(store, k) for k in (self._base_key, delta_key)}
        if self.events_url and self.job_id:
            post_event(self.events_url, {
                "kind": "checkpoint", "job_id": self.job_id, "step": int(step),
                "store": {"kind": "local_dir", "provider": self.provider,
                          "volume": self.volume, "path": self.store_dir},
                "base_key": self._base_key, "delta_key": delta_key,
                "sha256": digests,
                "meta": {"checkpoint_name": row.name}})

    def on_train_end(self, state: LoopState, artifacts) -> None:
        from ..compute.events_topic import post_event
        if self.events_url and self.job_id:
            post_event(self.events_url,
                       {"kind": "done", "job_id": self.job_id})


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
    "DeltaSnapshotCallback",
    "EarlyStoppingCallback",
    "LoopState",
    "PrintProgressCallback",
    "build_callbacks",
]

"""Tests for `evsys_sdk.training.callbacks`.

Two surfaces under test:
  1. The TrainingLoop's callback dispatch (sites, fail-safe semantics, early
     stop via state.request_stop()).
  2. The three built-in callbacks (PrintProgressCallback,
     CsvMetricsCallback, EarlyStoppingCallback).
"""

from __future__ import annotations

import asyncio
import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import tinker

from evsys_sdk.training import (
    Callback,
    CsvMetricsCallback,
    EarlyStoppingCallback,
    LoopState,
    MockBackend,
    PrintProgressCallback,
    StepBuilder,
    TrainingBatch,
    TrainingLoop,
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _StubLogStore:
    def __init__(self):
        self.rows: list[dict] = []

    def log_metrics(self, metrics, *, step, split="train"):
        self.rows.append({"step": step, "split": split, "metrics": dict(metrics)})


def _adam():
    return tinker.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8)


def _datum():
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={},
    )


@dataclass
class _ConstSB:
    async def build_batch(self, step_idx):
        return TrainingBatch(data=[_datum()], loss_fn="cross_entropy")
    def step_metrics(self, step_idx, batch, fb_result):
        return {}


# A tiny recorder callback covering every hook.
@dataclass
class _Recorder(Callback):
    train_start: int = 0
    train_end: int = 0
    steps: list[int] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    evals: list[tuple[int, str, dict]] = field(default_factory=list)

    def on_train_start(self, state):
        self.train_start += 1
    def on_train_end(self, state, artifacts):
        self.train_end += 1
    def on_step_end(self, state, step_idx, batch, metrics):
        self.steps.append(step_idx)
    def on_checkpoint(self, state, row):
        self.checkpoints.append(row.name)
    def on_eval(self, state, step_idx, eval_name, metrics):
        self.evals.append((step_idx, eval_name, dict(metrics)))


# ---------------------------------------------------------------------------
# Dispatch sites
# ---------------------------------------------------------------------------


def test_callback_hooks_fire_at_expected_moments(tmp_path: Path):
    rec = _Recorder()
    loop = TrainingLoop(
        backend=MockBackend(), step_builder=_ConstSB(),
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=2, callbacks=[rec],
    )
    asyncio.run(loop.run(num_steps=4))
    # one train_start, four step_ends, two save-driven checkpoints + one final,
    # one train_end.
    assert rec.train_start == 1
    assert rec.steps == [0, 1, 2, 3]
    # checkpoints fire at step 1 (save_every=2 → (step+1)%2==0), step 3, final.
    assert rec.checkpoints == ["step_2", "step_4", "final"]
    assert rec.train_end == 1


def test_callback_failure_does_not_kill_loop(tmp_path: Path):
    class _Boom(Callback):
        def on_step_end(self, *a, **k):
            raise RuntimeError("boom")
    log = _StubLogStore()
    loop = TrainingLoop(
        backend=MockBackend(), step_builder=_ConstSB(),
        log_store=log, output_dir=tmp_path, adam_params=_adam(),
        save_every=10, callbacks=[_Boom()],
    )
    artifacts = asyncio.run(loop.run(num_steps=3))
    # loop completed all 3 steps despite the failing callback
    assert artifacts.total_steps == 3
    assert sum(1 for r in log.rows if r["split"] == "train") == 3


def test_callbacks_run_in_order_each_hook(tmp_path: Path):
    """Multiple callbacks dispatch in registration order."""
    log_calls: list[tuple[str, int]] = []
    class _A(Callback):
        def on_step_end(self, state, step_idx, batch, metrics):
            log_calls.append(("A", step_idx))
    class _B(Callback):
        def on_step_end(self, state, step_idx, batch, metrics):
            log_calls.append(("B", step_idx))
    loop = TrainingLoop(
        backend=MockBackend(), step_builder=_ConstSB(),
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=10, callbacks=[_A(), _B()],
    )
    asyncio.run(loop.run(num_steps=2))
    assert log_calls == [("A", 0), ("B", 0), ("A", 1), ("B", 1)]


# ---------------------------------------------------------------------------
# state.request_stop early stop
# ---------------------------------------------------------------------------


def test_request_stop_breaks_loop_after_current_step(tmp_path: Path):
    """Callback flips state.stop_requested; loop honours after writing the
    current step's metric row (so logs flush cleanly)."""
    class _StopAt(Callback):
        def __init__(self, at_step: int):
            self.at_step = at_step
        def on_step_end(self, state, step_idx, batch, metrics):
            if step_idx == self.at_step:
                state.request_stop()

    log = _StubLogStore()
    loop = TrainingLoop(
        backend=MockBackend(), step_builder=_ConstSB(),
        log_store=log, output_dir=tmp_path, adam_params=_adam(),
        save_every=10, callbacks=[_StopAt(at_step=1)],
    )
    artifacts = asyncio.run(loop.run(num_steps=10))
    # Stop fired AFTER step 1's metric row was written; loop broke before step 2.
    train_rows = [r for r in log.rows if r["split"] == "train"]
    assert [r["step"] for r in train_rows] == [0, 1]
    # Final checkpoint still recorded for the stopped run.
    assert artifacts.checkpoints[-1].name == "final"


# ---------------------------------------------------------------------------
# Built-ins: PrintProgressCallback
# ---------------------------------------------------------------------------


def test_print_progress_every_n_steps(tmp_path: Path):
    stream = io.StringIO()
    cb = PrintProgressCallback(every=2, stream=stream)
    loop = TrainingLoop(
        backend=MockBackend(), step_builder=_ConstSB(),
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=10, callbacks=[cb],
    )
    asyncio.run(loop.run(num_steps=5))
    lines = [ln for ln in stream.getvalue().splitlines() if ln.strip()]
    # every=2 → fires on steps 0, 2, 4
    assert len(lines) == 3


def test_print_progress_keys_filter(tmp_path: Path):
    stream = io.StringIO()
    cb = PrintProgressCallback(every=1, keys=["progress/step"], stream=stream)
    loop = TrainingLoop(
        backend=MockBackend(), step_builder=_ConstSB(),
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=10, callbacks=[cb],
    )
    asyncio.run(loop.run(num_steps=1))
    line = stream.getvalue().strip()
    assert "progress/step" in line
    assert "optim/lr" not in line   # filtered out


# ---------------------------------------------------------------------------
# Built-ins: CsvMetricsCallback
# ---------------------------------------------------------------------------


def test_csv_metrics_writes_header_plus_one_row_per_step(tmp_path: Path):
    out_path = tmp_path / "metrics.csv"
    cb = CsvMetricsCallback(out_path=out_path)
    loop = TrainingLoop(
        backend=MockBackend(), step_builder=_ConstSB(),
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=10, callbacks=[cb],
    )
    asyncio.run(loop.run(num_steps=3))
    rows = list(csv.reader(out_path.read_text().splitlines()))
    # header + 3 data rows
    assert len(rows) == 4
    assert rows[0][0] == "step"
    assert {int(r[0]) for r in rows[1:]} == {0, 1, 2}


def test_csv_metrics_closes_file_on_train_end(tmp_path: Path):
    out_path = tmp_path / "metrics.csv"
    cb = CsvMetricsCallback(out_path=out_path)
    loop = TrainingLoop(
        backend=MockBackend(), step_builder=_ConstSB(),
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=10, callbacks=[cb],
    )
    asyncio.run(loop.run(num_steps=1))
    # The file handle should be closed after on_train_end.
    assert cb._fh is not None
    assert cb._fh.closed


# ---------------------------------------------------------------------------
# Built-ins: EarlyStoppingCallback
# ---------------------------------------------------------------------------


def test_early_stopping_request_stop_after_patience():
    """EarlyStoppingCallback should request_stop after `patience` evals
    without improvement on the watched metric."""
    cb = EarlyStoppingCallback(metric="pass_rate", patience=2, mode="max")
    state = _make_state()
    # eval 1: establish baseline → no stop
    cb.on_eval(state, step_idx=0, eval_name="val", metrics={"pass_rate": 0.5})
    assert not state.stop_requested
    # eval 2: lower → stale=1, no stop yet
    cb.on_eval(state, step_idx=1, eval_name="val", metrics={"pass_rate": 0.4})
    assert not state.stop_requested
    # eval 3: lower → stale=2 == patience → stop
    cb.on_eval(state, step_idx=2, eval_name="val", metrics={"pass_rate": 0.3})
    assert state.stop_requested


def test_early_stopping_resets_on_improvement():
    cb = EarlyStoppingCallback(metric="pass_rate", patience=2, mode="max")
    state = _make_state()
    cb.on_eval(state, 0, "val", {"pass_rate": 0.5})
    cb.on_eval(state, 1, "val", {"pass_rate": 0.4})   # stale=1
    cb.on_eval(state, 2, "val", {"pass_rate": 0.7})   # improvement → reset
    cb.on_eval(state, 3, "val", {"pass_rate": 0.6})   # stale=1
    assert not state.stop_requested


def test_early_stopping_ignores_other_evaluators():
    """eval_name=val should ignore metrics from a different evaluator."""
    cb = EarlyStoppingCallback(metric="pass_rate", patience=1,
                               eval_name="val", mode="max")
    state = _make_state()
    cb.on_eval(state, 0, "val", {"pass_rate": 0.5})
    cb.on_eval(state, 1, "bench", {"pass_rate": 0.3})   # ignored
    assert not state.stop_requested


def test_early_stopping_min_mode():
    """mode='min' treats lower as better — for loss-like metrics."""
    cb = EarlyStoppingCallback(metric="loss", patience=1, mode="min")
    state = _make_state()
    cb.on_eval(state, 0, "val", {"loss": 1.0})
    cb.on_eval(state, 1, "val", {"loss": 2.0})  # worse (higher loss) → stale=1
    assert state.stop_requested


def _make_state() -> LoopState:
    """Minimal LoopState for callback unit tests (no backend interaction)."""
    return LoopState(
        step=0, num_steps=10, output_dir=Path("."),
        backend=None,   # type: ignore[arg-type]
        log_store=None,
        checkpoint_mgr=None,  # type: ignore[arg-type]
    )

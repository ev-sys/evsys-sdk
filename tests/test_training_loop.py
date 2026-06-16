"""Tests for `evsys_sdk.training.loop.TrainingLoop` against MockBackend.

The loop is the algorithm-agnostic driver — it owns the for-loop, the
checkpoint cadence, eval dispatch, and the per-step metric row shape, but
knows nothing about SFT / SDFT / RL math. These tests exercise that
contract end-to-end without spending a real tinker session.

Custom losses (the `forward_backward_custom_async` path) are first-class —
the test below confirms a callable batch.loss_fn routes through the right
backend method, and a string routes through the named-loss method.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("tinker")  # optional dep; not installed in base CI
pytest.importorskip("torch")

import tinker

from evsys_sdk.training import (
    LoopArtifacts,
    MockBackend,
    MockSamplingClient,
    StepBuilder,
    TrainingBatch,
    TrainingLoop,
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _StubLogStore:
    """Captures every log_metrics call for assertion."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def log_metrics(self, metrics: dict[str, float], *, step: int,
                    split: str = "train") -> None:
        self.rows.append({"step": step, "split": split, "metrics": dict(metrics)})


def _datum() -> tinker.Datum:
    """Minimal Datum for the loop's purposes — the loop doesn't inspect it."""
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={},
    )


@dataclass
class _ConstantStepBuilder:
    """Returns a static batch of `n` Datums on every step.

    `extra_metrics` lets a test inject per-step algorithm-precomputed metrics
    (the same channel SDFT uses for `sdft/mean_teacher_entropy` etc.).
    """

    loss_fn: tinker.types.LossFnType | Any = "cross_entropy"
    loss_fn_config: dict | None = None
    n: int = 2
    extra_metrics: dict[str, float] = field(default_factory=dict)
    build_calls: list[int] = field(default_factory=list)
    metric_calls: list[int] = field(default_factory=list)

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        self.build_calls.append(step_idx)
        return TrainingBatch(
            data=[_datum() for _ in range(self.n)],
            loss_fn=self.loss_fn,
            loss_fn_config=self.loss_fn_config,
            metrics=dict(self.extra_metrics),
        )

    def step_metrics(self, step_idx: int, batch: TrainingBatch, fb_result: Any) -> dict[str, float]:
        self.metric_calls.append(step_idx)
        return {"train/algo_specific": 0.42}


@dataclass
class _CannedEvaluator:
    """Returns `metrics` on every call, lets tests count the call count."""

    name: str
    metrics: dict[str, float]
    run_every: int = 0
    calls: list[Any] = field(default_factory=list)

    async def evaluate(self, sampler, **kwargs) -> dict[str, float]:
        self.calls.append(sampler)
        return dict(self.metrics)


# ---------------------------------------------------------------------------
# Construction + happy path
# ---------------------------------------------------------------------------


def _adam() -> tinker.AdamParams:
    return tinker.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8)


def test_rejects_zero_num_steps(tmp_path: Path):
    loop = TrainingLoop(
        backend=MockBackend(), step_builder=_ConstantStepBuilder(),
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=10,
    )
    with pytest.raises(ValueError, match="num_steps"):
        asyncio.run(loop.run(num_steps=0))


def test_rejects_start_step_out_of_range(tmp_path: Path):
    loop = TrainingLoop(
        backend=MockBackend(), step_builder=_ConstantStepBuilder(),
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=10,
    )
    with pytest.raises(ValueError, match="start_step"):
        asyncio.run(loop.run(num_steps=5, start_step=5))


def test_single_step_runs_end_to_end(tmp_path: Path):
    backend = MockBackend()
    sb = _ConstantStepBuilder()
    log = _StubLogStore()
    loop = TrainingLoop(
        backend=backend, step_builder=sb,
        log_store=log, output_dir=tmp_path, adam_params=_adam(),
        save_every=1,
    )
    artifacts = asyncio.run(loop.run(num_steps=1))

    assert sb.build_calls == [0]
    assert sb.metric_calls == [0]
    assert len(backend.fb_calls) == 1
    assert len(backend.optim_calls) == 1
    # exactly one metrics row per step.
    assert len(log.rows) == 1
    row = log.rows[0]
    assert row["step"] == 0
    assert row["metrics"]["progress/step"] == 0.0
    assert row["metrics"]["progress/done_frac"] == 1.0
    assert row["metrics"]["optim/lr"] == pytest.approx(1e-4)
    assert row["metrics"]["train/algo_specific"] == 0.42
    # save_every=1 triggers at step 0; +1 final save at the end.
    assert backend.save_state_calls == ["step_1", "final"]
    assert backend.save_sampler_calls == ["step_1", "final"]
    assert artifacts.total_requested_steps == 1
    assert artifacts.train_seconds >= 0.0


# ---------------------------------------------------------------------------
# Save cadence
# ---------------------------------------------------------------------------


def test_save_every_triggers_at_correct_steps(tmp_path: Path):
    backend = MockBackend()
    loop = TrainingLoop(
        backend=backend, step_builder=_ConstantStepBuilder(),
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=3,
    )
    asyncio.run(loop.run(num_steps=10))
    # (step + 1) % 3 == 0 → steps 2, 5, 8 → names step_3, step_6, step_9
    assert backend.save_sampler_calls == ["step_3", "step_6", "step_9", "final"]


def test_save_every_zero_only_writes_final(tmp_path: Path):
    backend = MockBackend()
    loop = TrainingLoop(
        backend=backend, step_builder=_ConstantStepBuilder(),
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=0,
    )
    asyncio.run(loop.run(num_steps=5))
    assert backend.save_sampler_calls == ["final"]


def test_manifest_rows_appear_on_disk(tmp_path: Path):
    backend = MockBackend()
    loop = TrainingLoop(
        backend=backend, step_builder=_ConstantStepBuilder(),
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=2,
    )
    artifacts = asyncio.run(loop.run(num_steps=4))
    # manifest_path is checkpoints.jsonl; one row per save.
    rows = artifacts.manifest_path.read_text().splitlines()
    assert len(rows) == 3  # step_2, step_4, final
    parsed = [json.loads(r) for r in rows]
    assert [p["name"] for p in parsed] == ["step_2", "step_4", "final"]
    assert all("state_path" in p and "sampler_path" in p for p in parsed)


# ---------------------------------------------------------------------------
# Eval cadence
# ---------------------------------------------------------------------------


def test_per_evaluator_run_every_calls_each_evaluator_with_snapshot(tmp_path: Path):
    backend = MockBackend()
    ev_a = _CannedEvaluator(name="val_a", metrics={"pass_rate": 0.7}, run_every=2)
    ev_b = _CannedEvaluator(name="val_b", metrics={"pass_rate": 0.5}, run_every=2)
    log = _StubLogStore()
    loop = TrainingLoop(
        backend=backend, step_builder=_ConstantStepBuilder(),
        log_store=log, output_dir=tmp_path, adam_params=_adam(),
        save_every=10, evaluators=[ev_a, ev_b],
    )
    asyncio.run(loop.run(num_steps=4))
    # run_every=2 fires at (step + 1) % 2 == 0 → steps 1 and 3.
    assert len(ev_a.calls) == 2
    assert len(ev_b.calls) == 2
    eval_rows = [r for r in log.rows if r.get("split") == "val"]
    assert len(eval_rows) == 4  # 2 evaluators × 2 fires
    keys = {k for r in eval_rows for k in r["metrics"]}
    assert {"val/val_a/pass_rate", "val/val_b/pass_rate"} <= keys


def test_eval_skipped_when_evaluators_empty(tmp_path: Path):
    backend = MockBackend()
    log = _StubLogStore()
    loop = TrainingLoop(
        backend=backend, step_builder=_ConstantStepBuilder(),
        log_store=log, output_dir=tmp_path, adam_params=_adam(),
        save_every=10, evaluators=[],
    )
    asyncio.run(loop.run(num_steps=3))
    assert all(r.get("split") != "val" for r in log.rows)


def test_failing_evaluator_does_not_kill_loop(tmp_path: Path):
    class _BoomEv:
        name = "boom"
        run_every = 1
        async def evaluate(self, sampler, **kwargs):
            raise RuntimeError("boom")

    log = _StubLogStore()
    loop = TrainingLoop(
        backend=MockBackend(), step_builder=_ConstantStepBuilder(),
        log_store=log, output_dir=tmp_path, adam_params=_adam(),
        save_every=10, evaluators=[_BoomEv()],
    )
    artifacts = asyncio.run(loop.run(num_steps=2))
    assert artifacts.total_requested_steps == 2  # finished despite the evaluator
    assert all(r.get("split") != "val" for r in log.rows)


# ---------------------------------------------------------------------------
# Custom loss routing
# ---------------------------------------------------------------------------


def test_string_loss_routes_through_named_path(tmp_path: Path):
    backend = MockBackend()
    sb = _ConstantStepBuilder(loss_fn="cross_entropy",
                              loss_fn_config={"label_smoothing": 0.1})
    loop = TrainingLoop(
        backend=backend, step_builder=sb,
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=10,
    )
    asyncio.run(loop.run(num_steps=2))
    assert len(backend.fb_calls) == 2
    assert backend.fb_calls[0]["loss_fn"] == "cross_entropy"
    assert backend.fb_calls[0]["loss_fn_config"] == {"label_smoothing": 0.1}
    assert backend.fb_custom_calls == []   # custom path NOT touched


def test_callable_loss_routes_through_custom_path(tmp_path: Path):
    backend = MockBackend()

    def _custom_loss(model_out, batch_meta):
        return 0.42

    sb = _ConstantStepBuilder(loss_fn=_custom_loss, loss_fn_config=None)
    loop = TrainingLoop(
        backend=backend, step_builder=sb,
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=10,
    )
    asyncio.run(loop.run(num_steps=2))
    assert backend.fb_calls == []           # named path NOT touched
    assert len(backend.fb_custom_calls) == 2
    assert backend.fb_custom_calls[0]["loss_fn"] is _custom_loss


# ---------------------------------------------------------------------------
# Algorithm-precomputed metrics merge into the per-step row
# ---------------------------------------------------------------------------


def test_batch_metrics_appear_in_log_row(tmp_path: Path):
    log = _StubLogStore()
    sb = _ConstantStepBuilder(
        extra_metrics={"sdft/teacher_entropy": 0.71, "sdft/n_completions": 16.0}
    )
    loop = TrainingLoop(
        backend=MockBackend(), step_builder=sb,
        log_store=log, output_dir=tmp_path, adam_params=_adam(),
        save_every=10,
    )
    asyncio.run(loop.run(num_steps=1))
    assert log.rows[0]["metrics"]["sdft/teacher_entropy"] == 0.71
    assert log.rows[0]["metrics"]["sdft/n_completions"] == 16.0


# ---------------------------------------------------------------------------
# LoopArtifacts as_dict for the RunResult.artifacts surface
# ---------------------------------------------------------------------------


def test_artifacts_as_dict_includes_run_dir_and_sampler_uris(tmp_path: Path):
    backend = MockBackend()
    loop = TrainingLoop(
        backend=backend, step_builder=_ConstantStepBuilder(),
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=2,
    )
    artifacts = asyncio.run(loop.run(num_steps=4))
    d = artifacts.as_dict()
    assert d["run_dir"] == str(tmp_path)
    assert "checkpoint-step_2" in d
    assert "checkpoint-step_4" in d
    assert "checkpoint-final" in d
    assert d["checkpoint-final"].startswith("mock://sampler/")


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_resume_picks_last_state_path_from_manifest(tmp_path: Path):
    """`CheckpointManager.find_resume()` returns the last row whose
    state_path is set, so a re-run can pass that back to the backend."""
    backend = MockBackend()
    loop = TrainingLoop(
        backend=backend, step_builder=_ConstantStepBuilder(),
        log_store=_StubLogStore(), output_dir=tmp_path, adam_params=_adam(),
        save_every=2,
    )
    asyncio.run(loop.run(num_steps=4))

    # A new manager pointed at the same dir should see the manifest.
    from evsys_sdk.training.checkpoints import CheckpointManager
    fresh = CheckpointManager(log_path=tmp_path, save_every=2)
    resume = fresh.find_resume()
    assert resume is not None
    assert resume.label == "final"
    assert resume.weights_path is not None
    assert resume.weights_path.startswith("mock://state/")


def test_resume_returns_none_when_manifest_missing(tmp_path: Path):
    from evsys_sdk.training.checkpoints import CheckpointManager
    mgr = CheckpointManager(log_path=tmp_path, save_every=2)
    assert mgr.find_resume() is None

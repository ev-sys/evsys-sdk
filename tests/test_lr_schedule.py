"""Unit tests for ``evsys_sdk.training.lr_schedule`` + loop wiring."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("tinker")
pytest.importorskip("torch")

import tinker

from evsys_sdk.algorithms.base import BaseAlgorithmConfig
from evsys_sdk.training import MockBackend, TrainingBatch, TrainingLoop
from evsys_sdk.training.lr_schedule import compute_lr, make_lr_fn


# ---------------------------------------------------------------------------
# Pure schedule math
# ---------------------------------------------------------------------------


def test_constant_is_flat():
    for step in range(10):
        assert compute_lr(
            kind="constant", step=step, num_steps=10, base_lr=1e-4
        ) == pytest.approx(1e-4)


def test_linear_warmup_ramps_then_holds():
    vals = [
        compute_lr(
            kind="linear_warmup",
            step=s,
            num_steps=10,
            base_lr=1.0,
            warmup_steps=4,
        )
        for s in range(10)
    ]
    assert vals[0] == pytest.approx(0.25)  # (0+1)/4
    assert vals[3] == pytest.approx(1.0)
    assert vals[4:] == [pytest.approx(1.0)] * 6


def test_linear_decay_ends_at_min():
    lr = compute_lr(
        kind="linear_decay",
        step=9,
        num_steps=10,
        base_lr=1.0,
        warmup_steps=0,
        min_lr_ratio=0.1,
    )
    assert lr == pytest.approx(0.1)


def test_cosine_endpoints():
    base, ratio = 1.0, 0.1
    start = compute_lr(
        kind="cosine", step=0, num_steps=10, base_lr=base, min_lr_ratio=ratio
    )
    end = compute_lr(
        kind="cosine", step=9, num_steps=10, base_lr=base, min_lr_ratio=ratio
    )
    assert start == pytest.approx(base)
    assert end == pytest.approx(base * ratio)


def test_cosine_with_warmup_midpoint_below_base():
    mid = compute_lr(
        kind="cosine_with_warmup",
        step=6,
        num_steps=10,
        base_lr=1.0,
        warmup_steps=2,
        min_lr_ratio=0.0,
    )
    # after warmup, cosine should be strictly below base before the end
    assert 0.0 < mid < 1.0


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown lr_schedule"):
        compute_lr(kind="nope", step=0, num_steps=5, base_lr=1e-4)


def test_make_lr_fn_matches_compute():
    fn = make_lr_fn(
        kind="cosine_with_warmup",
        base_lr=2e-4,
        num_steps=20,
        warmup_steps=5,
        min_lr_ratio=0.05,
    )
    for s in range(20):
        assert fn(s) == pytest.approx(
            compute_lr(
                kind="cosine_with_warmup",
                step=s,
                num_steps=20,
                base_lr=2e-4,
                warmup_steps=5,
                min_lr_ratio=0.05,
            )
        )


def test_base_config_defaults_constant():
    cfg = BaseAlgorithmConfig()
    assert cfg.lr_schedule == "constant"
    assert cfg.lr_warmup_steps == 0
    assert cfg.lr_min_ratio == 0.0


# ---------------------------------------------------------------------------
# Loop applies lr_fn to optim steps
# ---------------------------------------------------------------------------


class _StubLogStore:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def log_metrics(self, metrics, *, step, split="train"):
        self.rows.append({"step": step, "metrics": dict(metrics)})


def _datum() -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={},
    )


class _ConstBuilder:
    async def build_batch(self, step_idx: int) -> TrainingBatch:
        return TrainingBatch(data=[_datum(), _datum()], loss_fn="cross_entropy")

    def step_metrics(self, step_idx, batch, fb_result):
        return {}


def test_loop_applies_lr_schedule(tmp_path):
    backend = MockBackend()
    log = _StubLogStore()
    base = tinker.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8)
    n = 8
    lr_fn = make_lr_fn(
        kind="linear_warmup",
        base_lr=1e-3,
        num_steps=n,
        warmup_steps=4,
    )
    loop = TrainingLoop(
        backend=backend,
        step_builder=_ConstBuilder(),
        log_store=log,
        output_dir=tmp_path,
        adam_params=base,
        save_every=n,
        lr_fn=lr_fn,
    )
    asyncio.run(loop.run(num_steps=n))

    # MockBackend records each AdamParams passed to optim_step_async
    lrs = [float(a.learning_rate) for a in backend.optim_calls]
    assert len(lrs) == n
    assert lrs[0] == pytest.approx(1e-3 * 0.25)
    assert lrs[3] == pytest.approx(1e-3)
    assert lrs[4] == pytest.approx(1e-3)
    # logged optim/lr matches
    logged = [r["metrics"]["optim/lr"] for r in log.rows]
    assert logged == pytest.approx(lrs)

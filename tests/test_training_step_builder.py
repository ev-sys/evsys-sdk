"""Tests for `evsys_sdk.training.step_builder.SFTStepBuilder`.

Pure unit tests — no backend involved, no real tinker session, no async
loop. The builder's only jobs are: slice a static list of Datums into
batches (wrapping modulo length so the loop can exceed one epoch), and
compute ``train_mean_nll`` from per-position logprobs + weights.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import tinker
import torch

from evsys_sdk.training.step_builder import SFTStepBuilder


def _datum(*, weights: list[float]) -> tinker.Datum:
    """Datum with a weight mask but a dummy model_input (the builder doesn't
    inspect input tokens)."""
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(len(weights)))),
        loss_fn_inputs={
            "weights": tinker.TensorData.from_torch(
                torch.tensor(weights, dtype=torch.float32)
            ),
        },
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_rejects_empty_datums():
    with pytest.raises(ValueError, match="datums is empty"):
        SFTStepBuilder(datums=[], batch_size=4)


def test_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        SFTStepBuilder(datums=[_datum(weights=[1.0])], batch_size=0)


def test_steps_per_epoch_floor_division():
    sb = SFTStepBuilder(
        datums=[_datum(weights=[1.0]) for _ in range(10)], batch_size=3,
    )
    assert sb.steps_per_epoch == 3   # floor(10 / 3)


# ---------------------------------------------------------------------------
# build_batch — slicing + wrap-around
# ---------------------------------------------------------------------------


def test_build_batch_returns_batch_size_data():
    sb = SFTStepBuilder(
        datums=[_datum(weights=[1.0]) for _ in range(10)], batch_size=3,
    )
    batch = asyncio.run(sb.build_batch(0))
    assert len(batch.data) == 3
    assert batch.loss_fn == "cross_entropy"


def test_build_batch_advances_across_steps():
    sb = SFTStepBuilder(
        datums=[_datum(weights=[float(i)]) for i in range(10)], batch_size=2,
    )
    b0 = asyncio.run(sb.build_batch(0))
    b1 = asyncio.run(sb.build_batch(1))
    # Different datums in different steps (we read the weights tensor to compare).
    w0 = [d.loss_fn_inputs["weights"].to_torch().tolist() for d in b0.data]
    w1 = [d.loss_fn_inputs["weights"].to_torch().tolist() for d in b1.data]
    assert w0 != w1


def test_build_batch_wraps_past_epoch_end():
    """Past one epoch the builder should still return ``batch_size`` Datums
    (wrap-around). Replaces the cookbook's _RepeatingSDFTProvider hack."""
    sb = SFTStepBuilder(
        datums=[_datum(weights=[1.0]) for _ in range(5)], batch_size=3,
    )
    # 5 datums, batch=3: step 0 → [0,1,2]; step 1 → [3,4, 0]; step 2 → [1,2,3].
    for step in range(10):
        batch = asyncio.run(sb.build_batch(step))
        assert len(batch.data) == 3


def test_seed_shuffles_order():
    """Different seeds → different first-batch composition (high probability)."""
    datums = [_datum(weights=[float(i)]) for i in range(20)]
    sb_a = SFTStepBuilder(datums=datums, batch_size=4, seed=1)
    sb_b = SFTStepBuilder(datums=datums, batch_size=4, seed=2)
    ba = asyncio.run(sb_a.build_batch(0))
    bb = asyncio.run(sb_b.build_batch(0))
    wa = [d.loss_fn_inputs["weights"].to_torch().item() for d in ba.data]
    wb = [d.loss_fn_inputs["weights"].to_torch().item() for d in bb.data]
    assert wa != wb


def test_seed_is_deterministic_across_constructions():
    datums = [_datum(weights=[float(i)]) for i in range(20)]
    sb_a = SFTStepBuilder(datums=datums, batch_size=4, seed=42)
    sb_b = SFTStepBuilder(datums=datums, batch_size=4, seed=42)
    ba = asyncio.run(sb_a.build_batch(0))
    bb = asyncio.run(sb_b.build_batch(0))
    wa = [d.loss_fn_inputs["weights"].to_torch().item() for d in ba.data]
    wb = [d.loss_fn_inputs["weights"].to_torch().item() for d in bb.data]
    assert wa == wb


# ---------------------------------------------------------------------------
# step_metrics — train_mean_nll math
# ---------------------------------------------------------------------------


def test_step_metrics_computes_train_mean_nll_from_dict_outputs():
    """MockBackend emits dict loss_fn_outputs with raw lists — the builder
    handles them via _coerce_floats."""
    sb = SFTStepBuilder(datums=[_datum(weights=[1.0, 1.0, 0.0])], batch_size=1)
    batch = asyncio.run(sb.build_batch(0))

    class _Result:
        # one Datum → one entry; logprob -0.5 on masked positions, -10 on the
        # unmasked third position which should be ignored by the weights.
        loss_fn_outputs = [{"logprobs": [-0.5, -0.5, -10.0]}]

    metrics = sb.step_metrics(0, batch, _Result())
    # mean_nll = -sum(logprob * w) / sum(w)
    # masked positions: [0, 1] → -((-0.5 + -0.5)/2) = 0.5
    assert metrics["train_mean_nll"] == pytest.approx(0.5)


def test_step_metrics_handles_tensor_data_outputs():
    """Real TinkerBackend returns TensorData — `to_torch()` is the seam."""
    sb = SFTStepBuilder(datums=[_datum(weights=[1.0, 1.0])], batch_size=1)
    batch = asyncio.run(sb.build_batch(0))

    class _TensorData:
        def __init__(self, vals):
            self._vals = vals
        def to_torch(self):
            return torch.tensor(self._vals, dtype=torch.float32)

    class _Result:
        loss_fn_outputs = [{"logprobs": _TensorData([-0.25, -0.75])}]

    metrics = sb.step_metrics(0, batch, _Result())
    assert metrics["train_mean_nll"] == pytest.approx(0.5)


def test_step_metrics_empty_on_missing_outputs():
    """If the result has no loss_fn_outputs (e.g. backend error), return
    an empty dict — the loop merges this with optimizer metrics so the row
    still lands, but train_mean_nll is missing for the step."""
    sb = SFTStepBuilder(datums=[_datum(weights=[1.0])], batch_size=1)
    batch = asyncio.run(sb.build_batch(0))

    class _Result:
        loss_fn_outputs = None

    assert sb.step_metrics(0, batch, _Result()) == {}


def test_step_metrics_safe_on_token_count_mismatch():
    """If a downstream backend returns fewer logprobs than the mask
    (rare but possible on truncation), truncate to min length — don't
    blow up the loop."""
    sb = SFTStepBuilder(datums=[_datum(weights=[1.0, 1.0, 1.0])], batch_size=1)
    batch = asyncio.run(sb.build_batch(0))

    class _Result:
        loss_fn_outputs = [{"logprobs": [-0.5]}]   # shorter than the mask

    metrics = sb.step_metrics(0, batch, _Result())
    # uses min length → 1 position → -((-0.5)*1)/1 = 0.5
    assert metrics["train_mean_nll"] == pytest.approx(0.5)


def test_step_metrics_aggregates_across_batch():
    """Multiple Datums in a batch → weighted average of their per-position
    logprobs by the per-position weights."""
    sb = SFTStepBuilder(
        datums=[
            _datum(weights=[1.0, 1.0]),
            _datum(weights=[1.0, 1.0]),
        ],
        batch_size=2,
    )
    batch = asyncio.run(sb.build_batch(0))

    class _Result:
        loss_fn_outputs = [
            {"logprobs": [-1.0, -1.0]},   # datum A
            {"logprobs": [-0.0, -0.0]},   # datum B
        ]

    metrics = sb.step_metrics(0, batch, _Result())
    # 4 masked positions total, sum of logprobs = -2.0 → mean_nll = 0.5
    assert metrics["train_mean_nll"] == pytest.approx(0.5)

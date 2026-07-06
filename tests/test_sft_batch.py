"""Unit tests for ``SFT``'s per-step batch logic.

The SFT batch slicing + ``train_mean_nll`` math used to live on the
now-deleted ``SFTStepBuilder``; it moved into
:class:`~evsys_sdk.algorithms.sft.SFT` (``build_batch`` / ``step_metrics``).
These are pure unit tests — no backend, no tinker session, no loop. We set
``algo._datums`` directly to exercise the per-step methods in isolation
(``setup`` is covered end-to-end in ``test_algorithms_sft``).
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("tinker")  # optional dep; not installed in base CI
pytest.importorskip("torch")

import tinker
import torch
from pydantic import ValidationError

from evsys_sdk.algorithms.sft import SFT, SFTConfig


def _datum(*, weights: list[float]) -> tinker.Datum:
    """Datum with a weight mask but a dummy model_input (the batch logic
    doesn't inspect input tokens)."""
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(len(weights)))),
        loss_fn_inputs={
            "weights": tinker.TensorData.from_torch(
                torch.tensor(weights, dtype=torch.float32)
            ),
        },
    )


def _sft(datums: list[tinker.Datum], batch_size: int) -> SFT:
    """Construct an SFT and inject the tokenized datums directly, bypassing
    setup() (which needs a backend/tokenizer)."""
    algo = SFT(batch_size=batch_size, max_steps=1)
    algo._datums = datums
    algo._steps_per_epoch = max(1, len(datums) // batch_size)
    return algo


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_non_positive_batch_size():
    with pytest.raises(ValidationError):
        SFTConfig(batch_size=0)


# ---------------------------------------------------------------------------
# build_batch — slicing + wrap-around
# ---------------------------------------------------------------------------


def test_build_batch_returns_batch_size_data():
    algo = _sft([_datum(weights=[1.0]) for _ in range(10)], batch_size=3)
    batch = asyncio.run(algo.build_batch(0))
    assert len(batch.data) == 3
    assert batch.loss_fn == "cross_entropy"


def test_build_batch_advances_across_steps():
    algo = _sft([_datum(weights=[float(i)]) for i in range(10)], batch_size=2)
    b0 = asyncio.run(algo.build_batch(0))
    b1 = asyncio.run(algo.build_batch(1))
    w0 = [d.loss_fn_inputs["weights"].to_torch().tolist() for d in b0.data]
    w1 = [d.loss_fn_inputs["weights"].to_torch().tolist() for d in b1.data]
    assert w0 != w1


def test_build_batch_wraps_past_epoch_end():
    """Past one epoch the slice still returns ``batch_size`` Datums
    (wrap-around). Replaces the cookbook's _RepeatingSDFTProvider hack."""
    algo = _sft([_datum(weights=[1.0]) for _ in range(5)], batch_size=3)
    # 5 datums, batch=3: step 0 → [0,1,2]; step 1 → [3,4,0]; step 2 → [1,2,3].
    for step in range(10):
        batch = asyncio.run(algo.build_batch(step))
        assert len(batch.data) == 3


# ---------------------------------------------------------------------------
# step_metrics — train_mean_nll math
# ---------------------------------------------------------------------------


def test_step_metrics_computes_train_mean_nll_from_dict_outputs():
    """MockBackend emits dict loss_fn_outputs with raw lists — handled via
    _coerce_floats."""
    algo = _sft([_datum(weights=[1.0, 1.0, 0.0])], batch_size=1)
    batch = asyncio.run(algo.build_batch(0))

    class _Result:
        def __init__(self):
            # logprob -0.5 on masked positions, -10 on the unmasked third position
            # which the weights should ignore.
            self.loss_fn_outputs = [{"logprobs": [-0.5, -0.5, -10.0]}]

    metrics = algo.step_metrics(0, batch, _Result())
    # mean_nll = -sum(logprob * w) / sum(w); masked [0,1] → -((-0.5 + -0.5)/2) = 0.5
    assert metrics["train_mean_nll"] == pytest.approx(0.5)


def test_step_metrics_handles_tensor_data_outputs():
    """Real TinkerBackend returns TensorData — `to_torch()` is the seam."""
    algo = _sft([_datum(weights=[1.0, 1.0])], batch_size=1)
    batch = asyncio.run(algo.build_batch(0))

    class _TensorData:
        def __init__(self, vals):
            self._vals = vals
        def to_torch(self):
            return torch.tensor(self._vals, dtype=torch.float32)

    class _Result:
        def __init__(self):
            self.loss_fn_outputs = [{"logprobs": _TensorData([-0.25, -0.75])}]

    metrics = algo.step_metrics(0, batch, _Result())
    assert metrics["train_mean_nll"] == pytest.approx(0.5)


def test_step_metrics_empty_on_missing_outputs():
    """No loss_fn_outputs (e.g. backend error) → empty dict; the loop still
    lands the row via optimizer metrics, just without train_mean_nll."""
    algo = _sft([_datum(weights=[1.0])], batch_size=1)
    batch = asyncio.run(algo.build_batch(0))

    class _Result:
        loss_fn_outputs = None

    assert algo.step_metrics(0, batch, _Result()) == {}


def test_step_metrics_safe_on_token_count_mismatch():
    """Fewer logprobs than the mask (rare, on truncation) → truncate to min
    length, don't blow up the loop."""
    algo = _sft([_datum(weights=[1.0, 1.0, 1.0])], batch_size=1)
    batch = asyncio.run(algo.build_batch(0))

    class _Result:
        def __init__(self):
            self.loss_fn_outputs = [{"logprobs": [-0.5]}]   # shorter than the mask

    metrics = algo.step_metrics(0, batch, _Result())
    assert metrics["train_mean_nll"] == pytest.approx(0.5)   # min len → 1 pos → 0.5


def test_step_metrics_aggregates_across_batch():
    """Multiple Datums → weighted average of per-position logprobs by the
    per-position weights."""
    algo = _sft(
        [_datum(weights=[1.0, 1.0]), _datum(weights=[1.0, 1.0])], batch_size=2,
    )
    batch = asyncio.run(algo.build_batch(0))

    class _Result:
        def __init__(self):
            self.loss_fn_outputs = [
                {"logprobs": [-1.0, -1.0]},   # datum A
                {"logprobs": [-0.0, -0.0]},   # datum B
            ]

    metrics = algo.step_metrics(0, batch, _Result())
    # 4 masked positions, sum logprobs = -2.0 → mean_nll = 0.5
    assert metrics["train_mean_nll"] == pytest.approx(0.5)

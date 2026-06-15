"""Tests for training/batch_utils.coerce_floats + the TinkerBackend coroutine
bridge (_CoroFuture).

Both were hardened after the first real-tinker smoke of the native loop:
  * coerce_floats must flatten 2D tensors (the SDFT top-K CE datums yield
    rank-2 logprobs on real tinker; SFT's are rank-1).
  * TinkerBackend.forward_backward_async / optim_step_async wrap tinker's
    coroutine-returning *_async in _CoroFuture so the loop's "fire now, await
    .result_async() later" contract holds.
"""

from __future__ import annotations

import asyncio

import pytest

# Importing anything under evsys_sdk.training runs the package __init__, which
# pulls tinker_backend (tinker) — an optional dep not installed in base CI.
pytest.importorskip("tinker")
pytest.importorskip("torch")

from evsys_sdk.training.batch_utils import coerce_floats


# ---------------------------------------------------------------------------
# coerce_floats
# ---------------------------------------------------------------------------


def test_coerce_floats_flat_list():
    assert coerce_floats([1, 2.5, -3]) == [1.0, 2.5, -3.0]


def test_coerce_floats_none():
    assert coerce_floats(None) is None


def test_coerce_floats_flattens_nested_list():
    # SDFT top-K CE on real tinker returns rank-2 logprobs.
    assert coerce_floats([[1.0, 2.0], [3.0, 4.0]]) == [1.0, 2.0, 3.0, 4.0]


def test_coerce_floats_non_numeric_leaf_returns_none():
    assert coerce_floats(["a", "b"]) is None


def test_coerce_floats_tensor_like_2d_flattened():
    """A TensorData-like object exposing to_torch(); reshape(-1) flattens."""
    torch = pytest.importorskip("torch")

    class _TD:
        def to_torch(self):
            return torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    assert coerce_floats(_TD()) == [1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# _CoroFuture bridge (TinkerBackend)
# ---------------------------------------------------------------------------


def test_coro_future_bridges_coroutine_to_result_async():
    pytest.importorskip("tinker")
    from evsys_sdk.training.tinker_backend import _CoroFuture

    class _Future:
        async def result_async(self):
            return "RESULT"

    async def _fb_coro():
        # tinker's *_async are coroutine functions that resolve to a future
        return _Future()

    # The loop fires the call (no await), then awaits .result_async() later.
    wrapper = _CoroFuture(_fb_coro())
    assert asyncio.run(wrapper.result_async()) == "RESULT"

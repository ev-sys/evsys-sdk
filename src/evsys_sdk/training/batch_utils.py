"""Small shared helpers for turning backend forward-backward / sampling
outputs into plain Python, used by the SFT / SDFT algorithms' ``step_metrics``
and batch construction.

Kept backend-agnostic: ``MockBackend`` emits Python lists; the real
``TinkerBackend`` emits ``tinker.TensorData`` (with ``.to_torch()``) and
tinker ``SamplingResponse`` objects. These helpers normalize both.
"""

from __future__ import annotations

from typing import Any

import tinker


def coerce_floats(value: Any) -> list[float] | None:
    """Best-effort: turn a TensorData / list / torch.Tensor into list[float].

    Returns ``None`` when the value can't be coerced (so callers can skip it).
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [float(v) for v in value]
    if hasattr(value, "to_torch"):
        return [float(v) for v in value.to_torch().tolist()]
    if hasattr(value, "tolist"):
        return [float(v) for v in value.tolist()]
    return None


def extract_weights(datum: tinker.Datum) -> Any:
    """Pull the per-position weight mask out of a Datum's loss_fn_inputs."""
    inputs = getattr(datum, "loss_fn_inputs", None)
    if not inputs:
        return None
    return inputs.get("weights")


def extract_completion_tokens_from_response(response: Any) -> list[int]:
    """Pull the token-id list out of a tinker SamplingResponse-shape object.

    Real tinker exposes ``.sequences[0].tokens``; MockSamplingClient does the
    same; either way we get a list of ints back.
    """
    seqs = getattr(response, "sequences", None)
    if not seqs:
        return []
    first = seqs[0]
    tokens = getattr(first, "tokens", None) or getattr(first, "token_ids", None)
    if not tokens:
        return []
    return [int(t) for t in tokens]


__all__ = [
    "coerce_floats",
    "extract_weights",
    "extract_completion_tokens_from_response",
]

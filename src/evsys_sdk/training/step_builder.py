"""Concrete StepBuilder implementations.

Each StepBuilder owns one concern: turn a step index into a
:class:`~evsys_sdk.training.loop.TrainingBatch` and compute per-step
algorithm-specific metrics from the forward-backward result. The loop
(``evsys_sdk.training.loop.TrainingLoop``) drives everything else.

This module currently ships :class:`SFTStepBuilder`. SDFT and RL builders
land in follow-up commits in the same file so researchers can write
inheritance-based variants (``class FocalSFT(SFTStepBuilder)``) without
imports across modules.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import tinker

from .loop import TrainingBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SFT — supervised fine-tuning on pre-tokenized Datums
# ---------------------------------------------------------------------------


@dataclass
class SFTStepBuilder:
    """Cycle through a pre-tokenized dataset, ``batch_size`` Datums per step.

    Parameters
    ----------
    datums:
        Output of :func:`evsys_sdk.training.sft_data.sft_tokenize` — list of
        :class:`tinker.Datum` with ``target_tokens`` + ``weights`` masks on
        the assistant span.
    batch_size:
        Datums per training step.
    seed:
        Shuffle seed. ``None`` keeps the row order (useful for parity with
        the cookbook which is deterministic-by-default).

    The builder wraps modulo dataset length so ``num_steps`` can exceed one
    epoch — this replaces the ``_RepeatingSDFTProvider`` hack the old
    cookbook wrapper needed (see :mod:`evsys_sdk.algorithms.tinker_sdft`).
    """

    datums: list[tinker.Datum]
    batch_size: int
    seed: int | None = None
    # populated lazily, on first build_batch — keeps the dataclass-with-default
    # invariant intact while allowing in-place shuffling.
    _order: list[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.datums:
            raise ValueError("SFTStepBuilder: datums is empty")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0 (got {self.batch_size})")
        self._order = list(range(len(self.datums)))
        if self.seed is not None:
            rng = random.Random(self.seed)
            rng.shuffle(self._order)

    @property
    def steps_per_epoch(self) -> int:
        """How many full batches fit in one epoch."""
        return max(1, len(self.datums) // self.batch_size)

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        """Slice ``batch_size`` Datums for ``step_idx``, wrapping around the
        dataset when needed."""
        n = len(self.datums)
        start = (step_idx * self.batch_size) % n
        end = start + self.batch_size
        # Wrap when the slice straddles the end.
        if end <= n:
            indices = self._order[start:end]
        else:
            indices = self._order[start:] + self._order[: end - n]
        data = [self.datums[i] for i in indices]
        return TrainingBatch(data=data, loss_fn="cross_entropy")

    def step_metrics(
        self,
        step_idx: int,
        batch: TrainingBatch,
        fb_result: Any,
    ) -> dict[str, float]:
        """Compute ``train_mean_nll`` from the per-position logprobs of
        each Datum, weighted by the loss mask.

        Tinker's cross_entropy loss returns ``loss_fn_outputs[i]["logprobs"]``:
        a per-position vector of log-probabilities of the target token (so
        a "perfect" prediction has logprob 0; otherwise it's negative).
        The mean NLL is ``-sum(logprob * weight) / sum(weight)`` over the
        loss-mask positions, averaged across the batch.
        """
        outputs = getattr(fb_result, "loss_fn_outputs", None)
        if not outputs:
            return {}

        total_logprob = 0.0
        total_weight = 0.0
        for datum, out in zip(batch.data, outputs):
            logprobs = _coerce_floats(out.get("logprobs") if isinstance(out, dict)
                                      else getattr(out, "logprobs", None))
            if logprobs is None:
                continue
            weights = _coerce_floats(_extract_weights(datum))
            if weights is None or len(weights) == 0:
                continue
            # Truncate to min length so we don't blow up on a token-count mismatch
            # between the per-position logprobs and the per-position mask.
            k = min(len(logprobs), len(weights))
            for j in range(k):
                total_logprob += logprobs[j] * weights[j]
                total_weight += weights[j]

        if total_weight <= 0:
            return {}
        return {"train_mean_nll": -float(total_logprob) / float(total_weight)}


# ---------------------------------------------------------------------------
# Helpers shared across step builders
# ---------------------------------------------------------------------------


def _coerce_floats(value: Any) -> list[float] | None:
    """Best-effort: turn a TensorData / list / torch.Tensor into list[float].

    The MockBackend emits Python lists; the real TinkerBackend emits
    ``tinker.TensorData`` (which exposes ``.to_torch()``). We handle both
    here so step_metrics works against either.
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


def _extract_weights(datum: tinker.Datum) -> Any:
    """Pull the per-position weight mask out of a Datum's loss_fn_inputs."""
    inputs = getattr(datum, "loss_fn_inputs", None)
    if not inputs:
        return None
    return inputs.get("weights")


__all__ = ["SFTStepBuilder"]

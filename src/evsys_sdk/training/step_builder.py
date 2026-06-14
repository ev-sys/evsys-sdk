"""RL step builder + SDFT data holders + shared datum-metric helpers.

A StepBuilder turns a step index into a
:class:`~evsys_sdk.training.loop.TrainingBatch` and computes per-step
algorithm-specific metrics; the loop drives everything else. SFT's and SDFT's
batch logic now live directly on their algorithms (the algorithm IS its own
step builder via :class:`~evsys_sdk.algorithms.base.BaseAlgorithm`); RL is
mid-migration to the same shape. The SDFT *data holders* (``SDFTDataset`` /
``SimpleSDFTDataset``) and the shared helpers stay here until RL moves over.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, Sequence, runtime_checkable

import tinker

from ..data_types import PromptExample
from .loop import TrainingBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers shared across step builders + algorithms
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


def _extract_completion_tokens_from_response(response: Any) -> list[int]:
    """Pull the token-id list out of a tinker SamplingResponse-shape object.

    Real tinker exposes ``.sequences[0].tokens``; MockSamplingClient does
    the same; either way we get a list of ints back.
    """
    seqs = getattr(response, "sequences", None)
    if not seqs:
        return []
    first = seqs[0]
    tokens = getattr(first, "tokens", None) or getattr(first, "token_ids", None)
    if not tokens:
        return []
    return [int(t) for t in tokens]


__all__ = ["RLDataset", "RLStepBuilder", "SDFTDataset", "SimpleSDFTDataset"]


# ---------------------------------------------------------------------------
# SDFT data holders (consumed by the SDFT algorithm)
# ---------------------------------------------------------------------------


@runtime_checkable
class SDFTDataset(Protocol):
    """The data interface the SDFT algorithm consumes per step.

    Per the SDFT paper, each step needs ``batch_size`` ``(question, golden_answer)``
    pairs — the student rolls out on the question, the teacher scores
    teacher-forced through the question+golden_answer demo.
    """

    def __len__(self) -> int: ...

    def get_batch(self, step_idx: int) -> tuple[list[str], list[str]]:
        """Return ``(questions, golden_answers)`` of length ``batch_size``."""
        ...


@dataclass
class SimpleSDFTDataset:
    """Stock :class:`SDFTDataset` over :class:`~evsys_sdk.data_types.PromptExample`
    rows: the question lives in ``inputs['question']`` and the gold answer in
    ``expected``. Wraps modulo dataset length so the loop can exceed one epoch
    (no ``_RepeatingSDFTProvider`` hack needed)."""

    rows: list[PromptExample]
    batch_size: int

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("SimpleSDFTDataset: rows is empty")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0 (got {self.batch_size})")
        missing = [
            i for i, r in enumerate(self.rows[:5])
            if not r.inputs.get("question") or r.expected is None
        ]
        if missing:
            raise ValueError(
                f"SimpleSDFTDataset: rows need inputs['question'] + expected "
                f"(indices {missing} of first 5)"
            )

    def __len__(self) -> int:
        return max(1, len(self.rows) // self.batch_size)

    def get_batch(self, step_idx: int) -> tuple[list[str], list[str]]:
        n = len(self.rows)
        start = (step_idx * self.batch_size) % n
        end = start + self.batch_size
        if end <= n:
            slice_ = self.rows[start:end]
        else:
            slice_ = self.rows[start:] + self.rows[: end - n]
        return (
            [r.inputs["question"] for r in slice_],
            [str(r.expected) for r in slice_],
        )


SamplerProvider = Callable[[], Awaitable[Any]]
"""Closure that returns the latest student sampling client.

Called at the start of each ``build_batch`` so the on-policy rollout uses the
current weights. The composer binds it to ``backend.snapshot_sampling_client``
— matches the cookbook's ``save_checkpoint_and_get_sampling_client`` pattern
but as an injectable seam so tests can use a static mock client.
"""


# ---------------------------------------------------------------------------
# RL — on-policy rollout + advantages + IS loss
# ---------------------------------------------------------------------------


@runtime_checkable
class RLDataset(Protocol):
    """Per-step ``Sequence[EnvGroupBuilder]`` source for :class:`RLStepBuilder`.

    The cookbook's ``RLDataset`` Protocol has the same shape — one
    ``EnvGroupBuilder`` per batch slot, ``batch_size`` builders per step.
    """

    def __len__(self) -> int: ...

    def get_batch(self, step_idx: int) -> Sequence[Any]:
        """Return ``batch_size`` :class:`~evsys_sdk.training.env.EnvGroupBuilder` instances."""
        ...


@dataclass
class RLStepBuilder:
    """On-policy rollout → group-normalized advantages → IS loss.

    Parameters
    ----------
    dataset:
        Returns ``batch_size`` :class:`~evsys_sdk.training.env.EnvGroupBuilder`
        instances per step.
    student_sampler_provider:
        Async callable returning the latest student sampler (same shape as the
        SDFT algorithm's provider — composer binds it to
        ``backend.snapshot_sampling_client``).
    num_samples:
        Trajectories per ``EnvGroupBuilder`` (cookbook calls this
        ``group_size``). >= 2 lets the advantage baseline subtract a
        within-group mean.
    drop_constant_reward:
        When True, groups whose rewards are all equal contribute no
        gradient under IS, so we drop them before training. Matches the
        cookbook's ``do_group_rollout_and_filter_constant_reward``.
    """

    dataset: "RLDataset"
    student_sampler_provider: SamplerProvider
    num_samples: int = 1
    max_tokens: int = 256
    temperature: float = 1.0
    drop_constant_reward: bool = False

    @property
    def steps_per_epoch(self) -> int:
        return len(self.dataset)

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        from .data_processing import (
            assemble_training_data,
            compute_advantages,
            compute_trajectory_metrics,
        )
        from .rollouts import do_group_rollouts

        builders = list(self.dataset.get_batch(step_idx))
        sampler = await self.student_sampler_provider()
        groups = await do_group_rollouts(
            sampler=sampler, builders=builders,
            num_samples=self.num_samples,
            max_tokens=self.max_tokens, temperature=self.temperature,
            drop_constant_reward=self.drop_constant_reward,
        )
        if not groups:
            # No usable groups — emit an empty batch + a zeroed metric row so
            # the loop's step counter advances cleanly.
            return TrainingBatch(
                data=[], loss_fn="importance_sampling",
                metrics={"reward/n_trajectories": 0.0},
            )

        advantages = compute_advantages(groups)
        datums, _meta = assemble_training_data(groups, advantages)
        metrics = compute_trajectory_metrics(groups)
        return TrainingBatch(
            data=datums, loss_fn="importance_sampling", metrics=metrics,
        )

    def step_metrics(
        self, step_idx: int, batch: TrainingBatch, fb_result: Any,
    ) -> dict[str, float]:
        # Reward stats already merged from batch.metrics by the loop; nothing
        # to add from the fb_result for IS (the loss is summed server-side and
        # bubbled through optim_result.metrics anyway).
        return {}

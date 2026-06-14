"""Rollout trajectory data model.

The SDK's training-side rollout shape: a prompt + a sampled completion (with
per-token logprobs) + a scalar reward. Produced by the harbor rollout engine
(:mod:`evsys_sdk.training.harbor_engine`) and consumed by the IS-loss data
prep (:mod:`evsys_sdk.training.data_processing`).

This is the *training* model (what the optimizer needs). Harbor's own
``RolloutDetail`` (ATIF: per-turn token ids + logprobs) is the richer
agent-rollout interchange shape; the harbor engine maps it into this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tinker


@dataclass
class Trajectory:
    """One rollout: prompt → completion (+ per-token logprobs) → reward."""

    prompt: tinker.ModelInput
    completion_tokens: list[int]
    completion_logprobs: list[float]
    """Per-token logprobs from the sampler (one per ``completion_tokens``
    entry). Used to construct the importance-sampling loss."""
    reward: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryGroup:
    """All rollouts sampled from one task (``num_samples`` of them). The
    group-relative advantage baseline subtracts the within-group mean reward."""

    trajectories: list[Trajectory]
    tags: list[str] = field(default_factory=list)
    """Logging tags (toolkit, task category, …) — propagated by
    :func:`~evsys_sdk.training.data_processing.compute_advantages` /
    ``assemble_training_data`` so metric breakdowns work."""

    @property
    def rewards(self) -> list[float]:
        return [t.reward for t in self.trajectories]


__all__ = ["Trajectory", "TrajectoryGroup"]

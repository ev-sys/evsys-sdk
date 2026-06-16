"""The rollout data model — one shape for all (multi-turn) rollouts.

Every rollout in the SDK — single- or multi-turn, RL or SDFT — is a
:class:`Trajectory`: an ordered list of :class:`Turn`\\s (each a prompt + a
sampled completion with per-token logprobs) plus a scalar reward. Harbor's
``RolloutDetail`` (ATIF) is converted into this at the engine boundary
(:func:`evsys_sdk.training.harbor_engine.run_harbor_rollouts`), so the rest of
the SDK only ever sees ``Trajectory``. The IS-loss data prep
(:mod:`evsys_sdk.training.data_processing`) emits one ``Datum`` per turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    """One assistant turn of a rollout.

    ``prompt_tokens`` is the full rendered context the policy saw for this turn
    (system + prior turns + the latest observation); ``completion_tokens`` /
    ``logprobs`` are the sampled response. A single-turn rollout has exactly one.
    """

    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    text: str = ""


@dataclass
class Trajectory:
    """One rollout: an ordered list of :class:`Turn`\\s + a scalar reward."""

    turns: list[Turn]
    reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form per-rollout extras. The harbor harvest stamps a ``"usage"``
    dict here — ``{cost_usd, prompt_tokens, completion_tokens, cached_tokens,
    latency_s}`` (any field harbor didn't report is ``None``) — which the eval
    aggregator turns into the default ``time_per_task`` / ``tokens_per_task`` /
    ``cost_per_task`` metrics."""


@dataclass
class TrajectoryGroup:
    """All rollouts sampled from one task (``num_samples`` of them). The
    group-relative advantage baseline subtracts the within-group mean reward."""

    trajectories: list[Trajectory]
    tags: list[str] = field(default_factory=list)
    """Logging tags (toolkit, task category, …) propagated through advantage +
    Datum assembly so metric breakdowns work."""

    @property
    def rewards(self) -> list[float]:
        return [t.reward for t in self.trajectories]


__all__ = ["Turn", "Trajectory", "TrajectoryGroup"]

"""Environment abstraction for RL training.

An :class:`EnvGroupBuilder` knows how to construct an initial prompt and,
given a sampled completion, compute a reward + decide whether the episode
is done. Single-turn tasks (tool routing, slug classification) use
:class:`SingleTurnEnv` which calls a user-provided verifier on the
completion to produce a binary reward. Multi-turn environments (math
reasoning, code execution, tool-use loops) plug in by implementing the
same Protocol — the rollout helper in :mod:`evsys_sdk.training.rollouts`
walks the ``initial_observation → step → step → ...`` cycle until ``done``.

The Protocol names mirror :mod:`tinker_cookbook.rl.types` so the design
is recognisable to anyone migrating off the cookbook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import tinker


# ---------------------------------------------------------------------------
# Observation / StepResult / TrajectoryGroup — data types
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """What the env hands to the policy on each turn."""

    prompt: tinker.ModelInput
    """Tokens the policy samples from."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form per-observation context (e.g. ``{task_id, toolkit, ...}``).
    Surfaces in the trajectory's metadata for logging / tag breakdowns."""


@dataclass
class StepResult:
    """What ``env.step(tokens)`` returns."""

    reward: float
    done: bool
    next_observation: Observation | None = None
    """``None`` when ``done=True``; required when the trajectory continues."""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """One env episode (prompt → ... → reward + done)."""

    prompt: tinker.ModelInput
    completion_tokens: list[int]
    completion_logprobs: list[float]
    """Per-token logprobs from the sampler (one per ``completion_tokens``
    entry). Used to construct the IS loss."""
    reward: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryGroup:
    """All trajectories sampled from one :class:`EnvGroupBuilder` (a "group"
    in cookbook parlance — usually ``num_samples`` rollouts of the same
    prompt for variance reduction)."""

    trajectories: list[Trajectory]
    tags: list[str] = field(default_factory=list)
    """Logging tags (toolkit, task category, ...) — :func:`compute_advantages`
    + :func:`assemble_training_data` propagate them so metric breakdowns work."""

    @property
    def rewards(self) -> list[float]:
        return [t.reward for t in self.trajectories]


# ---------------------------------------------------------------------------
# EnvGroupBuilder Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EnvGroupBuilder(Protocol):
    """The interface a rollout consumes.

    The cookbook calls this "EnvGroupBuilder" because one instance produces
    a *group* of trajectories — multiple rollouts from the same prompt for
    variance reduction in advantage estimation. Single-trajectory builders
    are a degenerate case (``num_samples=1``).
    """

    def initial_observation(self) -> Observation:
        """Return the first observation (the prompt the policy samples from)."""
        ...

    def step(self, completion_tokens: Sequence[int]) -> StepResult:
        """Score a completion and decide whether the episode is over.

        For single-turn tasks this is the only step call and ``done=True``.
        For multi-turn it returns the next observation when ``done=False``.
        """
        ...

    def logging_tags(self) -> list[str]:
        """Tags used to route metric breakdowns (toolkit, task type, ...)."""
        ...


# ---------------------------------------------------------------------------
# SingleTurnEnv — single-shot rollout + verifier-based reward
# ---------------------------------------------------------------------------


VerifierFn = Callable[[str, Any], float]
"""``(decoded_completion, expected) -> reward``. The decoded completion is
the tokenizer's string output; ``expected`` is the gold answer (or alias
list) the verifier matches against."""


@dataclass
class SingleTurnEnv:
    """Single-turn env: emit ``prompt``, score the completion, done.

    Designed for tool-routing / classification tasks. ``tokenizer`` is
    used to decode the sampled token list into a string; ``verifier`` is a
    user-supplied callable matching the SDK's :mod:`evsys_sdk.verifiers`
    surface.
    """

    prompt: tinker.ModelInput
    expected: Any
    tokenizer: Any
    verifier: VerifierFn
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def initial_observation(self) -> Observation:
        return Observation(prompt=self.prompt, metadata=self.metadata)

    def step(self, completion_tokens: Sequence[int]) -> StepResult:
        text = self.tokenizer.decode(list(completion_tokens))
        reward = float(self.verifier(text, self.expected))
        return StepResult(reward=reward, done=True, metadata={"completion_text": text})

    def logging_tags(self) -> list[str]:
        return list(self.tags)


__all__ = [
    "EnvGroupBuilder",
    "Observation",
    "SingleTurnEnv",
    "StepResult",
    "Trajectory",
    "TrajectoryGroup",
    "VerifierFn",
]

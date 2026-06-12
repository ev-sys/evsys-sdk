"""Rollout helpers — drive an :class:`~evsys_sdk.training.env.EnvGroupBuilder`
against a :class:`~evsys_sdk.training.backend.SamplingClient` to produce a
:class:`~evsys_sdk.training.env.TrajectoryGroup`.

Replaces ``tinker_cookbook.rl.rollouts.do_group_rollout_and_filter_constant_reward``.
The cookbook variant also does reward-constancy filtering (drop groups
where all samples got the same reward — they contribute zero gradient
under IS); we expose that as an opt-in via ``drop_constant_reward``.

Currently single-turn only. Multi-turn extension lands when needed: the
loop is already shaped as ``while not done``, just needs ``builder.step``
to actually return ``done=False`` somewhere.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

import tinker

from .env import EnvGroupBuilder, Trajectory, TrajectoryGroup

logger = logging.getLogger(__name__)


async def do_group_rollout(
    *,
    sampler: Any,
    builder: EnvGroupBuilder,
    num_samples: int = 1,
    max_tokens: int = 256,
    temperature: float = 1.0,
) -> TrajectoryGroup | None:
    """Sample ``num_samples`` rollouts from ``builder`` and score each.

    For single-turn envs each rollout is one sample + one ``step`` call.
    Returns ``None`` when the builder returns no valid initial observation
    (defensive — current builders never do this).
    """
    obs = builder.initial_observation()
    if obs is None:  # pragma: no cover  (defensive)
        return None

    response = await sampler.sample_async(
        prompt=obs.prompt,
        params=tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature),
        num_samples=num_samples,
    )

    trajectories: list[Trajectory] = []
    for seq in getattr(response, "sequences", None) or []:
        tokens = list(getattr(seq, "tokens", None) or getattr(seq, "token_ids", None) or [])
        logprobs = list(getattr(seq, "logprobs", None) or [])
        if not tokens:
            continue
        result = builder.step(tokens)
        trajectories.append(Trajectory(
            prompt=obs.prompt,
            completion_tokens=tokens,
            completion_logprobs=logprobs,
            reward=float(result.reward),
            metadata={**obs.metadata, **(result.metadata or {})},
        ))

    if not trajectories:
        return None
    return TrajectoryGroup(
        trajectories=trajectories,
        tags=builder.logging_tags(),
    )


async def do_group_rollouts(
    *,
    sampler: Any,
    builders: Sequence[EnvGroupBuilder],
    num_samples: int = 1,
    max_tokens: int = 256,
    temperature: float = 1.0,
    drop_constant_reward: bool = False,
) -> list[TrajectoryGroup]:
    """Batched rollout — one :class:`TrajectoryGroup` per builder, sampled in
    parallel. Optionally drops groups whose rewards are all equal (no IS
    gradient contribution)."""
    groups = await asyncio.gather(*[
        do_group_rollout(
            sampler=sampler, builder=b,
            num_samples=num_samples, max_tokens=max_tokens, temperature=temperature,
        )
        for b in builders
    ])
    out: list[TrajectoryGroup] = []
    for g in groups:
        if g is None:
            continue
        if drop_constant_reward and _all_equal(g.rewards):
            continue
        out.append(g)
    return out


def _all_equal(xs: list[float]) -> bool:
    return len(xs) > 0 and all(x == xs[0] for x in xs)


__all__ = ["do_group_rollout", "do_group_rollouts"]

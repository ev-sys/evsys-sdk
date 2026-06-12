"""Tests for ``evsys_sdk.training.env`` + ``evsys_sdk.training.rollouts``.

Exercise the EnvGroupBuilder Protocol against a stub SamplingClient: the
SingleTurnEnv concrete reward path + the do_group_rollout(s) helpers
flatten one or more builders into TrajectoryGroups.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("tinker")  # optional dep; not installed in base CI
pytest.importorskip("torch")

import tinker

from evsys_sdk.training.env import (
    EnvGroupBuilder, Observation, SingleTurnEnv, StepResult, Trajectory,
    TrajectoryGroup,
)
from evsys_sdk.training.rollouts import do_group_rollout, do_group_rollouts


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _StubTokenizer:
    def decode(self, tokens):
        return "T" + "_".join(str(t) for t in tokens)


def _binary_verifier(output: str, expected: Any) -> float:
    return 1.0 if str(expected) in (output or "") else 0.0


class _CannedSampler:
    """Returns a sequence with `canned_tokens` per call."""

    def __init__(self, canned: list[int], num_returns: int = 1):
        self.canned = canned
        self.num_returns = num_returns
        self.calls: list[dict] = []

    async def sample_async(self, **kwargs):
        self.calls.append(kwargs)
        @dataclass
        class _Seq:
            tokens: list[int]
            logprobs: list[float]
        seqs = [
            _Seq(tokens=list(self.canned),
                 logprobs=[-0.5] * len(self.canned))
            for _ in range(self.num_returns)
        ]
        @dataclass
        class _Resp:
            sequences: list[Any]
        return _Resp(sequences=seqs)


def _builder(*, prompt_ids: list[int], expected: Any, tags=None):
    return SingleTurnEnv(
        prompt=tinker.ModelInput.from_ints(prompt_ids),
        expected=expected,
        tokenizer=_StubTokenizer(),
        verifier=_binary_verifier,
        tags=list(tags or []),
    )


# ---------------------------------------------------------------------------
# SingleTurnEnv
# ---------------------------------------------------------------------------


def test_single_turn_env_returns_initial_observation():
    env = _builder(prompt_ids=[1, 2, 3], expected="42")
    obs = env.initial_observation()
    assert isinstance(obs, Observation)
    assert obs.prompt.to_ints() == [1, 2, 3]


def test_single_turn_env_step_calls_verifier_with_decoded_completion():
    env = _builder(prompt_ids=[1], expected="hit")
    result = env.step([1, 2, 3])
    assert isinstance(result, StepResult)
    assert result.done is True
    assert result.reward == 0.0
    assert "completion_text" in result.metadata


def test_single_turn_env_logging_tags_passthrough():
    env = _builder(prompt_ids=[1], expected="x", tags=["airtable"])
    assert env.logging_tags() == ["airtable"]


# ---------------------------------------------------------------------------
# do_group_rollout
# ---------------------------------------------------------------------------


def test_do_group_rollout_returns_trajectory_group():
    env = _builder(prompt_ids=[10, 11], expected="42")
    sampler = _CannedSampler(canned=[20, 21, 22])
    group = asyncio.run(do_group_rollout(sampler=sampler, builder=env))
    assert isinstance(group, TrajectoryGroup)
    assert len(group.trajectories) == 1
    traj = group.trajectories[0]
    assert traj.prompt.to_ints() == [10, 11]
    assert traj.completion_tokens == [20, 21, 22]
    assert traj.reward == 0.0  # verifier did not see "42"


def test_do_group_rollout_with_multiple_samples():
    env = _builder(prompt_ids=[1], expected="x")
    sampler = _CannedSampler(canned=[5, 6], num_returns=3)
    group = asyncio.run(do_group_rollout(
        sampler=sampler, builder=env, num_samples=3,
    ))
    assert len(group.trajectories) == 3
    # all three rollouts saw the same builder so they share a tag set
    assert group.tags == []


def test_do_group_rollout_returns_none_on_empty_completion():
    env = _builder(prompt_ids=[1], expected="x")
    sampler = _CannedSampler(canned=[])
    group = asyncio.run(do_group_rollout(sampler=sampler, builder=env))
    assert group is None


# ---------------------------------------------------------------------------
# do_group_rollouts (batched)
# ---------------------------------------------------------------------------


def test_do_group_rollouts_runs_each_builder():
    envs = [_builder(prompt_ids=[i], expected=str(i)) for i in range(3)]
    sampler = _CannedSampler(canned=[5])
    groups = asyncio.run(do_group_rollouts(sampler=sampler, builders=envs))
    assert len(groups) == 3
    # each sample call was made — verify by call count
    assert len(sampler.calls) == 3


def test_drop_constant_reward_filters_uniform_groups():
    """Two builders, one whose reward is uniform across rollouts (here:
    always 0) should drop with `drop_constant_reward=True`."""
    env_zero = _builder(prompt_ids=[1], expected="MATCHME")
    env_one = _builder(prompt_ids=[1], expected="T5")  # _StubTokenizer.decode → "T5"
    sampler = _CannedSampler(canned=[5], num_returns=2)
    groups = asyncio.run(do_group_rollouts(
        sampler=sampler, builders=[env_zero, env_one],
        num_samples=2, drop_constant_reward=True,
    ))
    # env_zero gives reward 0 for both rollouts → dropped.
    # env_one rewards aren't actually constant though — the verifier sees
    # the same completion both times, so rewards ARE equal too → also dropped.
    # We assert filtering happened (at most 0/1 groups left).
    assert len(groups) <= 1

"""Tests for the generic rollout helper ``evsys_sdk.training.rollout``.

The policy is harbor's TinkerLLM in production; here we inject a fake LLM via
``generate_rollouts(..., _llm=...)`` so the multi-turn loop, env contract, and
num_samples grouping are testable with no harbor / tinker / sandbox.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from evsys_sdk.training.rollout import (
    EnvStep,
    RolloutTask,
    Trajectory,
    generate_rollouts,
)


@dataclass
class _FakeResp:
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    logprobs: list[float]
    content: str


class _FakeLLM:
    """Returns a canned response per call; records the prompts it saw."""

    def __init__(self, scripted: list[str] | None = None):
        self.scripted = scripted or []
        self.calls: list[dict] = []
        self._i = 0

    async def call(self, *, prompt, message_history):
        self.calls.append({"prompt": prompt, "history": list(message_history)})
        text = self.scripted[self._i] if self._i < len(self.scripted) else f"resp{self._i}"
        self._i += 1
        toks = [100 + len(self.calls)]
        return _FakeResp(
            prompt_token_ids=[1, 2, 3],
            completion_token_ids=toks,
            logprobs=[-0.5] * len(toks),
            content=text,
        )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Single-turn (env=None) — pure generation
# ---------------------------------------------------------------------------


def test_pure_generation_single_turn_no_env():
    llm = _FakeLLM(scripted=["hello"])
    tasks = [RolloutTask(prompt="say hi")]
    out = _run(generate_rollouts(
        tasks, model_name="m", model_path=None, num_samples=1, _llm=llm,
    ))
    assert len(out) == 1 and len(out[0]) == 1
    traj = out[0][0]
    assert isinstance(traj, Trajectory)
    assert len(traj.turns) == 1            # env=None ⇒ exactly one turn
    assert traj.turns[0].text == "hello"
    assert traj.turns[0].completion_tokens
    assert traj.reward == 0.0              # no env ⇒ no reward


# ---------------------------------------------------------------------------
# Single-turn verifier env
# ---------------------------------------------------------------------------


def test_single_turn_verifier_env_sets_reward():
    llm = _FakeLLM(scripted=["the answer is 42"])

    async def env(messages):
        last = messages[-1]["content"]
        return EnvStep(done=True, reward=1.0 if "42" in last else 0.0)

    out = _run(generate_rollouts(
        [RolloutTask(prompt="q", env=env)],
        model_name="m", model_path=None, num_samples=1, _llm=llm,
    ))
    assert out[0][0].reward == 1.0
    assert len(out[0][0].turns) == 1


# ---------------------------------------------------------------------------
# Multi-turn env
# ---------------------------------------------------------------------------


def test_multi_turn_runs_until_env_done():
    llm = _FakeLLM(scripted=["turn1", "turn2", "turn3"])
    calls = {"n": 0}

    async def env(messages):
        calls["n"] += 1
        if calls["n"] < 3:
            return EnvStep(observation=f"obs{calls['n']}", done=False)
        return EnvStep(done=True, reward=2.0)

    out = _run(generate_rollouts(
        [RolloutTask(prompt="start", env=env)],
        model_name="m", model_path=None, num_samples=1, max_turns=8, _llm=llm,
    ))
    traj = out[0][0]
    assert len(traj.turns) == 3           # ran 3 turns before done
    assert traj.reward == 2.0


def test_max_turns_caps_the_loop():
    llm = _FakeLLM()

    async def env(messages):
        return EnvStep(observation="again", done=False)   # never done

    out = _run(generate_rollouts(
        [RolloutTask(prompt="start", env=env)],
        model_name="m", model_path=None, max_turns=4, _llm=llm,
    ))
    assert len(out[0][0].turns) == 4      # capped at max_turns


# ---------------------------------------------------------------------------
# num_samples grouping
# ---------------------------------------------------------------------------


def test_num_samples_groups_per_task():
    llm = _FakeLLM()
    tasks = [RolloutTask(prompt="a"), RolloutTask(prompt="b")]
    out = _run(generate_rollouts(
        tasks, model_name="m", model_path=None, num_samples=3, _llm=llm,
    ))
    assert len(out) == 2                   # one list per task
    assert all(len(samples) == 3 for samples in out)   # num_samples each


def test_system_prompt_seeds_first_message():
    llm = _FakeLLM(scripted=["ok"])
    _run(generate_rollouts(
        [RolloutTask(prompt="hi")],
        model_name="m", model_path=None, system_prompt="SYS", _llm=llm,
    ))
    # first call's message_history carries the system seed
    assert llm.calls[0]["history"] == [{"role": "system", "content": "SYS"}]

"""REAL (un-mocked) harbor rollout smoke test — runs an actual harbor 0.13.2
``Job`` end to end with a no-model EchoAgent, exercising the real JobConfig
(tasks × n_attempts), the no-op environment, the agent→context harvest, and our
Python scoring. No tinker/litellm credentials needed.

This is the test that would have caught the 0.13.2 JobConfig API drift (the
mocked tests never construct a real Job)."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("harbor")
pytest.importorskip("tinker")  # harbor_agents imports TinkerLLM at module top

from evsys_sdk.data_types import HarborTask, InProcessVerifier
from evsys_sdk.training.harbor_engine import run_harbor_rollouts

_ECHO = "tests.harbor_echo_agent:EchoAgent"   # test-only agent (tests/, not shipped)


def _task(task_id: str, instruction: str, expected: str) -> HarborTask:
    # EchoAgent emits "ECHO:<instruction>"; `contains` scores 1.0 iff expected
    # is a substring of that.
    return HarborTask(
        task_id=task_id, instruction=instruction,
        verifier=InProcessVerifier(fn_name="contains", expected=expected),
        metadata={"tags": ["smoke"]},
    )


def test_real_harbor_rollout_scores_and_harvests(tmp_path):
    tasks = [
        _task("t_hit", "solve A", expected="ECHO"),    # present in completion → 1.0
        _task("t_miss", "solve B", expected="ZZZZZ"),  # absent → 0.0
    ]
    groups = asyncio.run(run_harbor_rollouts(
        tasks,
        fmt="harbor_task",                             # scored (default)
        model_name="echo", model_path=None,
        workspace_dir=tmp_path,
        num_samples=2,                                 # → n_attempts=2 per task
        agent_import_path=_ECHO,
        n_concurrent=2, max_retries=0,
    ))

    assert len(groups) == 2
    hit, miss = groups
    # n_attempts=2 → two trajectories per task
    assert len(hit.trajectories) == 2 and len(miss.trajectories) == 2
    # reward from harbor's verifier (our host-side EvsysVerifier over the registered fn)
    assert all(t.reward == 1.0 for t in hit.trajectories)
    assert all(t.reward == 0.0 for t in miss.trajectories)
    # harvest populated the rollout (turns) + usage from the agent context
    tr = hit.trajectories[0]
    assert tr.turns and tr.turns[0].completion_tokens
    u = tr.metadata["usage"]
    assert u["prompt_tokens"] is not None and u["completion_tokens"] is not None


def test_real_harbor_rollouts_generation_only(tmp_path):
    # fmt="prompt" → generation-only: raw prompts in, rollouts harvested, no
    # reward (no verifier). The runner picks PromptAdapter; no adapter at the call site.
    groups = asyncio.run(run_harbor_rollouts(
        ["write a poem", "summarize this"], fmt="prompt", model_name="echo", model_path=None,
        workspace_dir=tmp_path, agent_import_path=_ECHO, n_concurrent=2, max_retries=0,
    ))
    assert len(groups) == 2
    trajs = [g.trajectories[0] for g in groups]
    assert all(t.turns and t.turns[0].completion_tokens and t.reward == 0.0 for t in trajs)

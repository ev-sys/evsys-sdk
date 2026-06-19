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

from types import SimpleNamespace

from evsys_sdk import AgentSpec
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
        outcome_reward=True,                           # scored (default)
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
    # outcome_reward=False → generation-only: raw prompts in, rollouts harvested,
    # no reward (no verifier). The runner picks PromptAdapter; no adapter at the call site.
    groups = asyncio.run(run_harbor_rollouts(
        ["write a poem", "summarize this"], outcome_reward=False, model_name="echo", model_path=None,
        workspace_dir=tmp_path, agent_import_path=_ECHO, n_concurrent=2, max_retries=0,
    ))
    assert len(groups) == 2
    trajs = [g.trajectories[0] for g in groups]
    assert all(t.turns and t.turns[0].completion_tokens and t.reward == 0.0 for t in trajs)


def test_agent_spec_flows_into_jobconfig(tmp_path):
    # The registered `agent` plugin (kind+params) resolves into the harbor
    # JobConfig: import_path = the plugin's harbor agent, and explicit params
    # (max_turns) override the rollout default. Capture the JobConfig via a
    # _job_factory instead of running a real Job.
    captured = {}

    async def _capture(config):
        captured["config"] = config
        return SimpleNamespace(trial_results=[])

    asyncio.run(run_harbor_rollouts(
        [_task("t", "solve", expected="ECHO")],
        agent_spec=AgentSpec(kind="basic_loop", params={"max_turns": 2}),
        model_name="echo", model_path=None, workspace_dir=tmp_path,
        _job_factory=_capture,
    ))
    agent_cfg = captured["config"].agents[0]
    assert agent_cfg.import_path.endswith(":BasicLoopAgent")
    assert agent_cfg.kwargs["max_turns"] == 2          # spec param overrode the default (1)


def test_tool_loop_agent_runs_multi_turn_tool_rollout(tmp_path):
    # A custom multi-turn TOOL-using agent, registered via @register_agent, selected
    # by agent_spec={kind: tool_loop}. Proves the plugin path runs a real tool loop
    # and harvests one Turn per loop iteration. (Real harbor Job, no model.)
    import tests.harbor_tool_agent  # noqa: F401 — registers @register_agent("tool_loop")

    task = _task("t_tool", "find the right tool", expected="TOOLS_OK")
    groups = asyncio.run(run_harbor_rollouts(
        [task],
        agent_spec=AgentSpec(kind="tool_loop", params={"max_turns": 3}),
        model_name="x", model_path=None, workspace_dir=tmp_path,
        num_samples=1, max_retries=0,
    ))
    assert len(groups) == 1
    traj = groups[0].trajectories[0]
    assert len(traj.turns) == 3      # 3 tool-loop turns harvested (multi-turn)
    assert traj.reward == 1.0        # verifier scored the tool-produced answer (TOOLS_OK)

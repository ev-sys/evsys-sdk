"""The agent extension point — registration, mission/argv parity with the
pre-refactor builders, the environment field, and the spec factory.

Parity is asserted against FROZEN copies of the old argv assembly (the exact
code that used to live inline in ``triggers/agent.py::build_command`` and
``triggers/remote.py``), so a drift in the new classes fails loudly here even
though the wrappers now delegate to them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from evsys_sdk.agents import (
    AutoresearchAgent,
    EvsysAgent,
    TriggerAgent,
    build_agent,
)
from evsys_sdk.agents.base import (
    AUTORESEARCH_OFF,
    AUTORESEARCH_ON,
    DEFAULT_PROMPT,
    DISTILL_PROMPT,
    REMOTE_AUTORESEARCH_PROMPT,
)
from evsys_sdk.config import AgentSpec, RemoteAgentConfig, SandboxSpec, TriggerAgentConfig
from evsys_sdk.registry import _agents, get_agent, list_agents, register_agent
from evsys_sdk.triggers import build_command

# 1. Registration -----------------------------------------------------------

def test_builtins_are_registered():
    assert get_agent("trigger") is TriggerAgent
    assert get_agent("autoresearch") is AutoresearchAgent
    assert {"trigger", "autoresearch"} <= set(list_agents())


def test_project_registered_agent():
    @register_agent("t_probe")
    class ProbeAgent(EvsysAgent):
        class Config(EvsysAgent.Config):
            target: str = "prompt"

        def build_prompt(self, **mission):
            return f"probe the {self.cfg.target}"

    try:
        agent = build_agent({"kind": "t_probe", "params": {"target": "planner"}})
        assert agent.build_command()[2] == "probe the planner"
    finally:
        _agents.unregister("t_probe")


def test_unknown_agent_fails_loudly():
    with pytest.raises(KeyError, match="No agent registered under 'nope'"):
        get_agent("nope")


def test_bad_agent_params_fail_loudly():
    with pytest.raises(ValidationError):
        build_agent(AgentSpec(kind="trigger", params={"modle": "distill"}))  # typo


# 2. TriggerAgent parity with the old build_command -------------------------

def _legacy_build_command(escalation_path, *, agent_cfg, root, verdict_path):
    """Frozen pre-refactor triggers/agent.py::build_command."""
    escalation_path = Path(escalation_path)
    root = Path(root)
    autoresearch = getattr(agent_cfg, "autoresearch", True)
    mode = getattr(agent_cfg, "mode", "verdict")
    default = DISTILL_PROMPT if mode == "distill" else DEFAULT_PROMPT
    template = getattr(agent_cfg, "prompt_template", None) or default
    distill = getattr(agent_cfg, "distill", None)
    prompt = template.format(
        escalation_path=escalation_path,
        traces_dir=(root.parent / "traces"),
        policy_path=(root / "policy.json"),
        verdict_path=verdict_path,
        autoresearch_clause=(AUTORESEARCH_ON if autoresearch else AUTORESEARCH_OFF),
        experiment_template=getattr(distill, "experiment_template", ""),
        holdout_fraction=getattr(distill, "holdout_fraction", 0.2),
        benchmark_dir=getattr(distill, "benchmark_dir", "data/benchmark"),
        train_dir=getattr(distill, "train_dir", "data/train"),
    )
    cmd = [getattr(agent_cfg, "claude_bin", "claude"), "-p", prompt,
           "--permission-mode", getattr(agent_cfg, "permission_mode", "acceptEdits")]
    if getattr(agent_cfg, "model", None):
        cmd += ["--model", agent_cfg.model]
    if getattr(agent_cfg, "plugin_dir", None):
        cmd += ["--plugin-dir", agent_cfg.plugin_dir]
    cmd += list(getattr(agent_cfg, "extra_args", None) or [])
    return cmd


TRIGGER_CFGS = [
    TriggerAgentConfig(enabled=True),
    TriggerAgentConfig(enabled=True, model="claude-opus-4-8", plugin_dir="/plug",
                       extra_args=["--verbose"], permission_mode="bypassPermissions"),
    TriggerAgentConfig(enabled=True, autoresearch=False),
    TriggerAgentConfig(enabled=True, mode="distill",
                       distill={"experiment_template": "tpl/opd.yaml",
                                "holdout_fraction": 0.25}),
    TriggerAgentConfig(enabled=True, prompt_template="CUSTOM {escalation_path}"),
]


@pytest.mark.parametrize("cfg", TRIGGER_CFGS, ids=[
    "defaults", "full-invocation", "autoresearch-off", "distill", "template-override"])
def test_trigger_agent_matches_legacy_argv(cfg):
    esc = "/s/escalations/escalation-00000010.json"
    expected = _legacy_build_command(esc, agent_cfg=cfg, root="/s",
                                     verdict_path="/s/verdicts/e.json")
    agent = TriggerAgent.from_config(cfg)
    assert agent.build_command(esc, root="/s", verdict_path="/s/verdicts/e.json") == expected
    # and the historical entry point (now a wrapper) is byte-identical too
    assert build_command(esc, agent_cfg=cfg, root="/s",
                         verdict_path="/s/verdicts/e.json") == expected


# 3. AutoresearchAgent parity ----------------------------------------------

def test_autoresearch_direct_prompt_matches_run_prompt_assembly():
    """The run_prompt argv: prompt verbatim + the FULL invocation surface."""
    cfg = TriggerAgentConfig(enabled=True, model="m", plugin_dir="/plug",
                             extra_args=["--verbose"])
    agent = AutoresearchAgent.from_config(cfg)
    assert agent.build_command("Improve the planner prompt") == [
        "claude", "-p", "Improve the planner prompt",
        "--permission-mode", "acceptEdits",
        "--model", "m", "--plugin-dir", "/plug", "--verbose",
    ]


def test_autoresearch_escalation_mission_matches_stage2_assembly():
    """run_remote stage 2: templated mission, NO plugin_dir / extra_args."""
    agent = AutoresearchAgent(claude_bin="claude", model="m",
                              permission_mode="acceptEdits")
    argv = agent.build_command(
        escalation_path=".evsys/triggers/escalations/e.json",
        verdict_path=".evsys/triggers/verdicts/e.json",
        traces_dir=".evsys/traces", artifacts="prompt.txt, planner.yaml",
    )
    expected_prompt = REMOTE_AUTORESEARCH_PROMPT.format(
        escalation_path=".evsys/triggers/escalations/e.json",
        verdict_path=".evsys/triggers/verdicts/e.json",
        traces_dir=".evsys/traces", artifacts="prompt.txt, planner.yaml",
    )
    assert argv == ["claude", "-p", expected_prompt,
                    "--permission-mode", "acceptEdits", "--model", "m"]


def test_autoresearch_template_override():
    agent = AutoresearchAgent(prompt_template="FIX {artifacts} for {escalation_path}")
    prompt = agent.build_prompt(escalation_path="e.json", verdict_path="v.json",
                                traces_dir="t", artifacts="prompt.txt")
    assert prompt == "FIX prompt.txt for e.json"


# 4. The environment field --------------------------------------------------

def test_environment_defaults_to_host():
    agent = TriggerAgent.from_config(TriggerAgentConfig(enabled=True))
    assert agent.environment is None
    assert agent.resolve_environment() is None


def test_environment_lifted_from_remote_config_when_enabled():
    cfg = TriggerAgentConfig(enabled=True, remote=RemoteAgentConfig(
        enabled=True, sandbox=SandboxSpec(kind="local")))
    agent = TriggerAgent.from_config(cfg)
    assert getattr(agent.environment, "kind", None) == "local"


def test_environment_ignored_while_remote_disabled():
    cfg = TriggerAgentConfig(enabled=True, remote=RemoteAgentConfig(
        enabled=False, sandbox=SandboxSpec(kind="local")))
    assert TriggerAgent.from_config(cfg).environment is None


def test_resolve_environment_returns_unstarted_sandbox():
    from evsys_sdk.sandboxes.local import LocalSandbox

    agent = AutoresearchAgent(environment=SandboxSpec(kind="local"))
    sbx = agent.resolve_environment(envs={"K": "v"})
    assert isinstance(sbx, LocalSandbox)
    assert sbx._started is False          # caller boots it (env fixups first)
    assert sbx.envs == {"K": "v"}


def test_environment_accepts_bare_kind_string():
    from evsys_sdk.sandboxes.local import LocalSandbox

    agent = AutoresearchAgent(environment="local")
    assert isinstance(agent.resolve_environment(), LocalSandbox)


# 5. The spec factory -------------------------------------------------------

def test_build_agent_from_spec():
    spec = AgentSpec(kind="trigger", params={"mode": "distill", "model": "m"})
    agent = build_agent(spec)
    assert isinstance(agent, TriggerAgent)
    assert agent.cfg.mode == "distill" and agent.cfg.model == "m"


def test_build_agent_overrides_win():
    agent = build_agent("autoresearch", model="override")
    assert agent.cfg.model == "override"

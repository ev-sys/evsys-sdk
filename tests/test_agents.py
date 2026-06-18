"""The agent harness as a registry extension ({kind, params} plugin).

These are tinker/harbor-free — they exercise the registry, AgentSpec, RunConfig
wiring, and resolve_agent's resolution logic, so they run in CI without extras.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from evsys_sdk import AgentSpec, get_agent, list_agents, register_agent
from evsys_sdk.config import AlgorithmConfig, DataConfig, ModelConfig, RunConfig
from evsys_sdk.training.harbor_engine import resolve_agent


def _run(**kw) -> RunConfig:
    base = dict(
        name="r", data=DataConfig(path="d.jsonl"),
        model=ModelConfig(name="Qwen/Qwen3-8B"),
        algorithm=AlgorithmConfig(kind="sft"),
    )
    base.update(kw)
    return RunConfig(**base)


# --- registry --------------------------------------------------------------


def test_basic_loop_is_registered():
    assert "basic_loop" in list_agents()
    plugin = get_agent("basic_loop")
    assert plugin.name == "basic_loop"
    assert plugin.agent_path == "evsys_sdk.training.harbor_agents:BasicLoopAgent"


def test_custom_agent_registers_and_resolves():
    @register_agent("custom_react")
    class _ReactPlugin:
        name = "custom_react"
        agent_path = "my.pkg:ReactAgent"

        class Config(BaseModel):
            model_config = ConfigDict(extra="forbid")
            max_tool_calls: int = 5
    try:
        path, params = resolve_agent(AgentSpec(kind="custom_react", params={"max_tool_calls": 9}))
        assert path == "my.pkg:ReactAgent"
        assert params == {"max_tool_calls": 9}
    finally:
        from evsys_sdk.registry import _agents
        _agents.unregister("custom_react")


# --- AgentSpec + RunConfig wiring ------------------------------------------


def test_agent_spec_defaults_to_basic_loop():
    assert AgentSpec().kind == "basic_loop"
    assert AgentSpec().params == {}


def test_runconfig_defaults_agent_to_basic_loop():
    assert _run().agent.kind == "basic_loop"


def test_runconfig_accepts_agent_spec():
    rc = _run(agent=AgentSpec(kind="basic_loop", params={"max_turns": 2}))
    assert rc.agent.params == {"max_turns": 2}


# --- resolve_agent ---------------------------------------------------------


def test_resolve_none_is_default_harness():
    assert resolve_agent(None) == ("evsys_sdk.training.harbor_agents:BasicLoopAgent", {})


def test_resolve_default_spec_yields_no_overrides():
    # exclude_unset → an unset spec does NOT clobber the rollout's own max_turns/system_prompt
    assert resolve_agent(AgentSpec()) == ("evsys_sdk.training.harbor_agents:BasicLoopAgent", {})


def test_resolve_explicit_params_override():
    _, params = resolve_agent(AgentSpec(kind="basic_loop", params={"max_turns": 3}))
    assert params == {"max_turns": 3}


def test_resolve_accepts_plain_dict():
    _, params = resolve_agent({"kind": "basic_loop", "params": {"system_prompt": "hi"}})
    assert params == {"system_prompt": "hi"}


def test_resolve_unknown_kind_raises():
    with pytest.raises(KeyError):
        resolve_agent({"kind": "does_not_exist"})


def test_resolve_rejects_unknown_param():
    with pytest.raises(Exception):  # pydantic ValidationError (extra=forbid)
        resolve_agent(AgentSpec(params={"bogus": 1}))

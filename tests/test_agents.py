"""Agent harness as a registry extension — a harbor ``BaseAgent`` subclass is
registered directly with ``@register_agent`` (same contract as harbor).

Registry mechanics / AgentSpec / RunConfig tests are tinker-free (CI). Tests that
resolve the built-in ``basic_loop`` import the real harbor ``BaseAgent`` (→ tinker),
so they go behind a tinker-guarded fixture.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from evsys_sdk import AgentSpec, get_agent, list_agents, register_agent
from evsys_sdk.config import AlgorithmConfig, DataConfig, ModelConfig, RunConfig


@pytest.fixture
def resolve_agent():
    pytest.importorskip("tinker")  # resolve imports harbor_agents (the real BaseAgent) → tinker
    from evsys_sdk.training.harbor_engine import resolve_agent as _resolve
    return _resolve


def _run(**kw) -> RunConfig:
    base = dict(name="r", data=DataConfig(path="d.jsonl"),
                model=ModelConfig(name="Qwen/Qwen3-8B"), algorithm=AlgorithmConfig(kind="sft"))
    base.update(kw)
    return RunConfig(**base)


# --- registry mechanics (CI-runnable, no harbor) ---------------------------


def test_register_agent_keeps_name_method():
    # A harbor BaseAgent exposes name() as a METHOD; registering must not clobber it.
    @register_agent("dummy_agent")
    class DummyAgent:
        @staticmethod
        def name() -> str:  # noqa: D401
            return "dummy_agent"

        @classmethod
        def import_path(cls) -> str:
            return "pkg.mod:DummyAgent"

        class Config(BaseModel):
            model_config = ConfigDict(extra="forbid")
            max_turns: int = 1
    try:
        assert "dummy_agent" in list_agents()
        cls = get_agent("dummy_agent")
        assert cls is DummyAgent
        assert callable(cls.name) and cls.name() == "dummy_agent"  # method preserved
    finally:
        from evsys_sdk.registry import _agents
        _agents.unregister("dummy_agent")


def test_get_unknown_agent_raises():
    with pytest.raises(KeyError):
        get_agent("nope_not_registered")


# --- AgentSpec + RunConfig wiring (CI-runnable) ----------------------------


def test_agent_spec_defaults_to_basic_loop():
    assert AgentSpec().kind == "basic_loop"
    assert AgentSpec().params == {}


def test_runconfig_defaults_agent_to_basic_loop():
    assert _run().agent.kind == "basic_loop"


def test_runconfig_accepts_agent_spec():
    rc = _run(agent=AgentSpec(kind="basic_loop", params={"max_turns": 2}))
    assert rc.agent.params == {"max_turns": 2}


# --- resolve_agent against the real BaseAgent (tinker-guarded) --------------


def test_resolve_none_is_basic_loop_class():
    pytest.importorskip("tinker")
    from evsys_sdk.training.harbor_engine import resolve_agent
    from evsys_sdk.training.harbor_agents import BasicLoopAgent
    path, params = resolve_agent(None)
    assert path == BasicLoopAgent.import_path()        # harbor's own import_path()
    assert "basic_loop" in list_agents()               # the real BaseAgent registered
    assert params == {}


def test_resolve_default_spec_yields_no_overrides(resolve_agent):
    _, params = resolve_agent(AgentSpec())             # exclude_unset → no clobber
    assert params == {}


def test_resolve_explicit_params_override(resolve_agent):
    _, params = resolve_agent(AgentSpec(kind="basic_loop", params={"max_turns": 3}))
    assert params == {"max_turns": 3}


def test_resolve_accepts_plain_dict(resolve_agent):
    _, params = resolve_agent({"kind": "basic_loop", "params": {"system_prompt": "hi"}})
    assert params == {"system_prompt": "hi"}


def test_resolve_rejects_unknown_param(resolve_agent):
    with pytest.raises(Exception):  # pydantic ValidationError (Config extra=forbid)
        resolve_agent(AgentSpec(params={"bogus": 1}))

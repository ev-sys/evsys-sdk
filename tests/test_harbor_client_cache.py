"""Harbor 0.13.2 agent wiring + the shared sampling-client cache.

Two regressions guarded here, both invisible to the mocked tests and the
EchoAgent smoke (empty kwargs / no TinkerLLM), found via a real tinker run:

1. ``_to_agent_config`` must lift ``model_name`` to the top-level
   ``AgentConfig.model_name`` field — harbor passes it to the agent explicitly,
   so leaving it in ``kwargs`` makes the agent constructor get it twice.
2. ``BasicLoopAgent._shared_llm`` caches the LLM per (event loop, config) so a
   single sampling client serves every trial of a job, built once even under
   concurrency, and isolated per event loop.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("harbor")
pytest.importorskip("tinker")  # harbor_agents imports TinkerLLM at module top

from evsys_sdk.training import harbor_agents as ha
from evsys_sdk.training import harbor_engine as he

# --- _to_agent_config: model_name lifted out of kwargs ----------------------


class _FakeAgentConfig:
    def __init__(self, *, import_path, model_name, kwargs):
        self.import_path = import_path
        self.model_name = model_name
        self.kwargs = kwargs


def test_to_agent_config_lifts_model_name_out_of_kwargs():
    ac = he._to_agent_config(
        _FakeAgentConfig,
        "mod:Agent",
        {"model_name": "Qwen/Qwen3-8B", "model_path": "p", "model_client": "tinker"},
    )
    assert ac.model_name == "Qwen/Qwen3-8B"
    assert "model_name" not in ac.kwargs  # harbor passes it explicitly; no double-pass
    assert ac.kwargs["model_path"] == "p"
    assert ac.kwargs["model_client"] == "tinker"


def test_to_agent_config_no_model_name_is_none():
    ac = he._to_agent_config(_FakeAgentConfig, "mod:Echo", {})
    assert ac.model_name is None and ac.kwargs == {}


# --- shared LLM cache -------------------------------------------------------


class _FakeLLM:
    def __init__(self):
        self.ensured = 0

    async def _ensure_client(self):
        self.ensured += 1


def _agent(tmp_path, **kw):
    defaults = dict(
        model_name="m", model_path="p", renderer_name="r", max_tokens=8,
        temperature=0.0, max_turns=1, system_prompt=None, model_client="tinker",
        logs_dir=tmp_path,
    )
    defaults.update(kw)
    return ha.BasicLoopAgent(**defaults)


@pytest.fixture(autouse=True)
def _clear_cache():
    ha._LLM_CACHE.clear()
    ha._LLM_LOCKS.clear()
    yield


def test_shared_llm_builds_once_under_concurrency(tmp_path, monkeypatch):
    builds = {"n": 0}

    def fake_build(self):
        builds["n"] += 1
        return _FakeLLM()

    monkeypatch.setattr(ha.BasicLoopAgent, "_build_llm", fake_build)

    async def main():
        # distinct agent instances (as harbor builds one per trial), same config
        agents = [_agent(tmp_path) for _ in range(8)]
        return await asyncio.gather(*(a._shared_llm() for a in agents))

    llms = asyncio.run(main())
    assert builds["n"] == 1                     # one build despite 8 concurrent agents
    assert len({id(x) for x in llms}) == 1      # all share the one instance
    assert llms[0].ensured == 1                 # sampling client warmed exactly once


def test_shared_llm_isolated_per_event_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(ha.BasicLoopAgent, "_build_llm", lambda self: _FakeLLM())

    async def one():
        return await _agent(tmp_path)._shared_llm()

    a = asyncio.run(one())
    b = asyncio.run(one())   # new event loop → fresh client (loop-bound httpx sessions)
    assert a is not b


def test_shared_llm_key_distinguishes_config(tmp_path, monkeypatch):
    monkeypatch.setattr(ha.BasicLoopAgent, "_build_llm", lambda self: _FakeLLM())

    async def main():
        x = await _agent(tmp_path, model_path="A")._shared_llm()
        y = await _agent(tmp_path, model_path="B")._shared_llm()
        z = await _agent(tmp_path, model_path="A")._shared_llm()
        return x, y, z

    x, y, z = asyncio.run(main())
    assert x is z and x is not y   # same config reused; different config separate

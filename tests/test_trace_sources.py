"""Layer-1 trace ingestion — model, LangGraph mapping, store/cursor, hook seam."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from evsys_sdk.trace_sources import LocalTraceStore
from evsys_sdk.trace_sources.langgraph import LangGraphTraceSource
from evsys_sdk.trace_types import Trace, trace_from_dict

FIX = Path(__file__).parent / "fixtures"


def _runs() -> list[dict]:
    return json.loads((FIX / "langsmith_run_tree.json").read_text())


def _source(root, hook=None) -> LangGraphTraceSource:
    return LangGraphTraceSource(store=LocalTraceStore(root), hook=hook, project_name="p")


def _lc(cls: str, **kwargs) -> dict:
    """A LangChain 'serialized constructor' message, as LangGraph stores in LangSmith."""
    return {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", cls], "kwargs": kwargs}


# 1. Model round-trip -------------------------------------------------------

def test_model_round_trip():
    t = Trace(
        trace_id="x",
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        feedback=[{"key": "thumbs", "score": 1.0, "comment": None, "source": "human", "turn": 1}],
        metadata={"source": "langgraph", "status": "success"},
    )
    assert trace_from_dict(json.loads(json.dumps(t.to_dict()))) == t
    assert t.input == "hi" and t.output == "yo"


# 2. LangGraph mapping (offline) -------------------------------------------

def test_langgraph_mapping(tmp_path):
    src = _source(tmp_path)
    raw = {
        "trace_id": "tr1",
        "runs": _runs(),
        "feedback": {
            "planner": [{"key": "step_quality", "score": 0.8, "comment": "good tool choice"}],
            "root": [{"key": "correctness", "score": 1.0, "comment": "correct"}],
        },
    }
    tr = src.to_trace(raw)

    assert tr.trace_id == "tr1"
    # full OpenAI conversation, tool call + tool result preserved
    assert [m["role"] for m in tr.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert tr.messages[2]["tool_calls"][0]["function"]["name"] == "search"
    assert tr.messages[3]["role"] == "tool" and tr.messages[3]["content"] == "Paris"
    # derived input/output
    assert tr.input == "What is the capital of France?"
    assert tr.output == "The capital of France is Paris."
    # per-turn vs whole-trace feedback
    fb = {f["key"]: f for f in tr.feedback}
    assert fb["step_quality"]["turn"] == 2  # the planner's assistant turn
    assert fb["step_quality"]["score"] == 0.8
    assert fb["correctness"]["turn"] is None  # root-level
    # metadata
    assert tr.metadata["source"] == "langgraph"
    assert tr.metadata["status"] == "success"
    assert tr.metadata["model"] == "gpt-4o"
    assert tr.metadata["tags"] == ["prod"]


# 2b. Real LangGraph/LangSmith serialization (regression — caught by a live agent)

def test_langchain_serialized_messages():
    """LangGraph runs store messages as LangChain 'constructor' objects and llm
    outputs as ChatResult 'generations' — not plain OpenAI shape. (Live-caught.)"""
    from evsys_sdk.trace_sources.langgraph import _normalize_msg, _run_output_messages

    assert _normalize_msg(_lc("HumanMessage", content="hi", type="human")) == {"role": "user", "content": "hi"}
    ai = _normalize_msg(_lc("AIMessage", content="", tool_calls=[{"name": "search", "args": {}, "id": "c1"}]))
    assert ai["role"] == "assistant" and ai["tool_calls"][0]["name"] == "search"
    assert _normalize_msg(_lc("ToolMessage", content="res", tool_call_id="c1")) == {
        "role": "tool", "content": "res", "tool_call_id": "c1",
    }
    # llm outputs come as {"generations": [[{"message": <constructor>}]]}
    outs = _run_output_messages({"generations": [[{"message": _lc("AIMessage", content="final")}]]})
    assert outs == [{"role": "assistant", "content": "final"}]


def test_langgraph_serialized_to_trace(tmp_path):
    src = _source(tmp_path)
    runs = [
        {"id": "root", "trace_id": "t2", "parent_run_id": None, "run_type": "chain", "name": "LangGraph",
         "status": "success", "start_time": "2026-07-06T00:00:00Z", "dotted_order": "a", "extra": {}},
        {"id": "planner", "trace_id": "t2", "parent_run_id": "root", "run_type": "llm", "dotted_order": "a.b",
         "inputs": {"messages": [[_lc("HumanMessage", content="Weather in Paris?", type="human")]]},
         "outputs": {"generations": [[{"message": _lc("AIMessage", content="",
             tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1"}])}]]}},
        {"id": "tool", "trace_id": "t2", "parent_run_id": "root", "run_type": "tool", "name": "get_weather",
         "dotted_order": "a.c", "inputs": {"city": "Paris"}, "outputs": {"result": "Sunny"}},
        {"id": "responder", "trace_id": "t2", "parent_run_id": "root", "run_type": "llm", "dotted_order": "a.d",
         "inputs": {"messages": [[
             _lc("HumanMessage", content="Weather in Paris?", type="human"),
             _lc("AIMessage", content="", tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1"}]),
             _lc("ToolMessage", content="Sunny in Paris", tool_call_id="c1"),
         ]]},
         "outputs": {"generations": [[{"message": _lc("AIMessage", content="It's sunny in Paris.")}]]}},
    ]
    raw = {"trace_id": "t2", "runs": runs,
           "feedback": {"planner": [{"key": "step", "score": 1.0}], "root": [{"key": "resolved", "score": 1.0}]}}
    tr = src.to_trace(raw)
    assert [m["role"] for m in tr.messages] == ["user", "assistant", "tool", "assistant"]
    assert tr.messages[1]["tool_calls"][0]["name"] == "get_weather"
    assert tr.messages[2] == {"role": "tool", "content": "Sunny in Paris", "tool_call_id": "c1"}
    assert tr.output == "It's sunny in Paris."
    fb = {f["key"]: f for f in tr.feedback}
    assert fb["step"]["turn"] == 1          # planner's tool-call turn
    assert fb["resolved"]["turn"] is None   # root → whole-trace


# 3. Local store + cursor + dedupe -----------------------------------------

def test_store_cursor_and_dedupe(tmp_path, monkeypatch):
    src = _source(tmp_path)
    raw = {"trace_id": "tr1", "runs": _runs(), "feedback": {}}
    monkeypatch.setattr(src, "pull_raw", lambda since: iter([raw]))

    assert src.run_once() == 1
    traces_file = tmp_path / "langgraph" / "traces.jsonl"
    assert len(traces_file.read_text().strip().splitlines()) == 1

    # re-pull the same trace → deduped, nothing new appended, cursor advanced
    assert src.run_once() == 0
    assert len(traces_file.read_text().strip().splitlines()) == 1
    cursor = json.loads((tmp_path / "langgraph" / "cursor.json").read_text())
    # cursor advanced to the trace's timestamp (Z or +00:00 form — same instant)
    assert cursor["since"].startswith("2026-06-27T14:23:45")
    assert "tr1" in cursor["seen_ids"]


# 4. Per-trace hook seam ----------------------------------------------------

def test_hook_fires_once_per_new_trace(tmp_path, monkeypatch):
    seen: list[str] = []
    src = _source(tmp_path, hook=lambda trace, ctx: seen.append(trace.trace_id))
    monkeypatch.setattr(src, "pull_raw", lambda since: iter([{"trace_id": "trX", "runs": _runs(), "feedback": {}}]))
    src.run_once()
    src.run_once()  # deduped → no second fire
    assert seen == ["trX"]


def test_raising_hook_is_swallowed(tmp_path, monkeypatch):
    def boom(trace, ctx):
        raise RuntimeError("hook exploded")

    src = _source(tmp_path, hook=boom)
    monkeypatch.setattr(src, "pull_raw", lambda since: iter([{"trace_id": "trY", "runs": _runs(), "feedback": {}}]))
    # daemon survives: the trace still lands and run_once returns normally
    assert src.run_once() == 1
    assert (tmp_path / "langgraph" / "traces.jsonl").exists()


# 5. Env-gated live pull ----------------------------------------------------

@pytest.mark.langsmith
@pytest.mark.skipif(not os.environ.get("LANGSMITH_API_KEY"), reason="needs LANGSMITH_API_KEY")
def test_live_pull(tmp_path):
    project = os.environ.get("LANGSMITH_TEST_PROJECT")
    if not project:
        pytest.skip("set LANGSMITH_TEST_PROJECT to run the live pull")
    src = LangGraphTraceSource(store=LocalTraceStore(tmp_path), project_name=project, limit=5)
    n = src.run_once(limit=5)
    assert n >= 0  # smoke: connects, maps, lands traces without crashing

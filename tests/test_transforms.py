"""Transform implementations."""

from __future__ import annotations

import pytest

from trajectory_experiments.transforms.composio import (
    ComposioRLNoToolsTransform,
    ComposioSFTNoToolsTransform,
)
from trajectory_experiments.transforms.identity import IdentityTransform
from trajectory_experiments.transforms.jsonl_to_chat import JSONLToChatTransform


def test_identity():
    rows = [{"x": 1}, {"y": 2}]
    out = list(IdentityTransform()(rows))
    assert out == rows


def test_jsonl_to_chat_user_only():
    t = JSONLToChatTransform(
        system_prompt="sys",
        user_template="Q: {query}",
    )
    out = list(t([{"query": "hi"}]))
    assert out[0]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Q: hi"},
    ]


def test_jsonl_to_chat_with_assistant():
    t = JSONLToChatTransform(
        user_template="Q: {query}",
        assistant_template="A: {answer}",
    )
    out = list(t([{"query": "hi", "answer": "ok"}]))
    assert any(m["role"] == "assistant" for m in out[0]["messages"])
    assert out[0]["messages"][-1]["content"] == "A: ok"


def test_composio_sft_no_tools_drops_tool_list(composio_rows):
    t = ComposioSFTNoToolsTransform(include_assistant=True)
    out = list(t(composio_rows))
    assert len(out) == len(composio_rows)
    for row in out:
        msgs = row["messages"]
        # 3 messages: system, user, assistant
        assert len(msgs) == 3
        # Critical: user message must NOT contain "Available tools:"
        user = next(m for m in msgs if m["role"] == "user")
        assert "Available tools" not in user["content"]
        assert user["content"].startswith("Query: ")
        # Assistant should have the answer
        asst = next(m for m in msgs if m["role"] == "assistant")
        assert "<answer>" in asst["content"]
        assert row["tool_slug"] in asst["content"]


def test_composio_sft_no_tools_no_assistant(composio_rows):
    t = ComposioSFTNoToolsTransform(include_assistant=False)
    out = list(t(composio_rows))
    for row in out:
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["system", "user"]


def test_composio_rl_no_tools_emits_prompt(composio_rows):
    t = ComposioRLNoToolsTransform()
    out = list(t(composio_rows))
    for row in out:
        assert "prompt" in row
        assert "Available tools" not in row["prompt"]
        assert "Query: " in row["prompt"]
        assert "tool_slug" in row

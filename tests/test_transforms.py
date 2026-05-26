"""Transform implementations."""

from __future__ import annotations

import pytest

from trajectory_labs.transforms.identity import IdentityTransform
from trajectory_labs.transforms.jsonl_to_chat import JSONLToChatTransform


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







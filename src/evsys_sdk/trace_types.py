"""Ingested agent-trace data model.

A :class:`Trace` is one recorded run of an external (hosted) agent — pulled from
an observability platform (LangSmith / LangGraph today) into a local,
provider-agnostic form. It is intentionally MINIMAL and **message-centric**: the
OpenAI-format ``messages`` list IS the trace (the full conversation, including
tool calls and tool results), plus per-turn ``feedback`` and a small ``metadata``
bag. The rich observability wrapper (nested spans, per-node latency/cost, parent
ids, dotted-order) is deliberately dropped — the full platform trace stays one
``trace_id`` lookup away.

This is the ingestion shape only. Consumers (the trigger fn / autoresearch, added
in later layers) read ``Trace`` directly; there are deliberately no conversion
helpers here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Trace:
    """One ingested agent run.

    ``messages`` — the conversation in OpenAI chat format: each item is
    ``{"role": system|user|assistant|tool, "content": ...}``; an assistant turn
    may carry ``"tool_calls": [{"id", "type": "function", "function": {"name",
    "arguments"}}]`` and a tool result is ``{"role": "tool", "tool_call_id",
    "content"}``. This is the whole trace — nothing the downstream agents consume
    is lost.

    ``feedback`` — zero or more signals, each attached to a specific turn OR to
    the whole trace: ``{"key": str, "score": float | None, "comment": str | None,
    "source": str, "turn": int | None}`` where ``turn`` indexes into ``messages``
    (``None`` = whole-trace feedback).

    ``metadata`` — freeform: ``source`` (the adapter name), ``model``, ``status``,
    ``tags``, ``timestamp``, ...
    """

    trace_id: str
    messages: list[dict]
    feedback: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # -- derived convenience (not stored) -------------------------------------

    @property
    def input(self) -> Any:
        """The first ``user`` message's content — the task the agent received."""
        for m in self.messages:
            if m.get("role") == "user":
                return m.get("content")
        return None

    @property
    def output(self) -> Any:
        """The last ``assistant`` message's content — the agent's final result."""
        for m in reversed(self.messages):
            if m.get("role") == "assistant":
                return m.get("content")
        return None

    def to_dict(self) -> dict:
        return asdict(self)


def trace_from_dict(row: dict) -> Trace:
    """Build a :class:`Trace` from a plain dict (e.g. one JSONL row)."""
    return Trace(
        trace_id=str(row["trace_id"]),
        messages=list(row.get("messages") or []),
        feedback=list(row.get("feedback") or []),
        metadata=dict(row.get("metadata") or {}),
    )


def iter_traces_jsonl(path: str):
    """Yield :class:`Trace`\\s from a JSONL file (one trace per line)."""
    import json

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield trace_from_dict(json.loads(line))


__all__ = ["Trace", "iter_traces_jsonl", "trace_from_dict"]

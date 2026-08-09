"""LangGraph / LangSmith trace adapter.

Pulls runs from a LangSmith project (`client.list_runs`) + their feedback
(`client.list_feedback`), and maps each trace's run-tree into the minimal,
message-centric :class:`~evsys_sdk.trace_types.Trace`:

  * ``messages`` — the OpenAI-format conversation, taken from the trace's final
    (responder) LLM run's ``inputs.messages`` (the full prior history, incl.
    ``tool_calls`` + tool-role results) plus its ``outputs`` (the final turn).
  * ``feedback`` — per-turn or per-trace: each LangSmith feedback (attached to a
    run id) becomes ``{key, score, comment, source, turn}``; a child-run's
    feedback maps to the turn that run produced, root-run feedback → ``turn=None``.

Accessors tolerate both real LangSmith ``Run``/``Feedback`` objects and plain
dicts (so the mapping is unit-testable from a fixture without the SDK).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..logger import get_logger
from ..registry import register_trace_source
from ..trace_types import Trace
from .base import BaseTraceSource

log = get_logger(__name__)

_LC_ROLE = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool", "function": "tool"}
# LangChain message-class name → OpenAI role (for the serialized "constructor" form).
_CLASS_ROLE = {
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "AIMessageChunk": "assistant",
    "SystemMessage": "system",
    "ToolMessage": "tool",
    "FunctionMessage": "tool",
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute-or-dict accessor (works for LangSmith objects and fixtures)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_msg(m: Any) -> dict:
    """Normalize one message to OpenAI shape (role/content [+ tool_calls / tool_call_id])."""
    if not isinstance(m, dict):
        return {"role": "assistant", "content": str(m)}
    if "role" in m:  # already OpenAI-ish
        out: dict = {"role": m["role"], "content": m.get("content", "")}
        if m.get("tool_calls"):
            out["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id"):
            out["tool_call_id"] = m["tool_call_id"]
        return out
    # LangChain serialized "constructor" form (what LangGraph runs store in LangSmith):
    # {"lc": 1, "type": "constructor", "id": [..., "HumanMessage"], "kwargs": {"content", ...}}
    if m.get("type") == "constructor" and isinstance(m.get("id"), list) and isinstance(m.get("kwargs"), dict):
        cls = m["id"][-1] if m["id"] else ""
        kw = m["kwargs"]
        out = {"role": _CLASS_ROLE.get(cls, "assistant"), "content": kw.get("content", "")}
        tc = kw.get("tool_calls") or (kw.get("additional_kwargs") or {}).get("tool_calls")
        if tc:
            out["tool_calls"] = tc
        if kw.get("tool_call_id"):
            out["tool_call_id"] = kw["tool_call_id"]
        return out
    # LangChain simple form: {"type": "human"|"ai"|..., "data": {"content", ...}}
    t = m.get("type") or (m.get("data") or {}).get("type")
    data = m.get("data") or m
    out = {"role": _LC_ROLE.get(t or "", "assistant"), "content": data.get("content", "")}
    tc = data.get("tool_calls") or (data.get("additional_kwargs") or {}).get("tool_calls")
    if tc:
        out["tool_calls"] = tc
    if data.get("tool_call_id"):
        out["tool_call_id"] = data["tool_call_id"]
    return out


def _messages_from_io(io: Any) -> list[dict]:
    """Extract an OpenAI message list from a run's ``inputs`` or ``outputs``."""
    if not isinstance(io, dict):
        return []
    msgs = io.get("messages")
    if not msgs:
        return []
    if isinstance(msgs[0], list):  # batched (list-of-lists)
        msgs = msgs[0]
    return [_normalize_msg(m) for m in msgs if m]


def _run_output_messages(outputs: Any) -> list[dict]:
    """Assistant message(s) from an llm run's ``outputs`` — handles both a
    ``{"messages": [...]}`` shape and LangChain's ChatResult
    ``{"generations": [[{"message": {...}}]]}``."""
    if not isinstance(outputs, dict):
        return []
    if outputs.get("messages"):
        return _messages_from_io(outputs)
    gens = outputs.get("generations")
    if not gens:
        return []
    flat = gens[0] if isinstance(gens[0], list) else gens
    out: list[dict] = []
    for g in flat:
        if not isinstance(g, dict):
            continue
        if g.get("message") is not None:
            out.append(_normalize_msg(g["message"]))
        elif g.get("text"):
            out.append({"role": "assistant", "content": g["text"]})
    return out


def _build_messages(llm_runs: list) -> list[dict]:
    """Full conversation = the last LLM run's input history + its final output turn."""
    if not llm_runs:
        return []
    last = llm_runs[-1]
    msgs = _messages_from_io(_get(last, "inputs"))
    out = _run_output_messages(_get(last, "outputs"))
    if out:
        msgs = [*msgs, out[-1]]
    return msgs


def _build_feedback(runs: list, root: Any, feedback_by_run: dict, messages: list[dict]) -> list[dict]:
    # Map an assistant message's content → its turn index (for per-turn feedback).
    content_to_turn: dict[str, int] = {}
    for i, m in enumerate(messages):
        c = m.get("content")
        if m.get("role") == "assistant" and isinstance(c, str):
            content_to_turn.setdefault(c, i)
    root_id = str(_get(root, "id")) if root is not None else None
    out: list[dict] = []
    for run in runs:
        rid = str(_get(run, "id"))
        for fb in feedback_by_run.get(rid, []):
            turn = None
            if rid != root_id:
                run_out = _run_output_messages(_get(run, "outputs"))
                if run_out:
                    c = run_out[-1].get("content")
                    turn = content_to_turn.get(c) if isinstance(c, str) else None
            out.append(
                {
                    "key": _get(fb, "key"),
                    "score": _get(fb, "score"),
                    "comment": _get(fb, "comment"),
                    "source": "langsmith",
                    "turn": turn,
                }
            )
    return out


class LangGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_name: str
    filter: str | None = None
    select: list[str] | None = None
    limit: int = 100
    api_key_env: str = "LANGSMITH_API_KEY"


@register_trace_source("langgraph")
class LangGraphTraceSource(BaseTraceSource):
    name = "langgraph"
    Config = LangGraphConfig

    def pull_raw(self, since: datetime | None) -> Iterable[Any]:
        try:
            from langsmith import Client
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "LangGraph/LangSmith ingestion needs the 'traces' extra: "
                "pip install evsys-sdk[traces]"
            ) from e

        cfg = self.cfg
        assert cfg is not None
        api_key = os.environ.get(cfg.api_key_env)
        client = Client(api_key=api_key) if api_key else Client()

        kwargs: dict[str, Any] = {"project_name": cfg.project_name, "limit": cfg.limit}
        if since is not None:
            kwargs["start_time"] = since
        if cfg.filter:
            kwargs["filter"] = cfg.filter
        if cfg.select:
            kwargs["select"] = cfg.select

        runs = list(client.list_runs(**kwargs))
        by_trace: dict[str, list] = {}
        for run in runs:
            tid = str(_get(run, "trace_id") or _get(run, "id"))
            by_trace.setdefault(tid, []).append(run)

        run_ids = [_get(r, "id") for rl in by_trace.values() for r in rl]
        feedback_by_run: dict[str, list] = {}
        if run_ids:
            try:
                for fb in client.list_feedback(run_ids=run_ids):
                    feedback_by_run.setdefault(str(_get(fb, "run_id")), []).append(fb)
            except Exception as e:  # feedback is best-effort
                log.debug("list_feedback failed: %s", e)

        for tid, rl in by_trace.items():
            yield {"trace_id": tid, "runs": rl, "feedback": feedback_by_run}

    def to_trace(self, raw: Any) -> Trace:
        runs = list(raw.get("runs") or [])
        tid = str(raw["trace_id"])
        feedback_by_run = raw.get("feedback") or {}

        runs_sorted = sorted(
            runs,
            key=lambda r: str(_get(r, "dotted_order") or _get(r, "start_time") or ""),
        )
        root = next(
            (r for r in runs_sorted if _get(r, "parent_run_id") is None),
            runs_sorted[0] if runs_sorted else None,
        )
        llm_runs = [r for r in runs_sorted if _get(r, "run_type") == "llm"]
        messages = _build_messages(llm_runs)
        feedback = _build_feedback(runs_sorted, root, feedback_by_run, messages)

        extra = _get(root, "extra") or {}
        model = (extra.get("metadata") or {}).get("ls_model_name") or (extra.get("metadata") or {}).get("model")
        start = _get(root, "start_time")
        metadata = {
            "source": "langgraph",
            "model": model,
            "status": _get(root, "status"),
            "tags": _get(root, "tags") or [],
            "timestamp": str(start) if start else None,
        }
        return Trace(trace_id=tid, messages=messages, feedback=feedback, metadata=metadata)


__all__ = ["LangGraphConfig", "LangGraphTraceSource"]

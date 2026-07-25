"""Claude Code session-transcript adapter — ingest coding traces as you code.

Reads Claude Code's on-disk session transcripts
(``~/.claude/projects/<cwd-slug>/<session-uuid>.jsonl``) for one project and
folds each *settled* session into the minimal, message-centric
:class:`~evsys_sdk.trace_types.Trace`:

  * ``messages`` — the OpenAI-format conversation: user prompts, assistant
    turns (text + ``tool_calls``; consecutive transcript lines sharing a
    ``requestId`` are one turn; thinking blocks dropped), and tool results as
    ``role: tool`` messages.
  * ``feedback`` — cheap deterministic signals only (user-correction lexicon,
    interruptions, tool-error rate). Judging whether a session is a *failure
    worth learning from* stays with the gate / trigger agent.

Sessions are ingested only after ``min_idle_s`` of transcript inactivity:
the store dedupes by ``trace_id`` (= sessionId), so a session folded
mid-conversation would never pick up its later turns.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..logger import get_logger
from ..registry import register_trace_source
from ..trace_types import Trace
from .base import BaseTraceSource

log = get_logger(__name__)

# Everything that is not a user/assistant conversation line is session
# bookkeeping (last-prompt, mode, ai-title, file-history-snapshot, attachment,
# queue-operation, system, pr-link, ...) and is skipped by the allowlist below.

_CORRECTION_MARKERS = (
    "no,", "no ", "that's wrong", "thats wrong", "not what i", "revert", "undo that",
    "you broke", "that broke", "doesn't work", "doesnt work", "still failing", "still broken",
)
_INTERRUPT_MARKER = "[request interrupted by user"


def project_slug(project_path: str | Path) -> str:
    """Claude Code's cwd→directory-name rule: ``/`` and ``.`` become ``-``."""
    return str(project_path).replace("/", "-").replace(".", "-")


def _text(content: Any) -> str:
    """Stringify a tool_result / message content that may be str or block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return "" if content is None else str(content)


def fold_session(lines: list[dict], *, include_sidechains: bool = False) -> tuple[list[dict], dict]:
    """Fold one session's transcript lines into (OpenAI messages, metadata).

    Pure — unit-testable from a fixture. Transcript file order is authoritative
    (the file is append-ordered); consecutive ``assistant`` lines sharing a
    ``requestId`` merge into a single assistant turn.
    """
    conv = [
        ln for ln in lines
        if ln.get("type") in ("user", "assistant")
        and isinstance(ln.get("message"), dict)
        and (include_sidechains or not ln.get("isSidechain"))
    ]

    messages: list[dict] = []
    n_tools = 0
    n_tool_errors = 0
    cur_req: str | None = None  # requestId of the assistant turn being merged

    for ln in conv:
        msg = ln["message"]
        if ln["type"] == "user":
            cur_req = None
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        n_tools += 1
                        if block.get("is_error"):
                            n_tool_errors += 1
                        messages.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": _text(ln.get("toolUseResult") or block.get("content")),
                        })
                    elif block.get("type") == "text" and block.get("text"):
                        messages.append({"role": "user", "content": block["text"]})
            elif content:
                messages.append({"role": "user", "content": content})
            continue

        # assistant line: one content block per line, requestId groups a turn
        req = ln.get("requestId")
        if req is None or req != cur_req or not messages or messages[-1]["role"] != "assistant":
            messages.append({"role": "assistant", "content": ""})
            cur_req = req
        turn = messages[-1]
        for block in msg.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                turn["content"] = (turn["content"] + "\n" + block["text"]).strip()
            elif block.get("type") == "tool_use":
                turn.setdefault("tool_calls", []).append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                })
            # thinking / redacted_thinking: dropped

    # drop empty assistant shells (thinking-only lines with no text/tool_use)
    messages = [m for m in messages if m["role"] != "assistant" or m["content"] or m.get("tool_calls")]

    first = conv[0] if conv else {}
    model = next(
        (ln["message"].get("model") for ln in reversed(conv)
         if ln["type"] == "assistant" and ln["message"].get("model")),
        None,
    )
    stamps = [str(ln["timestamp"]) for ln in lines if ln.get("timestamp")]
    meta = {
        "source": "claude_code",
        "model": model,
        "cwd": first.get("cwd"),
        "git_branch": first.get("gitBranch"),
        "claude_code_version": first.get("version"),
        "timestamp": max(stamps) if stamps else None,
        "n_tool_results": n_tools,
        "n_tool_errors": n_tool_errors,
    }
    return messages, meta


def derive_feedback(messages: list[dict], meta: dict) -> list[dict]:
    """Cheap deterministic signals; scores follow the grader convention (1 good, 0 bad)."""
    out: list[dict] = []
    for i, m in enumerate(messages):
        if m["role"] != "user" or not isinstance(m.get("content"), str):
            continue
        low = m["content"].lower()
        if _INTERRUPT_MARKER in low:
            out.append({"key": "interrupted", "score": 0.0,
                        "comment": "user interrupted the assistant", "source": "claude_code", "turn": i})
        elif i > 0 and any(low.startswith(t) or t in low[:120] for t in _CORRECTION_MARKERS):
            out.append({"key": "user_correction", "score": 0.0,
                        "comment": m["content"][:160], "source": "claude_code", "turn": i})
    n_tools, n_err = meta.get("n_tool_results", 0), meta.get("n_tool_errors", 0)
    if n_tools:
        out.append({"key": "tool_error_rate", "score": round(1.0 - n_err / n_tools, 4),
                    "comment": f"{n_err}/{n_tools} tool calls errored",
                    "source": "claude_code", "turn": None})
    return out


class ClaudeCodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    projects_dir: str = "~/.claude/projects"
    project: str | None = None
    """Absolute cwd of the repo whose sessions to ingest; None → the daemon's cwd."""
    include_sidechains: bool = False
    """Include subagent (sidechain) transcript lines in the folded conversation."""
    min_idle_s: float = 300.0
    """Only fold sessions idle for at least this long (the store dedupes by
    session id, so a session ingested mid-flight would never be re-read)."""
    derive_feedback: bool = True
    limit: int = 100
    """Max session files considered per pull."""


@register_trace_source("claude_code")
class ClaudeCodeTraceSource(BaseTraceSource):
    name = "claude_code"
    Config = ClaudeCodeConfig

    def pull_raw(self, since: datetime | None) -> Iterable[Any]:
        cfg = self.cfg
        assert cfg is not None
        project = cfg.project or str(Path.cwd())
        root = Path(cfg.projects_dir).expanduser() / project_slug(project)
        if not root.is_dir():
            log.debug("[claude_code] no transcript dir for project %s (%s)", project, root)
            return
        now = time.time()
        files = sorted(root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        for path in files[-cfg.limit:]:
            mtime = path.stat().st_mtime
            if now - mtime < cfg.min_idle_s:
                continue  # still live — fold once it settles
            if since is not None and datetime.fromtimestamp(mtime, UTC) <= since:
                continue
            lines: list[dict] = []
            for raw_line in path.read_text().splitlines():
                if not raw_line.strip():
                    continue
                try:
                    lines.append(json.loads(raw_line))
                except json.JSONDecodeError:
                    continue  # torn tail line
            if lines:
                yield {"session_path": str(path), "lines": lines}

    def to_trace(self, raw: Any) -> Trace:
        cfg = self.cfg
        assert cfg is not None
        lines = raw["lines"]
        messages, meta = fold_session(lines, include_sidechains=cfg.include_sidechains)
        meta["session_path"] = raw["session_path"]
        trace_id = next(
            (ln.get("sessionId") for ln in lines if ln.get("sessionId")),
            Path(raw["session_path"]).stem,
        )
        feedback = derive_feedback(messages, meta) if cfg.derive_feedback else []
        return Trace(trace_id=str(trace_id), messages=messages, feedback=feedback, metadata=meta)


__all__ = ["ClaudeCodeConfig", "ClaudeCodeTraceSource", "fold_session", "derive_feedback", "project_slug"]

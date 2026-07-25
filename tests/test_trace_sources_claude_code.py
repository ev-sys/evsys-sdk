"""claude_code trace source — session folding, feedback heuristics, pull
mechanics (idle threshold, cursor, dedupe) over a synthetic transcript dir."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from evsys_sdk.registry import get_trace_source
from evsys_sdk.trace_sources.claude_code import (
    ClaudeCodeTraceSource,
    derive_feedback,
    fold_session,
    project_slug,
)
from evsys_sdk.trace_sources.store import LocalTraceStore

SESSION_ID = "11111111-2222-3333-4444-555555555555"


def _line(type_: str, content, *, req: str | None = None, sidechain: bool = False,
          ts: str = "2026-07-25T10:00:00.000Z", **top) -> dict:
    ln = {
        "type": type_, "sessionId": SESSION_ID, "timestamp": ts, "isSidechain": sidechain,
        "cwd": "/repo", "gitBranch": "main", "version": "2.1.202",
        "message": {"role": type_, "content": content}, **top,
    }
    if req is not None:
        ln["requestId"] = req
    return ln


def _session_lines() -> list[dict]:
    return [
        {"type": "ai-title", "aiTitle": "Fix the bug"},                       # bookkeeping: skipped
        _line("user", "fix the failing test in utils"),
        # one assistant response split across 3 lines sharing a requestId
        _line("assistant", [{"type": "thinking", "thinking": "hmm"}], req="req_1"),
        _line("assistant", [{"type": "text", "text": "Looking at the test."}], req="req_1"),
        _line("assistant", [{"type": "tool_use", "id": "tu_1", "name": "Bash",
                             "input": {"command": "pytest -q"}}], req="req_1"),
        _line("user", [{"type": "tool_result", "tool_use_id": "tu_1",
                        "content": "1 failed", "is_error": True}]),
        _line("assistant", [{"type": "text", "text": "Fixed. Tests pass now."}], req="req_2",
              ts="2026-07-25T10:05:00.000Z"),
        _line("user", "no, that's wrong - you deleted the assertion",
              ts="2026-07-25T10:06:00.000Z"),
        _line("assistant", [{"type": "text", "text": "sidechain noise"}], req="req_3", sidechain=True),
        {"type": "queue-operation", "operation": "add"},                      # bookkeeping: skipped
    ]


class TestFolding:
    def test_fold_session_shapes(self):
        messages, meta = fold_session(_session_lines())
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "tool", "assistant", "user"]
        # requestId merge: text + tool_use in ONE assistant turn, thinking dropped
        turn = messages[1]
        assert turn["content"] == "Looking at the test."
        assert turn["tool_calls"][0]["function"]["name"] == "Bash"
        assert json.loads(turn["tool_calls"][0]["function"]["arguments"]) == {"command": "pytest -q"}
        assert messages[2] == {"role": "tool", "tool_call_id": "tu_1", "content": "1 failed"}
        # sidechain line excluded by default
        assert all("sidechain" not in str(m.get("content")) for m in messages)
        assert meta["cwd"] == "/repo" and meta["git_branch"] == "main"
        assert meta["timestamp"] == "2026-07-25T10:06:00.000Z"  # max stamp drives the cursor
        assert meta["n_tool_results"] == 1 and meta["n_tool_errors"] == 1

    def test_fold_includes_sidechains_when_asked(self):
        messages, _ = fold_session(_session_lines(), include_sidechains=True)
        assert any("sidechain noise" in str(m.get("content")) for m in messages)

    def test_slug(self):
        assert project_slug("/Users/x/proj.name") == "-Users-x-proj-name"


class TestFeedback:
    def test_correction_interrupt_and_tool_errors(self):
        messages, meta = fold_session(_session_lines())
        fb = derive_feedback(messages, meta)
        keys = {f["key"]: f for f in fb}
        assert keys["user_correction"]["score"] == 0.0
        assert keys["user_correction"]["turn"] == len(messages) - 1
        assert keys["tool_error_rate"]["score"] == 0.0  # 1/1 errored
        assert keys["tool_error_rate"]["turn"] is None

    def test_interruption(self):
        messages = [{"role": "assistant", "content": "working..."},
                    {"role": "user", "content": "[Request interrupted by user]"}]
        fb = derive_feedback(messages, {})
        assert fb and fb[0]["key"] == "interrupted"

    def test_clean_session_only_rate(self):
        messages = [{"role": "user", "content": "add a test"},
                    {"role": "assistant", "content": "done"}]
        assert derive_feedback(messages, {"n_tool_results": 4, "n_tool_errors": 0}) == [
            {"key": "tool_error_rate", "score": 1.0, "comment": "0/4 tool calls errored",
             "source": "claude_code", "turn": None}]


def _write_session(root: Path, project: str, name: str, lines: list[dict], *, age_s: float) -> Path:
    d = root / project_slug(project)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.jsonl"
    p.write_text("".join(json.dumps(ln) + "\n" for ln in lines) + "{torn")
    stamp = time.time() - age_s
    os.utime(p, (stamp, stamp))
    return p


class TestPull:
    def _source(self, tmp_path: Path, **params) -> ClaudeCodeTraceSource:
        store = LocalTraceStore(tmp_path / "state")
        cls = get_trace_source("claude_code")
        assert cls is ClaudeCodeTraceSource
        return cls(store=store,
                   projects_dir=str(tmp_path / "projects"), project="/repo", **params)

    def test_run_once_ingests_settled_sessions_only(self, tmp_path: Path):
        root = tmp_path / "projects"
        _write_session(root, "/repo", SESSION_ID, _session_lines(), age_s=3600)
        live = [dict(ln, sessionId="live-session") for ln in _session_lines()]
        _write_session(root, "/repo", "live-session", live, age_s=5)  # still typing

        src = self._source(tmp_path, min_idle_s=300)
        assert src.run_once() == 1  # only the settled session
        traces_file = tmp_path / "state" / "claude_code" / "traces.jsonl"
        rows = [json.loads(l) for l in traces_file.read_text().splitlines()]
        assert [r["trace_id"] for r in rows] == [SESSION_ID]
        assert rows[0]["metadata"]["session_path"].endswith(f"{SESSION_ID}.jsonl")

        # second pull: dedupe by session id, nothing new
        assert src.run_once() == 0

    def test_other_project_ignored(self, tmp_path: Path):
        _write_session(tmp_path / "projects", "/other", SESSION_ID, _session_lines(), age_s=3600)
        assert self._source(tmp_path).run_once() == 0

    def test_missing_dir_is_quiet(self, tmp_path: Path):
        assert self._source(tmp_path).run_once() == 0

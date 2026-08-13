"""Platform sandbox + trace source + queue adapter tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from evsys_sdk.compute.platform_queue import queue_enabled, submit_training_config
from evsys_sdk.sandboxes.platform import PlatformSandbox
from evsys_sdk.trace_sources.platform import PlatformLangSmithTraceSource
from evsys_sdk.trace_sources.store import LocalTraceStore
from evsys_sdk.triggers import remote as remotemod


class TestPlatformSandbox:
    def test_start_exec_kill(self, monkeypatch):
        client = MagicMock()
        client.create_sandbox.return_value = {"id": "sb-1", "status": "running"}
        client.exec_sandbox.return_value = {"stdout": "hi\n", "stderr": "", "exit_code": 0}
        monkeypatch.setattr("evsys_sdk.sandboxes.platform.PlatformClient", lambda **kw: client)

        sbx = PlatformSandbox(envs={}, timeout_s=60)
        sbx.start()
        code, out = sbx.exec("echo hi", timeout_s=10)
        assert code == 0
        assert "hi" in out
        sbx.kill()
        client.stop_sandbox.assert_called_once_with("sb-1", kill=True)

    def test_write_read_via_exec(self, monkeypatch):
        fs: dict[str, str] = {}

        def _exec(sandbox_id, command, **kw):
            if "base64 -d >" in command:
                parts = command.split("echo ", 1)[1].split(" | base64 -d > ")
                b64 = parts[0].strip("'\"")
                path = parts[1].strip()
                import base64

                fs[path] = base64.b64decode(b64).decode()
            elif "base64 -w0" in command:
                path = command.split("base64 -w0 ")[1].split(" ")[0].strip("'\"")
                import base64

                data = fs.get(path, "")
                return {"stdout": base64.b64encode(data.encode()).decode(), "stderr": "", "exit_code": 0}
            return {"stdout": "", "stderr": "", "exit_code": 0}

        client = MagicMock()
        client.create_sandbox.return_value = {"id": "sb-2"}
        client.exec_sandbox.side_effect = lambda sid, cmd, **kw: _exec(sid, cmd, **kw)
        monkeypatch.setattr("evsys_sdk.sandboxes.platform.PlatformClient", lambda **kw: client)

        sbx = PlatformSandbox(envs={}, timeout_s=60)
        sbx.start()
        sbx.write("/box/a.txt", "hello")
        assert sbx.read("/box/a.txt") == "hello"


class TestPlatformTraceSource:
    def test_pull_maps_traces(self, monkeypatch):
        client = MagicMock()
        client.langsmith_status.return_value = {"connected": True, "project_name": "demo"}
        client.langsmith_pull.return_value = {
            "traces": [{
                "trace_id": "t1",
                "messages": [{"role": "user", "content": "hi"}],
                "feedback": [],
                "metadata": {"source": "langgraph"},
            }],
            "count": 1,
        }
        monkeypatch.setattr("evsys_sdk.trace_sources.platform.PlatformClient", lambda **kw: client)

        store = LocalTraceStore(root="/tmp/traces-test")
        src = PlatformLangSmithTraceSource(store=store, project_name="demo")
        raw = list(src.pull_raw(since=datetime(2026, 1, 1, tzinfo=timezone.utc)))
        assert len(raw) == 1
        trace = src.to_trace(raw[0])
        assert trace.trace_id == "t1"
        assert trace.messages[0]["content"] == "hi"


class TestRemotePlatformDefault:
    def test_prefers_platform_without_local_e2b_key(self, monkeypatch):
        monkeypatch.setenv("EVSYS_API_KEY", "sk_test")
        monkeypatch.delenv("E2B_API_KEY", raising=False)
        built = []

        def _fake_build(spec, **kw):
            built.append(spec)
            return MagicMock()

        monkeypatch.setattr(remotemod, "build_sandbox", _fake_build)
        cfg = MagicMock()
        cfg.sandbox = MagicMock(kind="e2b", params={})
        cfg.timeout_s = 60
        remotemod._make_sandbox(cfg, {})
        assert built[0].kind == "platform"


class TestPlatformQueue:
    def test_queue_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("EVSYS_COMPUTE_QUEUE", raising=False)
        assert queue_enabled() is False

    def test_submit_enqueues(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EVSYS_COMPUTE_QUEUE", "1")
        cfg = tmp_path / "train.yaml"
        cfg.write_text("recipe: sft\n")
        q = __import__("evsys_sdk.compute.queue", fromlist=["Queue"]).Queue(path=tmp_path / "q.jsonl")
        job = submit_training_config(cfg, model="Qwen/Qwen3-4B", queue=q, auto_schedule=False)
        assert job.state == "queued"
        assert job.model == "Qwen/Qwen3-4B"

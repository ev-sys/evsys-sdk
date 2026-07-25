"""Remote (E2B) agent execution — staging manifest, two-stage run, artifact
copy-back, spawn dispatch, and the --remote CLI override. The E2B SDK is never
imported: tests replace the _SANDBOX_FACTORY seam with a fake."""

from __future__ import annotations

import json
from pathlib import Path

import evsys_sdk.triggers.agent as agentmod
import evsys_sdk.triggers.remote as remotemod
from evsys_sdk.config import RemoteAgentConfig, TriggerAgentConfig
from evsys_sdk.triggers.remote import WORKDIR, build_manifest, run_remote


class FakeSandbox:
    """Records writes/commands; serves reads from an in-memory fs the test
    (or a scripted 'agent') populates."""

    def __init__(self):
        self.fs: dict[str, str] = {}
        self.commands: list[str] = []
        self.killed = False
        self.on_run = None  # callable(cmd) -> (exit_code, output), may mutate fs

    def write(self, path: str, content: str) -> None:
        self.fs[path] = content

    def read(self, path: str) -> str | None:
        return self.fs.get(path)

    def run(self, cmd: str, *, timeout_s: float, cwd: str | None = None):
        self.commands.append(cmd)
        if self.on_run:
            return self.on_run(cmd)
        return 0, f"ran: {cmd[:60]}"

    def kill(self) -> None:
        self.killed = True


def _project(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A demo-shaped project: prompt, gate, skills, traces, policy, escalation."""
    cwd = tmp_path
    (cwd / "prompt.txt").write_text("seed prompt")
    (cwd / "gate.py").write_text("# gate fn")
    (cwd / "skills" / "fix-prompt").mkdir(parents=True)
    (cwd / "skills" / "fix-prompt" / "SKILL.md").write_text("# how to fix")
    root = cwd / ".evsys" / "triggers"
    (root / "escalations").mkdir(parents=True)
    (root / "policy.json").write_text(json.dumps(
        {"kind": "failing_answers", "import_path": "gate.py", "params": {}}))
    esc = root / "escalations" / "escalation-00000007.json"
    esc.write_text(json.dumps({"reason": "failure 100%", "trace_ids": ["t1"]}))
    tr = cwd / ".evsys" / "traces" / "claude_code"
    tr.mkdir(parents=True)
    tr.joinpath("traces.jsonl").write_text(
        "".join(json.dumps({"trace_id": f"t{i}"}) + "\n" for i in range(10)))
    return cwd, root, esc


def _cfg(**remote_kw) -> TriggerAgentConfig:
    return TriggerAgentConfig(enabled=True, remote=RemoteAgentConfig(enabled=True, **remote_kw))


class TestManifest:
    def test_contents_and_tail(self, tmp_path):
        cwd, root, esc = _project(tmp_path)
        m = build_manifest(escalation_path=esc, root=root, cwd=cwd,
                           prompt_file="prompt.txt", trace_tail_lines=3)
        assert m[".evsys/triggers/escalations/escalation-00000007.json"]
        assert m[".evsys/triggers/policy.json"]
        assert m["prompt.txt"] == "seed prompt"
        assert m["gate.py"] == "# gate fn"          # from policy.import_path
        assert m["skills/fix-prompt/SKILL.md"] == "# how to fix"
        # window mode tails the traces
        assert len(m[".evsys/traces/claude_code/traces.jsonl"].splitlines()) == 3

    def test_include_all_traces(self, tmp_path):
        cwd, root, esc = _project(tmp_path)
        m = build_manifest(escalation_path=esc, root=root, cwd=cwd,
                           prompt_file="prompt.txt", include_traces="all")
        assert len(m[".evsys/traces/claude_code/traces.jsonl"].splitlines()) == 10


class TestRunRemote:
    def _wire(self, monkeypatch, boxes: list[FakeSandbox]):
        it = iter(boxes)
        monkeypatch.setattr(remotemod, "_SANDBOX_FACTORY", lambda cfg, envs: next(it))

    def test_no_verdict_single_stage(self, tmp_path, monkeypatch):
        cwd, root, esc = _project(tmp_path)
        sbx = FakeSandbox()
        self._wire(monkeypatch, [sbx])
        result = run_remote(esc, agent_cfg=_cfg(), root=root, cwd=cwd,
                            verdict_path=root / "verdicts" / "escalation-00000007.json",
                            log_file=root / "agent-runs" / "escalation-00000007.log")
        # setup + claude ran, in the workdir layout, sandbox torn down
        assert "npm install -g @anthropic-ai/claude-code" in sbx.commands[0]
        assert sbx.commands[1].startswith("claude -p")
        assert f"{WORKDIR}/.evsys/triggers/escalations/escalation-00000007.json" in sbx.commands[1]
        assert sbx.killed
        # agent wrote nothing → no verdict, no stage 2
        assert result["stage2_exit"] is None and result["artifacts"] == []
        assert "remote trigger-agent" in (root / "agent-runs" / "escalation-00000007.log").read_text()

    def test_yes_verdict_two_sandboxes_and_copy_back(self, tmp_path, monkeypatch):
        cwd, root, esc = _project(tmp_path)
        s1, s2 = FakeSandbox(), FakeSandbox()

        def agent1(cmd):  # the trigger agent: verdict YES + retuned policy
            s1.fs[f"{WORKDIR}/.evsys/triggers/verdicts/escalation-00000007.json"] = json.dumps(
                {"worth_autoresearch": True, "hypothesis": "add format line"})
            s1.fs[f"{WORKDIR}/.evsys/triggers/policy.json"] = json.dumps(
                {"kind": "failing_answers", "import_path": "gate.py", "params": {"cooldown": 99}})
            return 0, "verdict written"

        def agent2(cmd):  # autoresearch: rewrites the prompt
            s2.fs[f"{WORKDIR}/prompt.txt"] = "IMPROVED prompt"
            return 0, "rewrote prompt"

        s1.on_run, s2.on_run = agent1, agent2
        self._wire(monkeypatch, [s1, s2])
        result = run_remote(esc, agent_cfg=_cfg(), root=root, cwd=cwd,
                            verdict_path=root / "verdicts" / "escalation-00000007.json",
                            log_file=root / "agent-runs" / "e.log")
        # copy-back landed on the host: verdict, retuned policy, rewritten prompt
        assert json.loads((root / "verdicts" / "escalation-00000007.json").read_text())[
            "worth_autoresearch"] is True
        assert json.loads((root / "policy.json").read_text())["params"]["cooldown"] == 99
        assert (cwd / "prompt.txt").read_text() == "IMPROVED prompt"
        # stage 2 ran in its OWN sandbox, with the autoresearch mission + skills staged
        assert result["stage2_exit"] == 0 and s2.killed
        assert any("autoresearch agent" in c for c in s2.commands)
        assert f"{WORKDIR}/skills/fix-prompt/SKILL.md" in s2.fs

    def test_no_verdict_means_no_stage2(self, tmp_path, monkeypatch):
        cwd, root, esc = _project(tmp_path)
        s1 = FakeSandbox()

        def agent1(cmd):
            s1.fs[f"{WORKDIR}/.evsys/triggers/verdicts/escalation-00000007.json"] = json.dumps(
                {"worth_autoresearch": False, "reasoning": "noise"})
            return 0, "no"

        s1.on_run = agent1
        self._wire(monkeypatch, [s1])  # a second sandbox request would StopIteration
        result = run_remote(esc, agent_cfg=_cfg(), root=root, cwd=cwd,
                            verdict_path=root / "verdicts" / "escalation-00000007.json",
                            log_file=root / "agent-runs" / "e.log")
        assert result["stage2_exit"] is None

    def test_setup_failure_raises_and_kills(self, tmp_path, monkeypatch):
        cwd, root, esc = _project(tmp_path)
        sbx = FakeSandbox()
        sbx.on_run = lambda cmd: (1, "npm exploded")
        self._wire(monkeypatch, [sbx])
        import pytest

        with pytest.raises(RuntimeError, match="setup_cmd failed"):
            run_remote(esc, agent_cfg=_cfg(), root=root, cwd=cwd,
                       verdict_path=root / "verdicts" / "v.json",
                       log_file=root / "agent-runs" / "e.log")
        assert sbx.killed


class TestSpawnDispatch:
    def test_spawn_routes_to_remote_when_enabled(self, tmp_path, monkeypatch):
        cwd, root, esc = _project(tmp_path)
        called = {}

        def fake_spawn_remote(escalation_path, **kw):
            called.update(kw, escalation=escalation_path)
            return "REMOTE"

        monkeypatch.setattr(remotemod, "spawn_remote", fake_spawn_remote)
        out = agentmod.spawn(esc, agent_cfg=_cfg(), root=root, cwd=cwd, detach=False)
        assert out == "REMOTE" and called["detach"] is False
        assert called["verdict_path"].name == "escalation-00000007.json"

    def test_spawn_stays_local_when_disabled(self, tmp_path, monkeypatch):
        cwd, root, esc = _project(tmp_path)
        launches = []
        monkeypatch.setattr(agentmod, "_LAUNCH",
                            lambda cmd, **kw: launches.append(cmd) or None)
        agentmod.spawn(esc, agent_cfg=TriggerAgentConfig(enabled=True), root=root, cwd=cwd)
        assert launches and launches[0][0] == "claude"

    def test_detached_remote_runs_in_thread(self, tmp_path, monkeypatch):
        cwd, root, esc = _project(tmp_path)
        sbx = FakeSandbox()
        monkeypatch.setattr(remotemod, "_SANDBOX_FACTORY", lambda cfg, envs: sbx)
        t = remotemod.spawn_remote(esc, agent_cfg=_cfg(), root=root, cwd=cwd,
                                   verdict_path=root / "verdicts" / "v.json",
                                   log_file=root / "agent-runs" / "e.log", detach=True)
        t.join(timeout=10)
        assert not t.is_alive() and sbx.killed


class TestCLI:
    def test_remote_flag_overrides_config(self, tmp_path, monkeypatch):
        import argparse

        import yaml

        from evsys_sdk import cli

        cwd, root, esc = _project(tmp_path)
        cfg_path = tmp_path / "system.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "trigger": {"kind": "failing_answers", "agent": {"enabled": True}},
        }))
        seen = {}
        import evsys_sdk.triggers as triggers_pkg

        def fake_spawn(escalation, *, agent_cfg, root, detach):
            seen["remote_enabled"] = agent_cfg.remote.enabled
            return type("P", (), {"pid": 1, "returncode": 0, "stdout": ""})()

        # _cmd_trigger_agent does `from .triggers import spawn` at call time,
        # so patching the package attribute is enough.
        monkeypatch.setattr(triggers_pkg, "spawn", fake_spawn)
        ns = argparse.Namespace(config=str(cfg_path), escalation=str(esc),
                                detach=True, print_command=False, remote=True)
        cli._cmd_trigger_agent(ns)
        assert seen["remote_enabled"] is True

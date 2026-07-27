"""Headless trigger-agent invocation — the command builder, the driver's detached
spawn on escalation, config gating, and error isolation.

The real ``claude`` is never shelled out to: tests monkeypatch the ``_LAUNCH``
seam and record what would have been spawned.
"""

from __future__ import annotations

import json

import evsys_sdk.triggers.agent as agentmod
from evsys_sdk.config import TriggerAgentConfig, TriggerConfig
from evsys_sdk.protocols import TriggerDecision
from evsys_sdk.registry import register_trigger
from evsys_sdk.trace_types import Trace
from evsys_sdk.triggers import (
    LocalTriggerStore,
    TriggerDriver,
    TriggerPolicy,
    build_command,
    resolve_hook,
)


@register_trigger("ag_always")
class AgAlways:
    name = "ag_always"

    class Config:
        def __init__(self, **kw):
            pass

    def __init__(self, **params):
        pass

    def evaluate(self, state) -> TriggerDecision:
        return TriggerDecision(True, "always", {"x": 1}, [s["trace_id"] for s in state.window])


def mk_trace(i: int) -> Trace:
    return Trace(trace_id=f"t{i}", messages=[{"role": "user", "content": "q"}],
                 metadata={"source": "test", "status": "error"})


def _record_launch(monkeypatch):
    """Patch the launch seam; return a list that captures (cmd, kwargs)."""
    calls: list[dict] = []

    def fake(cmd, *, cwd, log_file, detach, env=None):
        calls.append({"cmd": cmd, "cwd": str(cwd), "detach": detach, "env": env or {}})
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("stub")

        class P:  # pretend Popen
            pid = 4242
        return P()

    monkeypatch.setattr(agentmod, "_LAUNCH", fake)
    return calls


# 1. Command builder --------------------------------------------------------

def test_build_command_shape():
    cfg = TriggerAgentConfig(enabled=True, model="claude-opus-4-8", plugin_dir="/plug",
                             extra_args=["--verbose"])
    cmd = build_command("/s/escalations/escalation-00000010.json", agent_cfg=cfg,
                        root="/s", verdict_path="/s/verdicts/e.json")
    assert cmd[0] == "claude" and cmd[1] == "-p"
    assert "--permission-mode" in cmd and "acceptEdits" in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-opus-4-8"
    assert cmd[cmd.index("--plugin-dir") + 1] == "/plug"
    assert cmd[-1] == "--verbose"
    prompt = cmd[2]
    assert "escalation-00000010.json" in prompt and "training-decider" in prompt


def test_build_command_autoresearch_off():
    cfg = TriggerAgentConfig(enabled=True, autoresearch=False)
    prompt = build_command("/s/e.json", agent_cfg=cfg, root="/s", verdict_path="/s/v.json")[2]
    assert "Do NOT launch autoresearch" in prompt


# 2. Driver spawns on escalation (gated by config) -------------------------

def test_driver_spawns_agent_on_escalation(tmp_path, monkeypatch):
    calls = _record_launch(monkeypatch)
    store = LocalTriggerStore(tmp_path)
    drv = TriggerDriver(store, seed_policy=TriggerPolicy(kind="ag_always", every_n=1),
                        agent_cfg=TriggerAgentConfig(enabled=True))
    drv(mk_trace(0), None)
    assert len(calls) == 1  # spawned exactly once, detached
    assert calls[0]["detach"] is True
    assert calls[0]["cmd"][0] == "claude"
    log = [json.loads(x) for x in store.log_path().read_text().splitlines()]
    assert any(r["event"] == "spawn" for r in log)


def test_driver_does_not_spawn_when_disabled(tmp_path, monkeypatch):
    calls = _record_launch(monkeypatch)
    store = LocalTriggerStore(tmp_path)
    drv = TriggerDriver(store, seed_policy=TriggerPolicy(kind="ag_always", every_n=1),
                        agent_cfg=TriggerAgentConfig(enabled=False))
    drv(mk_trace(0), None)
    assert calls == []  # escalation still written, but no agent spawned
    assert list(store.escalations_dir().glob("*.json"))


def test_driver_no_agent_cfg_is_fine(tmp_path, monkeypatch):
    calls = _record_launch(monkeypatch)
    store = LocalTriggerStore(tmp_path)
    drv = TriggerDriver(store, seed_policy=TriggerPolicy(kind="ag_always", every_n=1))  # agent_cfg=None
    drv(mk_trace(0), None)
    assert calls == []


# 3. resolve_hook threads the agent config through -------------------------

def test_resolve_hook_passes_agent_cfg(tmp_path, monkeypatch):
    calls = _record_launch(monkeypatch)
    cfg = TriggerConfig(kind="ag_always", every_n=1, state_dir=str(tmp_path),
                        agent=TriggerAgentConfig(enabled=True, model="m"))
    hook = resolve_hook(cfg)
    hook(mk_trace(0), None)
    assert len(calls) == 1 and "m" in calls[0]["cmd"]


# 4. A failing launch never kills ingestion --------------------------------

def test_spawn_failure_is_isolated(tmp_path, monkeypatch):
    def boom(cmd, *, cwd, log_file, detach, env=None):
        raise OSError("claude not found")

    monkeypatch.setattr(agentmod, "_LAUNCH", boom)
    store = LocalTriggerStore(tmp_path)
    drv = TriggerDriver(store, seed_policy=TriggerPolicy(kind="ag_always", every_n=1),
                        agent_cfg=TriggerAgentConfig(enabled=True))
    drv(mk_trace(0), None)  # must not raise
    log = [json.loads(x) for x in store.log_path().read_text().splitlines()]
    assert any(r["event"] == "spawn_error" for r in log)


# 5. The real _launch seam (covered with a trivial python cmd, not claude) --

def test_real_launch_foreground(tmp_path):
    import sys

    log_file = tmp_path / "runs" / "fg.log"
    proc = agentmod._launch([sys.executable, "-c", "print('hi')"], cwd=tmp_path,
                            log_file=log_file, detach=False)
    assert proc.returncode == 0 and "hi" in log_file.read_text()


def test_real_launch_detached(tmp_path):
    import sys

    log_file = tmp_path / "runs" / "bg.log"
    p = agentmod._launch([sys.executable, "-c", "print('bg')"], cwd=tmp_path,
                         log_file=log_file, detach=True)
    p.wait()  # let the detached process finish so the log is flushed
    assert log_file.read_text().strip() == "bg"


# 6. Foreground spawn returns the completed process ------------------------

def test_spawn_foreground_returns_proc(tmp_path, monkeypatch):
    def fake(cmd, *, cwd, log_file, detach, env=None):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("ran")
        assert detach is False

        class Done:
            returncode = 0
            stdout = "verdict written"
        return Done()

    monkeypatch.setattr(agentmod, "_LAUNCH", fake)
    proc = agentmod.spawn(tmp_path / "escalations" / "e.json",
                          agent_cfg=TriggerAgentConfig(enabled=True), root=tmp_path, detach=False)
    assert proc.returncode == 0 and proc.stdout == "verdict written"


# 7. Prompt snapshot at spawn time -----------------------------------------

def test_spawn_snapshots_prompt(tmp_path, monkeypatch):
    _record_launch(monkeypatch)
    root = tmp_path / ".evsys" / "triggers"
    (tmp_path / "prompt.txt").write_text("old prompt\n")
    agentmod.spawn(root / "escalations" / "escalation-00000003.json",
                   agent_cfg=TriggerAgentConfig(enabled=True), root=root)  # default cwd = root/../..
    snap = root / "prompt-snapshots" / "escalation-00000003.txt"
    assert snap.read_text() == "old prompt\n"


def test_spawn_snapshots_custom_prompt_file(tmp_path, monkeypatch):
    _record_launch(monkeypatch)
    (tmp_path / "sys.md").write_text("custom artifact")
    agentmod.spawn(tmp_path / "escalations" / "e.json",
                   agent_cfg=TriggerAgentConfig(enabled=True, prompt_file="sys.md"),
                   root=tmp_path, cwd=tmp_path)
    assert (tmp_path / "prompt-snapshots" / "e.txt").read_text() == "custom artifact"


def test_spawn_no_prompt_file_no_snapshot(tmp_path, monkeypatch):
    _record_launch(monkeypatch)
    agentmod.spawn(tmp_path / "escalations" / "e.json",
                   agent_cfg=TriggerAgentConfig(enabled=True), root=tmp_path, cwd=tmp_path)
    assert not (tmp_path / "prompt-snapshots").exists()


# 8. Distill mode ------------------------------------------------------------

def test_distill_mode_selects_distill_prompt():
    cfg = TriggerAgentConfig(enabled=True, mode="distill",
                             distill={"experiment_template": "tpl/opd.yaml",
                                      "holdout_fraction": 0.25})
    prompt = build_command("/s/escalations/escalation-00000001.json", agent_cfg=cfg,
                           root="/s", verdict_path="/s/v.json")[2]
    assert "distiller agent" in prompt
    assert "tpl/opd.yaml" in prompt and "0.25" in prompt
    # the two hard rules are spelled out
    assert "Do NOT invoke the `training-decider`" in prompt
    assert "distill-traces" in prompt
    # verdict-mode phrasing absent
    assert "autoresearch budget" not in prompt


def test_verdict_mode_unchanged_by_distill_fields():
    cfg = TriggerAgentConfig(enabled=True, mode="verdict")
    prompt = build_command("/s/e.json", agent_cfg=cfg, root="/s", verdict_path="/s/v.json")[2]
    assert "trigger agent" in prompt and "distiller" not in prompt


def test_prompt_template_overrides_distill_default():
    cfg = TriggerAgentConfig(enabled=True, mode="distill", prompt_template="CUSTOM {escalation_path}")
    prompt = build_command("/s/e.json", agent_cfg=cfg, root="/s", verdict_path="/s/v.json")[2]
    assert prompt == "CUSTOM /s/e.json"

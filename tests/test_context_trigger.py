"""The context gate — the SAME trigger machinery (fn, driver, escalation, agent
spawn) run over ingested context instead of traces."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from evsys_sdk.config import SystemConfig
from evsys_sdk.ingest import run_all_once
from evsys_sdk.protocols import TriggerDecision
from evsys_sdk.registry import _triggers, register_trigger
from evsys_sdk.triggers.state import LocalTriggerStore


def _corpus(root):
    (root / "user-a").mkdir(parents=True)
    (root / "user-a" / "m1.txt").write_text("The export button is broken on billing.")
    (root / "user-b").mkdir(parents=True)
    (root / "user-b" / "note.txt").write_text("Prefers concise answers.")


def test_context_gate_escalates_via_same_machinery(tmp_path):
    """A @register_trigger fn over context items escalates and writes an escalation
    event — identical to the trace gate, just fed context."""

    @register_trigger("ctx_needle")
    class CtxNeedle:
        name = "ctx_needle"

        class Config(BaseModel):
            model_config = ConfigDict(extra="forbid")
            needle: str = "broken"

        def __init__(self, **params):
            self.cfg = self.Config(**params)

        def evaluate(self, state) -> TriggerDecision:
            hits = [i for i in state.window if self.cfg.needle in i.get("content", "")]
            return TriggerDecision(bool(hits), f"{len(hits)} item(s) mention '{self.cfg.needle}'",
                                   signal={"hits": len(hits)},
                                   trace_ids=[i["item_id"] for i in hits])

    try:
        _corpus(tmp_path / "ctx")
        ct_dir = tmp_path / ".evsys" / "context_triggers"
        cfg = SystemConfig(
            context={"context_sources": [{"kind": "directory",
                                          "params": {"path": str(tmp_path / "ctx")},
                                          "state_dir": str(tmp_path / ".evsys" / "context")}]},
            context_trigger={"kind": "ctx_needle", "params": {"needle": "broken"},
                             "every_n": 1, "state_dir": str(ct_dir),
                             "agent": {"enabled": False}},  # gate only; no claude spawn in a test
        )
        res = run_all_once(cfg)
        assert res["context:directory"] == 2                 # both items ingested + cached

        store = LocalTriggerStore(ct_dir)
        escalations = list(store.escalations_dir().glob("*.json"))
        assert escalations, "context gate should have escalated on the 'broken' item"
        import json
        event = json.loads(escalations[0].read_text())
        assert "broken" in event["reason"]
        assert event["trace_ids"] == ["user-a/m1.txt"]       # the implicated context item
        # the gate wrote its own policy.json under the CONTEXT trigger dir (not the trace one)
        assert store.read_policy().kind == "ctx_needle"
    finally:
        _triggers.unregister("ctx_needle")


def test_trace_and_context_gates_are_independent(tmp_path):
    """Configuring both a `trigger` and a `context_trigger` keeps their state
    dirs (policy/escalations) separate."""
    @register_trigger("ctx_all")
    class CtxAll:
        name = "ctx_all"
        Config = BaseModel

        def __init__(self, **params):
            pass

        def evaluate(self, state) -> TriggerDecision:
            return TriggerDecision(True, "always", trace_ids=[i["item_id"] for i in state.window])

    try:
        _corpus(tmp_path / "ctx")
        cfg = SystemConfig(
            context={"context_sources": [{"kind": "directory",
                                          "params": {"path": str(tmp_path / "ctx")},
                                          "state_dir": str(tmp_path / ".evsys" / "context")}]},
            context_trigger={"kind": "ctx_all", "every_n": 1,
                             "state_dir": str(tmp_path / ".evsys" / "context_triggers"),
                             "agent": {"enabled": False}},
        )
        run_all_once(cfg)
        assert list((tmp_path / ".evsys" / "context_triggers" / "escalations").glob("*.json"))
        # the trace-trigger dir was never created (no trace trigger configured)
        assert not (tmp_path / ".evsys" / "triggers").exists()
    finally:
        _triggers.unregister("ctx_all")

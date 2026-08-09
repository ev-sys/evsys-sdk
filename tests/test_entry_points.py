"""Third-party extensions load via Python entry points — including triggers and
trace sources, so a pip-installed gate/adapter auto-registers with no config."""

from __future__ import annotations

import evsys_sdk._entry_points as ep_mod
from evsys_sdk.registry import _triggers, list_triggers


def test_every_registry_kind_has_an_entry_point_group():
    """A user-extensible registry that isn't in _GROUPS can't be shipped as an
    installed plugin — guard against a new kind forgetting its group."""
    for group in ("evsys_sdk.triggers", "evsys_sdk.trace_sources",
                  "evsys_sdk.algorithms", "evsys_sdk.transforms"):
        assert group in ep_mod._GROUPS


def test_load_group_imports_and_registers_a_trigger(monkeypatch):
    """_load_group loads each entry point → its @register_trigger fires → the
    fn is in the registry, exactly like a built-in."""
    class _FakeEP:
        name = "ep_gate"

        def load(self):
            from pydantic import BaseModel

            from evsys_sdk.protocols import TriggerDecision
            from evsys_sdk.registry import register_trigger

            @register_trigger("ep_gate")
            class EPGate:
                name = "ep_gate"
                Config = BaseModel

                def __init__(self, **params):
                    pass

                def evaluate(self, state) -> TriggerDecision:
                    return TriggerDecision(False, "from an installed plugin")

    monkeypatch.setattr(
        ep_mod, "entry_points",
        lambda group=None: [_FakeEP()] if group == "evsys_sdk.triggers" else [],
    )
    try:
        ep_mod._load_group("evsys_sdk.triggers")
        assert "ep_gate" in list_triggers()  # auto-registered, no import_path
    finally:
        _triggers.unregister("ep_gate")


def test_load_group_swallows_a_broken_plugin(monkeypatch):
    """A third-party import error is logged, never fatal (one bad plugin must
    not break `import evsys_sdk`)."""
    class _BoomEP:
        name = "boom"

        def load(self):
            raise ImportError("bad plugin")

    monkeypatch.setattr(ep_mod, "entry_points", lambda group=None: [_BoomEP()])
    ep_mod._load_group("evsys_sdk.triggers")  # must not raise

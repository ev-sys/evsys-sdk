"""The trigger — the cheap, always-on gate over ingested traces (Layer 2).

A two-stage gate on the continual-learning loop: a **deterministic fn** runs over
accumulated trace state every ``policy.every_n`` traces and, when it decides so,
emits an **escalation event** for the (heavier) trigger agent to judge. This
package ships the *mechanism* — the registry, the editable policy, the
accumulated state, the driver, and the logging — but **no built-in fns**: the
deterministic fn is registered by the researcher, or authored by the trigger
agent itself (the self-improving gate). Register one exactly like any other
extension::

    from evsys_sdk import register_trigger
    from evsys_sdk.protocols import TriggerDecision

    @register_trigger("my_gate")
    class MyGate:
        name = "my_gate"
        class Config(BaseModel):  # extra="forbid"
            threshold: float = 0.4
        def __init__(self, **params): self.cfg = self.Config(**params)
        def evaluate(self, state) -> TriggerDecision: ...
"""

from __future__ import annotations

from .agent import build_command, spawn
from .driver import TriggerDriver
from .runtime import build_trigger, resolve_hook
from .state import LocalTriggerStore, TriggerPolicy, TriggerState

__all__ = [
    "LocalTriggerStore",
    "TriggerDriver",
    "TriggerPolicy",
    "TriggerState",
    "build_command",
    "build_trigger",
    "resolve_hook",
    "spawn",
]

"""The trigger — the cheap, always-on gate over ingested traces (Layer 2).

A two-stage gate on the continual-learning loop: a **deterministic fn** (this
package) runs over accumulated trace state every ``policy.every_n`` traces and,
when a threshold trips, emits an **escalation event** for the (heavier) trigger
agent to judge. This slice ships the deterministic gate + its state/policy/logging;
the headless agent invocation is a follow-up.

Importing this package registers the built-in fns (side-effect import below), so
``@register_trigger`` fires on ``import evsys_sdk``.
"""

from __future__ import annotations

# Side-effect import: registers failure_rate / feedback_drop / volume / novelty.
from . import builtins as _builtins  # noqa: F401
from .driver import TriggerDriver
from .runtime import build_trigger, resolve_hook
from .state import LocalTriggerStore, TriggerPolicy, TriggerState

__all__ = [
    "LocalTriggerStore",
    "TriggerDriver",
    "TriggerPolicy",
    "TriggerState",
    "build_trigger",
    "resolve_hook",
]

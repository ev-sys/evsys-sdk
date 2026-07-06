"""The gate driver — a :data:`~evsys_sdk.trace_sources.base.TraceHook` that binds
the deterministic trigger to Layer-1 ingestion.

On every ingested trace the driver:
  1. re-reads ``policy.json`` (so an agent retune takes effect without a restart),
  2. pushes a compact summary of the trace into the accumulated state,
  3. every ``policy.every_n`` traces, resolves the fn from the policy and calls
     ``evaluate(state)`` over the ENTIRE state,
  4. on escalation, writes an escalation event + an activity-log row (spawning the
     trigger agent on that event is the follow-up slice).

All of this runs inside Layer-1's error-isolated ``_dispatch`` — a raising trigger
never kills ingestion — but the driver also guards its own eval so state still
persists across a bad evaluation.
"""

from __future__ import annotations

from typing import Any

from ..logger import get_logger
from .runtime import build_trigger
from .state import LocalTriggerStore, TriggerPolicy

log = get_logger(__name__)


class TriggerDriver:
    """Callable ``(trace, ctx) -> None`` — the deterministic gate as a TraceHook."""

    def __init__(self, store: LocalTriggerStore, *, seed_policy: TriggerPolicy | None = None) -> None:
        self.store = store
        if seed_policy is not None:
            self.store.seed_policy(seed_policy)

    def __call__(self, trace: Any, ctx: Any = None) -> None:
        policy = self.store.read_policy() or TriggerPolicy()
        state = self.store.read_state()
        state.push(trace, policy)

        if state.counters.get("since_last_eval", 0) >= policy.every_n:
            state.counters["since_last_eval"] = 0
            try:
                trigger = build_trigger(policy)
                decision = trigger.evaluate(state)
            except Exception as e:  # a bad fn/params never kills ingestion
                log.warning("[trigger] evaluate failed (kind=%s): %s", policy.kind, e)
                self.store.append_log({"event": "error", "kind": policy.kind, "error": str(e)})
                self.store.write_state(state)
                return

            self.store.append_log({
                "event": "evaluate",
                "kind": policy.kind,
                "escalate": decision.escalate,
                "reason": decision.reason,
                "signal": decision.signal,
                "n_seen": state.counters.get("n_seen"),
            })
            if decision.escalate:
                seq = int(state.counters.get("n_seen", 0))
                state.counters["since_last_escalation"] = 0
                event = {
                    "reason": decision.reason,
                    "signal": decision.signal,
                    "trace_ids": decision.trace_ids,
                    "kind": policy.kind,
                    "n_seen": seq,
                    "aggregates": state.aggregates,
                }
                path = self.store.write_escalation(event, seq=seq)
                self.store.append_log({"event": "escalation", "reason": decision.reason,
                                       "signal": decision.signal, "path": str(path)})
                log.info("[trigger] ESCALATE: %s", decision.reason)

        self.store.write_state(state)


__all__ = ["TriggerDriver"]

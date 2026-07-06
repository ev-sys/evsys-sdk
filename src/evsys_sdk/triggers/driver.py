"""The gate driver — a :data:`~evsys_sdk.trace_sources.base.TraceHook` that binds
the deterministic trigger to Layer-1 ingestion.

On every ingested trace the driver:
  1. re-reads ``policy.json`` (so an agent retune takes effect without a restart),
  2. pushes the raw trace into the accumulated state's rolling window,
  3. every ``policy.every_n`` traces, resolves the fn from the policy and calls
     ``evaluate(state)`` over the ENTIRE state,
  4. on escalation, writes an escalation event + an activity-log row, and — when a
     trigger agent is configured (``trigger.agent.enabled``) — spawns it detached
     (``claude -p``) on that event.

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

    def __init__(
        self,
        store: LocalTriggerStore,
        *,
        seed_policy: TriggerPolicy | None = None,
        agent_cfg: Any = None,
        cwd: Any = None,
    ) -> None:
        self.store = store
        self.agent_cfg = agent_cfg
        self.cwd = cwd
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
                }
                path = self.store.write_escalation(event, seq=seq)
                self.store.append_log({"event": "escalation", "reason": decision.reason,
                                       "signal": decision.signal, "path": str(path)})
                log.info("[trigger] ESCALATE: %s", decision.reason)
                self._maybe_spawn_agent(path)

        self.store.write_state(state)

    def _maybe_spawn_agent(self, escalation_path: Any) -> None:
        """Hand the escalation to the headless trigger agent when configured.
        Detached + error-isolated: launching Claude never blocks or kills ingestion."""
        if self.agent_cfg is None or not getattr(self.agent_cfg, "enabled", False):
            return
        from .agent import spawn

        try:
            spawn(escalation_path, agent_cfg=self.agent_cfg, root=self.store.root, cwd=self.cwd)
            self.store.append_log({"event": "spawn", "escalation": str(escalation_path),
                                   "bin": getattr(self.agent_cfg, "claude_bin", "claude")})
        except Exception as e:  # a failed launch never kills ingestion
            log.warning("[trigger] agent spawn failed: %s", e)
            self.store.append_log({"event": "spawn_error", "escalation": str(escalation_path),
                                   "error": str(e)})


__all__ = ["TriggerDriver"]

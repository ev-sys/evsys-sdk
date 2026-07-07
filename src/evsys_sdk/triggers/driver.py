"""The gate driver — a :data:`~evsys_sdk.trace_sources.base.TraceHook` that binds
the deterministic trigger to Layer-1 ingestion.

On every ingested trace the driver:
  1. re-reads ``policy.json`` (so an agent retune takes effect without a restart),
  2. pushes the raw trace into the accumulated state's rolling window,
  3. every ``policy.every_n`` traces: hot-reloads the fn's code if the policy's
     ``import_path`` or the file changed (the agent rewriting its own fn), then
     resolves the fn from the policy and calls ``evaluate(state)`` over the state,
  4. on escalation, writes an escalation event + an activity-log row (spawning the
     trigger agent on that event is the follow-up slice).

All of this runs inside Layer-1's error-isolated ``_dispatch`` — a raising trigger
never kills ingestion — but the driver also guards its own eval so state still
persists across a bad evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..logger import get_logger
from .runtime import build_trigger, import_trigger_code
from .state import LocalTriggerStore, TriggerPolicy

log = get_logger(__name__)


class TriggerDriver:
    """Callable ``(trace, ctx) -> None`` — the deterministic gate as a TraceHook."""

    def __init__(self, store: LocalTriggerStore, *, seed_policy: TriggerPolicy | None = None) -> None:
        self.store = store
        if seed_policy is not None:
            self.store.seed_policy(seed_policy)
        # what's currently loaded, so we only re-import when it actually changes.
        # resolve_hook already imported the seed's fn, so start in sync with it.
        self._loaded = self._fn_sig(seed_policy)

    @staticmethod
    def _fn_sig(policy: TriggerPolicy | None) -> tuple | None:
        """Identity of the loaded fn code: (import_path, file-mtime). A changed
        path OR a rewritten file (new mtime) means reload."""
        ip = getattr(policy, "import_path", None) if policy else None
        if not ip:
            return None
        p = Path(ip)
        try:
            mtime = p.stat().st_mtime if p.suffix == ".py" and p.exists() else None
        except OSError:
            mtime = None
        return (ip, mtime)

    def _maybe_reload_fn(self, policy: TriggerPolicy) -> None:
        """Hot-reload the fn's code when the agent has repointed or rewritten it."""
        if not policy.import_path:
            return
        sig = self._fn_sig(policy)
        if sig == self._loaded:
            return
        try:
            import_trigger_code(policy.import_path, kind=policy.kind)
            self._loaded = sig
            self.store.append_log({"event": "reload_fn", "kind": policy.kind,
                                   "import_path": policy.import_path})
            log.info("[trigger] hot-reloaded fn '%s' from %s", policy.kind, policy.import_path)
        except Exception as e:  # a bad rewrite must not kill ingestion
            log.warning("[trigger] fn reload failed (%s): %s", policy.import_path, e)
            self.store.append_log({"event": "reload_error", "kind": policy.kind,
                                   "import_path": policy.import_path, "error": str(e)})

    def __call__(self, trace: Any, ctx: Any = None) -> None:
        policy = self.store.read_policy() or TriggerPolicy()
        state = self.store.read_state()
        state.push(trace, policy)

        if state.counters.get("since_last_eval", 0) >= policy.every_n:
            state.counters["since_last_eval"] = 0
            self._maybe_reload_fn(policy)
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

        self.store.write_state(state)


__all__ = ["TriggerDriver"]

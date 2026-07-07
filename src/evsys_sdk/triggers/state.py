"""The trigger's persistent surface — the editable **policy**, the accumulated
**state**, and a thread-safe **local store** for both (plus the activity log).

Three pieces, mirroring ``trace_sources/store.py``:

  * :class:`TriggerPolicy` — the agent-editable knobs (which fn, cadence, window).
    Config *seeds* it; thereafter ``policy.json`` is authoritative and re-read live
    each cycle, so a trigger-agent retune survives restarts and takes effect
    without one.
  * :class:`TriggerState` — what the deterministic fn receives: a bounded rolling
    window of the **raw ingested traces** (nothing extracted or interpreted), plus
    the cadence ``counters`` the driver needs, plus a freeform ``extras`` bag the
    fn owns. The SDK does NOT summarize traces or compute aggregates — the fn reads
    the raw traces and reduces them however it wants.
  * :class:`LocalTriggerStore` — persists ``policy.json`` / ``state.json``,
    appends the activity log to ``log.jsonl``, and drops escalation events under
    ``escalations/``. Atomic writes, one lock — same discipline as
    :class:`~evsys_sdk.trace_sources.store.LocalTraceStore`.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TriggerPolicy:
    """The agent-editable surface. Persisted to ``policy.json`` and read live.

    Everything here belongs to the trigger agent — thresholds AND the fn's own
    code location (``import_path``). The agent can author a new ``.py``, point
    ``kind`` + ``import_path`` at it, and the driver hot-loads it on the next
    eval: the self-improving gate owns its logic, not just its knobs.
    """

    kind: str = ""
    """Registry key of the deterministic fn — a researcher- or agent-registered
    ``@register_trigger``. No built-in fns ship; an empty/unregistered kind makes
    ``build_trigger`` raise (the driver catches it and logs an error event)."""
    import_path: str | None = None
    """Where the fn's ``@register_trigger`` code lives (a ``.py`` path or dotted
    module). The driver re-imports it whenever this or the file's mtime changes,
    so an agent-rewritten fn goes live without a daemon restart."""
    params: dict = field(default_factory=dict)
    every_n: int = 20
    window: int = 100

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "import_path": self.import_path,
            "params": self.params,
            "every_n": self.every_n,
            "window": self.window,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TriggerPolicy:
        d = d or {}
        return cls(
            kind=d.get("kind", ""),
            import_path=d.get("import_path"),
            params=dict(d.get("params") or {}),
            every_n=int(d.get("every_n", 20)),
            window=int(d.get("window", 100)),
        )


@dataclass
class TriggerState:
    """What the deterministic fn receives — bounded and JSON-serializable.

    ``window`` holds the last ``policy.window`` **raw traces** exactly as ingested
    (``{trace_id, messages, feedback, metadata}``); the fn reduces them itself.
    ``counters`` is the driver's cadence bookkeeping. ``extras`` is a freeform,
    JSON-serializable bag the fn (agent-authored) owns end to end — it can stash
    **anything** there (rolling aggregates, EWMAs, seen-hashes, a learned
    threshold the agent injected) and it round-trips untouched across evaluations;
    the SDK never reads or writes it. This is the "add anything to the state"
    channel for a self-improving fn.
    """

    window: list[dict] = field(default_factory=list)
    counters: dict = field(default_factory=lambda: {"n_seen": 0, "since_last_eval": 0, "since_last_escalation": 0})
    extras: dict = field(default_factory=dict)

    def push(self, trace: Any, policy: TriggerPolicy) -> None:
        """Append the raw ``trace`` (as a dict), evict past ``policy.window``,
        bump the counters. No extraction, no aggregates."""
        row = trace.to_dict() if hasattr(trace, "to_dict") else dict(trace)
        self.window.append(row)
        if len(self.window) > policy.window:
            self.window = self.window[-policy.window :]
        for k in ("n_seen", "since_last_eval", "since_last_escalation"):
            self.counters[k] = int(self.counters.get(k, 0)) + 1

    def to_dict(self) -> dict:
        return {"window": self.window, "counters": self.counters, "extras": self.extras}

    @classmethod
    def from_dict(cls, d: dict) -> TriggerState:
        d = d or {}
        st = cls(
            window=list(d.get("window") or []),
            counters=dict(d.get("counters") or {}),
            extras=dict(d.get("extras") or {}),
        )
        for k in ("n_seen", "since_last_eval", "since_last_escalation"):
            st.counters.setdefault(k, 0)
        return st


class LocalTriggerStore:
    """Thread-safe local store for the policy, state, activity log + escalations."""

    def __init__(self, root: str | Path = ".evsys/triggers") -> None:
        self.root = Path(root).expanduser()
        self._lock = threading.Lock()

    # -- paths -------------------------------------------------------------
    def policy_path(self) -> Path:
        return self.root / "policy.json"

    def state_path(self) -> Path:
        return self.root / "state.json"

    def log_path(self) -> Path:
        return self.root / "log.jsonl"

    def escalations_dir(self) -> Path:
        return self.root / "escalations"

    # -- policy ------------------------------------------------------------
    def read_policy(self) -> TriggerPolicy | None:
        path = self.policy_path()
        if not path.exists():
            return None
        try:
            return TriggerPolicy.from_dict(json.loads(path.read_text()))
        except Exception:
            return None

    def write_policy(self, policy: TriggerPolicy) -> None:
        self._atomic_json(self.policy_path(), policy.to_dict())

    def seed_policy(self, policy: TriggerPolicy) -> TriggerPolicy:
        """Write ``policy`` only if none exists yet — never clobber an
        agent-edited policy. Returns the live policy."""
        existing = self.read_policy()
        if existing is not None:
            return existing
        self.write_policy(policy)
        return policy

    # -- state -------------------------------------------------------------
    def read_state(self) -> TriggerState:
        path = self.state_path()
        if not path.exists():
            return TriggerState()
        try:
            return TriggerState.from_dict(json.loads(path.read_text()))
        except Exception:
            return TriggerState()

    def write_state(self, state: TriggerState) -> None:
        self._atomic_json(self.state_path(), state.to_dict())

    # -- log + escalations -------------------------------------------------
    def append_log(self, record: dict) -> None:
        path = self.log_path()
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")

    def write_escalation(self, event: dict, *, seq: int) -> Path:
        path = self.escalations_dir() / f"escalation-{seq:08d}.json"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(event, indent=2, default=str))
        return path

    # -- internal ----------------------------------------------------------
    def _atomic_json(self, path: Path, payload: Any) -> None:
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, default=str))
            tmp.replace(path)


__all__ = ["LocalTriggerStore", "TriggerPolicy", "TriggerState"]

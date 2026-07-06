"""The trigger's persistent surface — the editable **policy**, the accumulated
**state**, and a thread-safe **local store** for both (plus the activity log).

Three pieces, mirroring ``trace_sources/store.py``:

  * :class:`TriggerPolicy` — the agent-editable knobs (which fn, cadence, window,
    thresholds, tracked signals). Config *seeds* it; thereafter ``policy.json``
    is authoritative and re-read live each cycle, so a trigger-agent retune
    (a follow-up) survives restarts and takes effect without one.
  * :class:`TriggerState` — the bounded, cheap accumulator the deterministic fn
    reads: a rolling window of compact trace summaries + running aggregates +
    counters + a freeform ``extras`` bag (so a retuned fn can track a *new*
    signal without a schema change).
  * :class:`LocalTriggerStore` — persists ``policy.json`` / ``state.json``,
    appends the activity log to ``log.jsonl``, and drops escalation events under
    ``escalations/``. Atomic writes, one lock — same discipline as
    :class:`~evsys_sdk.trace_sources.store.LocalTraceStore`.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..trace_types import Trace


def _primary_reward(trace: Trace) -> float | None:
    """The trace's headline reward: a whole-trace feedback score if present, else
    the first numeric per-turn score. ``None`` when the trace carries no score."""
    scored = [f for f in trace.feedback if isinstance(f.get("score"), (int, float))]
    if not scored:
        return None
    whole = [f for f in scored if f.get("turn") is None]
    return float((whole or scored)[0]["score"])


def _n_tool_calls(trace: Trace) -> int:
    return sum(len(m.get("tool_calls") or []) for m in trace.messages if m.get("role") == "assistant")


def _summarize(trace: Trace, signals: list[str]) -> dict:
    """A compact, JSON-safe summary of one trace — only the fields the fn needs."""
    full = {
        "trace_id": trace.trace_id,
        "reward": _primary_reward(trace),
        "status": trace.metadata.get("status"),
        "input_sig": (str(trace.input)[:120] if trace.input is not None else None),
        "n_tool_calls": _n_tool_calls(trace),
        "timestamp": trace.metadata.get("timestamp"),
    }
    # ``trace_id`` is always kept (it identifies the summary); ``signals`` selects
    # the rest, so a retuned policy can widen/narrow what state tracks.
    keep = set(signals or []) | {"trace_id"}
    return {k: v for k, v in full.items() if k in keep}


@dataclass
class TriggerPolicy:
    """The agent-editable knobs. Persisted to ``policy.json`` and read live."""

    kind: str = "failure_rate"
    params: dict = field(default_factory=dict)
    every_n: int = 20
    window: int = 100
    signals: list[str] = field(
        default_factory=lambda: ["reward", "status", "input_sig", "n_tool_calls", "timestamp"]
    )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "params": self.params,
            "every_n": self.every_n,
            "window": self.window,
            "signals": self.signals,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TriggerPolicy:
        d = d or {}
        return cls(
            kind=d.get("kind", "failure_rate"),
            params=dict(d.get("params") or {}),
            every_n=int(d.get("every_n", 20)),
            window=int(d.get("window", 100)),
            signals=list(d.get("signals") or []) or TriggerPolicy().signals,
        )


@dataclass
class TriggerState:
    """The accumulated gate state — bounded, cheap, JSON-serializable."""

    window: list[dict] = field(default_factory=list)
    aggregates: dict = field(default_factory=dict)
    counters: dict = field(default_factory=lambda: {"n_seen": 0, "since_last_eval": 0, "since_last_escalation": 0})
    extras: dict = field(default_factory=dict)

    def push(self, trace: Trace, policy: TriggerPolicy) -> None:
        """Append a summary of ``trace``, evict past ``policy.window``, and
        recompute aggregates + counters."""
        self.window.append(_summarize(trace, policy.signals))
        if len(self.window) > policy.window:
            self.window = self.window[-policy.window :]
        for k in ("n_seen", "since_last_eval", "since_last_escalation"):
            self.counters[k] = int(self.counters.get(k, 0)) + 1
        self._recompute()

    def _recompute(self) -> None:
        w = self.window
        rewards = [s["reward"] for s in w if isinstance(s.get("reward"), (int, float))]
        self.aggregates = {
            "window_size": len(w),
            "avg_reward": (sum(rewards) / len(rewards)) if rewards else None,
            "n_failed": sum(1 for s in w if s.get("status") == "error"),
            "status_hist": dict(Counter(s.get("status") for s in w)),
        }

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "aggregates": self.aggregates,
            "counters": self.counters,
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TriggerState:
        d = d or {}
        st = cls(
            window=list(d.get("window") or []),
            aggregates=dict(d.get("aggregates") or {}),
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

"""Base trace source — the generic pull loop, local write, and per-trace hook.

Adapters (e.g. ``langgraph``) subclass this and implement ONLY ``pull_raw`` +
``to_trace``. Everything else — cursor bookkeeping, dedupe, local landing, the
error-isolated per-trace hook, and the ``--watch`` daemon loop — lives here.

The ``hook`` is the seam for later layers: Layer 1 ships the well-defined call
site + a no-op default; Layer 2 (the deterministic trigger fn → trigger agent)
swaps in a real hook.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from ..logger import get_logger
from ..protocols import TraceContext
from ..trace_types import Trace
from .store import LocalTraceStore, _parse_iso

log = get_logger(__name__)

TraceHook = Callable[[Trace, TraceContext], None]


def _duration_seconds(s: str | int | float) -> float:
    """'24h' / '30m' / '90' → seconds."""
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return float(s[:-1]) * units[s[-1]] if s and s[-1] in units else float(s)


def _noop_hook(trace: Trace, ctx: TraceContext) -> None:
    """Default per-trace hook — does nothing. Layer 2 replaces this."""


class BaseTraceSource:
    """Owns the pull loop; adapters implement ``pull_raw`` + ``to_trace``."""

    name: str = ""
    Config: type | None = None

    def __init__(
        self,
        *,
        store: LocalTraceStore,
        hook: TraceHook | None = None,
        spec: Any = None,
        **params: Any,
    ) -> None:
        self.store = store
        self.hook = hook or _noop_hook
        self.spec = spec
        # Validate adapter params against the adapter's Config (loud on a typo).
        self.cfg = self.Config(**params) if self.Config is not None else None
        self._ctx = TraceContext(source=self.name, store=store, spec=spec)

    # -- adapter responsibilities -----------------------------------------

    def pull_raw(self, since: datetime | None) -> Iterable[Any]:
        raise NotImplementedError

    def to_trace(self, raw: Any) -> Trace:
        raise NotImplementedError

    # -- generic orchestration --------------------------------------------

    def _initial_since(self) -> datetime | None:
        """When there is no cursor yet, start from the spec's ``since`` (ISO) or
        ``now - window`` (e.g. '24h'); otherwise pull everything."""
        spec = self.spec
        s = getattr(spec, "since", None) if spec is not None else None
        if s:
            return _parse_iso(s)
        w = getattr(spec, "window", None) if spec is not None else None
        if w:
            from datetime import timedelta

            return datetime.now(UTC) - timedelta(seconds=_duration_seconds(w))
        return None

    def run_once(self, *, limit: int | None = None, since_override: datetime | None = None) -> int:
        """Pull traces newer than the cursor, land the new ones locally, fire the
        hook per new trace, and advance the cursor. Returns the count of new traces."""
        since, seen = self.store.read_cursor(self.name)
        if since is None:
            since = since_override or self._initial_since()
        new_count = 0
        max_ts: str | None = None
        for raw in self.pull_raw(since):
            trace = self.to_trace(raw)
            if trace.trace_id in seen:
                continue
            self.store.write_trace(self.name, trace)
            seen.add(trace.trace_id)
            ts = trace.metadata.get("timestamp") or trace.metadata.get("start_time")
            if ts and (max_ts is None or str(ts) > max_ts):
                max_ts = str(ts)
            self._dispatch(trace)
            new_count += 1
            if limit is not None and new_count >= limit:
                break
        self.store.write_cursor(self.name, max_ts or since, seen)
        return new_count

    def watch(self, interval_s: float, *, stop: Any = None) -> None:
        """Daemon loop: ``run_once`` then sleep ``interval_s``, until ``stop`` is set."""
        while True:
            if stop is not None and stop.is_set():
                break
            try:
                n = self.run_once()
                log.info("[%s] ingested %d new trace(s)", self.name, n)
            except Exception as e:  # keep the daemon alive across a bad pull
                log.warning("[%s] pull failed: %s", self.name, e)
            if stop is not None:
                if stop.wait(interval_s):
                    break
            else:
                time.sleep(interval_s)

    def _dispatch(self, trace: Trace) -> None:
        try:
            self.hook(trace, self._ctx)
        except Exception as e:  # error-isolated — one bad hook never kills ingestion
            log.warning("[%s] trace hook raised on %s: %s", self.name, trace.trace_id, e)


__all__ = ["BaseTraceSource", "TraceHook"]

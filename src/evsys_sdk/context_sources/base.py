"""Base context source — the generic pull loop + local cache.

Adapters (e.g. ``directory``) subclass this and implement ONLY ``pull_raw`` +
``to_item``. Everything else — cursor bookkeeping, dedupe, local landing, and the
``--watch`` daemon loop — lives here. The exact mirror of
:class:`~evsys_sdk.trace_sources.base.BaseTraceSource`, minus the per-trace hook:
context is pull-and-cache; its consumer (the autoresearch agent) reads the cache.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from ..context_types import ContextItem
from ..logger import get_logger
from .store import LocalContextStore, _parse_iso

log = get_logger(__name__)


def _duration_seconds(s: str | int | float) -> float:
    """'24h' / '30m' / '90' → seconds."""
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return float(s[:-1]) * units[s[-1]] if s and s[-1] in units else float(s)


class BaseContextSource:
    """Owns the pull loop; adapters implement ``pull_raw`` + ``to_item``."""

    name: str = ""
    Config: type | None = None

    def __init__(self, *, store: LocalContextStore, spec: Any = None, **params: Any) -> None:
        self.store = store
        self.spec = spec
        self.cfg = self.Config(**params) if self.Config is not None else None

    # -- adapter responsibilities -----------------------------------------

    def pull_raw(self, since: datetime | None) -> Iterable[Any]:
        raise NotImplementedError

    def to_item(self, raw: Any) -> ContextItem:
        raise NotImplementedError

    # -- generic orchestration --------------------------------------------

    def _initial_since(self) -> datetime | None:
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
        """Pull items newer than the cursor, land the new ones locally, advance
        the cursor. Returns the count of new items."""
        since, seen = self.store.read_cursor(self.name)
        if since is None:
            since = since_override or self._initial_since()
        new_count = 0
        max_ts: str | None = None
        for raw in self.pull_raw(since):
            item = self.to_item(raw)
            if item.item_id in seen:
                continue
            self.store.write_item(self.name, item)
            seen.add(item.item_id)
            ts = item.metadata.get("timestamp")
            if ts and (max_ts is None or str(ts) > max_ts):
                max_ts = str(ts)
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
                log.info("[%s] ingested %d new context item(s)", self.name, n)
            except Exception as e:  # keep the daemon alive across a bad pull
                log.warning("[%s] context pull failed: %s", self.name, e)
            if stop is not None:
                if stop.wait(interval_s):
                    break
            else:
                time.sleep(interval_s)


__all__ = ["BaseContextSource"]

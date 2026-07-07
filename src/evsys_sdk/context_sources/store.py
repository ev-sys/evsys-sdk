"""Local landing + incremental pull-cursor for ingested context.

Items append to ``<root>/<source>/items.jsonl``; a small ``cursor.json`` next to
it records the last-pulled timestamp (``since``) + recently-seen item ids so a
re-pull only lands genuinely new context. The exact mirror of
:class:`evsys_sdk.trace_sources.store.LocalTraceStore` — thread-safe, atomic
cursor write — one store per ingestion kind.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from ..context_types import ContextItem


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


class LocalContextStore:
    """Thread-safe local store: appends context items to JSONL + tracks a cursor."""

    def __init__(self, root: str | Path = ".evsys/context") -> None:
        self.root = Path(root).expanduser()
        self._lock = threading.Lock()

    def _source_dir(self, source: str) -> Path:
        return self.root / source

    def items_path(self, source: str) -> Path:
        return self._source_dir(source) / "items.jsonl"

    def cursor_path(self, source: str) -> Path:
        return self._source_dir(source) / "cursor.json"

    def write_item(self, source: str, item: ContextItem) -> None:
        path = self.items_path(source)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(item.to_dict(), default=str) + "\n")

    def read_cursor(self, source: str) -> tuple[datetime | None, set[str]]:
        path = self.cursor_path(source)
        if not path.exists():
            return None, set()
        try:
            data = json.loads(path.read_text())
        except Exception:
            return None, set()
        since = data.get("since")
        dt = _parse_iso(since) if isinstance(since, str) else None
        return dt, set(data.get("seen_ids") or [])

    def write_cursor(
        self,
        source: str,
        since: datetime | str | None,
        seen_ids: set[str],
        *,
        keep: int = 5000,
    ) -> None:
        """Persist the cursor. ``seen_ids`` is bounded to the last ``keep`` ids."""
        path = self.cursor_path(source)
        payload: dict[str, Any] = {
            "since": since.isoformat() if isinstance(since, datetime) else since,
            "seen_ids": list(seen_ids)[-keep:],
        }
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2, default=str))
            tmp.replace(path)


__all__ = ["LocalContextStore"]

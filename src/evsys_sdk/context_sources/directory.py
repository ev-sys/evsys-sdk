"""``directory`` context adapter — pull context from a local folder of text files.

Each ``.txt`` file under ``path`` becomes a
:class:`~evsys_sdk.context_types.ContextItem`: the file body is ``content``, its
**parent folder name** is the ``entity`` (so a layout like ``context/<user-id>/
note.txt`` attributes the item to that user), and the file mtime drives the
incremental cursor.

This is the dependency-free reference adapter (the sibling of the ``langgraph``
trace adapter). Any other source is the same shape — subclass ``BaseContextSource``,
implement ``pull_raw`` + ``to_item``, and ``@register_context_source``.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..context_types import ContextItem
from ..registry import register_context_source
from .base import BaseContextSource


@register_context_source("directory")
class DirectoryContextSource(BaseContextSource):
    """Ingest a folder of text files as context items."""

    name = "directory"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        path: str
        """Root folder to ingest from."""
        glob: str = "**/*.txt"
        """Which files to include (default: all .txt files, recursively)."""

    def pull_raw(self, since: datetime | None) -> Iterable[Any]:
        root = Path(self.cfg.path).expanduser()
        if not root.exists():
            return
        for p in sorted(root.glob(self.cfg.glob), key=lambda q: q.stat().st_mtime if q.is_file() else 0):
            if not p.is_file():
                continue
            mtime = datetime.fromtimestamp(p.stat().st_mtime, UTC)
            if since is not None and mtime <= since:
                continue
            yield (p, root, mtime)

    def to_item(self, raw: Any) -> ContextItem:
        p, root, mtime = raw
        rel = p.relative_to(root)
        entity = rel.parent.name or None  # the folder the file sits in
        return ContextItem(
            item_id=str(rel),
            source=self.name,
            content=p.read_text(errors="replace"),
            entity=entity,
            metadata={"source": self.name, "filename": p.name,
                      "timestamp": mtime.isoformat(), "path": str(p)},
        )


__all__ = ["DirectoryContextSource"]

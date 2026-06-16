"""Shared upload logic for harbor-format directories.

``benchmark_upload`` registers a local ``<dir>/`` (``tasks.jsonl`` +
``metadata.yaml``) with the dashboard. The reusable body lives here and the
wrapper passes in its create/add-rows/list callables, so a future harbor-style
entity can reuse it.

Re-uploading the same content is a no-op (we hash ``tasks.jsonl``);
re-uploading changed content registers a new version.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .benchmark import Benchmark
from .data_types import to_dict

HARBOR_FORMAT = "harbor"


@dataclass(frozen=True)
class HarborUploadResult:
    """Result of one harbor upload call."""

    id: str
    name: str
    version: int
    content_hash: str
    status: str
    """``"created" | "updated" | "unchanged"``."""
    n_tasks: int


def upload_harbor(
    path: str | Path,
    *,
    create_record: Callable[..., dict],
    add_rows: Callable[[str, list[dict]], Any],
    list_existing: Callable[[], list[dict] | None],
    format: str = HARBOR_FORMAT,
) -> HarborUploadResult:
    """Upload (or re-upload) a harbor directory via the supplied store callables.

    Steps:
      1. Load with ``Benchmark.from_dir`` (catches malformed jsonl / metadata).
      2. Hash ``tasks.jsonl`` content for idempotency.
      3. Look up an existing record with the same ``name``:
         - same hash → return ``"unchanged"``.
         - different hash → create a new version row, push tasks.
         - none → create version 1, push tasks.
    """
    bench = Benchmark.from_dir(path)
    assert bench.root is not None  # from_dir always sets root
    content_hash = _hash_file(bench.root / "tasks.jsonl")

    existing = _find_existing(list_existing, bench.name)
    if existing and (existing.get("metadata") or {}).get("content_hash") == content_hash:
        return HarborUploadResult(
            id=str(existing["id"]),
            name=bench.name,
            version=int(existing.get("version") or 1),
            content_hash=content_hash,
            status="unchanged",
            n_tasks=len(bench.tasks),
        )

    next_version = (int(existing["version"]) + 1) if existing else 1
    metadata = {**bench.metadata, "content_hash": content_hash}
    record = create_record(
        name=bench.name,
        format=format,
        version=next_version,
        source_kind="harbor_jsonl",
        metadata=metadata,
    )
    record_id = str(record["id"])

    rows = [to_dict(task) for task in bench.tasks]
    if rows:
        add_rows(record_id, rows)

    return HarborUploadResult(
        id=record_id,
        name=bench.name,
        version=next_version,
        content_hash=content_hash,
        status="updated" if existing else "created",
        n_tasks=len(bench.tasks),
    )


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_existing(list_existing: Callable[[], list[dict] | None], name: str) -> dict | None:
    """Return the highest-version record with ``name``, if any."""
    try:
        records = list_existing() or []
    except Exception:
        return None
    matching = [b for b in records if b.get("name") == name]
    if not matching:
        return None
    return max(matching, key=lambda b: int(b.get("version") or 1))


__all__ = ["HARBOR_FORMAT", "HarborUploadResult", "upload_harbor"]

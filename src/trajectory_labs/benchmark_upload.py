"""Register a local harbor-format benchmark with the dashboard.

Each ``data/benchmark/<name>/`` directory becomes a row in the dashboard's
benchmark table so experiments can reference it by id. Re-uploading the
same content is a no-op (we hash ``tasks.jsonl``); re-uploading changed
content registers a new version.

CLI entry point: ``trajex benchmark upload data/benchmark/<name>``.
Programmatic: ``upload_benchmark(store, path)``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark import Benchmark
from .data_types import to_dict

BENCHMARK_FORMAT = "harbor"


@dataclass(frozen=True)
class UploadResult:
    """Result of one upload call."""

    benchmark_id: str
    name: str
    version: int
    content_hash: str
    status: str
    """``"created" | "updated" | "unchanged"``."""
    n_tasks: int


def upload_benchmark(store: Any, path: str | Path) -> UploadResult:
    """Upload (or re-upload) a harbor benchmark directory.

    Steps:
      1. Load with ``Benchmark.from_dir`` (catches malformed jsonl / metadata).
      2. Hash ``tasks.jsonl`` content for idempotency.
      3. Look up an existing benchmark with the same ``name``:
         - same hash → return ``"unchanged"``.
         - different hash → create a new version row, push tasks.
         - none → create version 1, push tasks.
    """
    bench = Benchmark.from_dir(path)
    assert bench.root is not None  # from_dir always sets root
    tasks_path = bench.root / "tasks.jsonl"
    content_hash = _hash_file(tasks_path)

    existing = _find_existing(store, bench.name)
    if existing and (existing.get("metadata") or {}).get("content_hash") == content_hash:
        return UploadResult(
            benchmark_id=str(existing["id"]),
            name=bench.name,
            version=int(existing.get("version") or 1),
            content_hash=content_hash,
            status="unchanged",
            n_tasks=len(bench.tasks),
        )

    next_version = (int(existing["version"]) + 1) if existing else 1
    metadata = {**bench.metadata, "content_hash": content_hash}
    record = store.create_benchmark(
        name=bench.name,
        format=BENCHMARK_FORMAT,
        version=next_version,
        source_kind="harbor_jsonl",
        metadata=metadata,
    )
    benchmark_id = str(record["id"])

    rows = [to_dict(task) for task in bench.tasks]
    if rows:
        store.add_benchmark_rows(benchmark_id, rows)

    return UploadResult(
        benchmark_id=benchmark_id,
        name=bench.name,
        version=next_version,
        content_hash=content_hash,
        status="updated" if existing else "created",
        n_tasks=len(bench.tasks),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_existing(store: Any, name: str) -> dict | None:
    """Return the highest-version benchmark with ``name``, if any."""
    try:
        all_benchmarks = store.list_benchmarks() or []
    except Exception:
        return None
    matching = [b for b in all_benchmarks if b.get("name") == name]
    if not matching:
        return None
    return max(matching, key=lambda b: int(b.get("version") or 1))


__all__ = ["BENCHMARK_FORMAT", "UploadResult", "upload_benchmark"]

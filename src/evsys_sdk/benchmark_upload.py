"""Register a local harbor-format benchmark with the dashboard.

Each ``data/benchmark/<name>/`` directory becomes a row in the dashboard's
benchmark table so experiments can reference it by id. Re-uploading the
same content is a no-op (we hash ``tasks.jsonl``); re-uploading changed
content registers a new version.

CLI entry point: ``evsys benchmark upload data/benchmark/<name>``.
Programmatic: ``upload_benchmark(store, path)``.

The harbor upload body is shared with ``validation_upload`` via
``_harbor_upload.upload_harbor`` — this module only wires the benchmark store
methods and keeps the historical ``UploadResult`` shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._harbor_upload import HARBOR_FORMAT, upload_harbor

BENCHMARK_FORMAT = HARBOR_FORMAT


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

    See ``_harbor_upload.upload_harbor`` for the shared idempotency logic.
    """
    result = upload_harbor(
        path,
        create_record=store.create_benchmark,
        add_rows=store.add_benchmark_rows,
        list_existing=store.list_benchmarks,
        format=BENCHMARK_FORMAT,
    )
    return UploadResult(
        benchmark_id=result.id,
        name=result.name,
        version=result.version,
        content_hash=result.content_hash,
        status=result.status,
        n_tasks=result.n_tasks,
    )


__all__ = ["BENCHMARK_FORMAT", "UploadResult", "upload_benchmark"]

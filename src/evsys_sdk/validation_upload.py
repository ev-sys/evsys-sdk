"""Register a local harbor-format validation set with the dashboard.

A validation set is harbor-format (``tasks.jsonl`` + ``metadata.yaml``), just
like a benchmark — but it's scored *during* training (every N steps) to drive
model selection, never as the final/test measurement. It lives as its own
entity so a validation id and a benchmark (test) id can never be confused.

CLI entry point: ``evsys validation upload data/validation/<name>``.
Programmatic: ``upload_validation_dataset(store, path)``.

The harbor upload body is shared with ``benchmark_upload`` via
``_harbor_upload.upload_harbor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._harbor_upload import HARBOR_FORMAT, upload_harbor

VALIDATION_FORMAT = HARBOR_FORMAT


@dataclass(frozen=True)
class ValidationUploadResult:
    """Result of one validation-set upload call."""

    validation_dataset_id: str
    name: str
    version: int
    content_hash: str
    status: str
    """``"created" | "updated" | "unchanged"``."""
    n_tasks: int


def upload_validation_dataset(store: Any, path: str | Path) -> ValidationUploadResult:
    """Upload (or re-upload) a harbor validation-set directory.

    See ``_harbor_upload.upload_harbor`` for the shared idempotency logic.
    """
    result = upload_harbor(
        path,
        create_record=store.create_validation_dataset,
        add_rows=store.add_validation_dataset_rows,
        list_existing=store.list_validation_datasets,
        format=VALIDATION_FORMAT,
    )
    return ValidationUploadResult(
        validation_dataset_id=result.id,
        name=result.name,
        version=result.version,
        content_hash=result.content_hash,
        status=result.status,
        n_tasks=result.n_tasks,
    )


__all__ = ["VALIDATION_FORMAT", "ValidationUploadResult", "upload_validation_dataset"]

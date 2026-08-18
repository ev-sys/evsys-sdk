"""Push a local static HTML report directory to the dashboard.

Zips ``local_dir`` and POSTs it to
``POST /api/dashboard/api/sdk/reports/push/`` (multipart).
"""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .constants import (
    API_PREFIX,
    DEFAULT_API_URL,
    DEFAULT_TIMEOUT_S,
    EP_REPORT_PUSH,
    EVSYS_API_KEY_ENV,
    EVSYS_API_URL_ENV,
    EVSYS_PROJECT_ID_ENV,
    bearer,
)


@dataclass
class ReportPushResult:
    report_id: str
    path: str
    content_hash: str
    n_files: int
    size_bytes: int
    entry_file: str
    url: str = ""


def _zip_dir(local_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(local_dir):
            for fname in files:
                full = Path(root) / fname
                # Skip junk
                if fname.startswith(".") and fname != ".htaccess":
                    continue
                arc = full.relative_to(local_dir).as_posix()
                zf.write(full, arcname=arc)
    return buf.getvalue()


def push_report(
    local_dir: str | Path,
    *,
    path: str,
    project_id: str | None = None,
    name: str | None = None,
    entry_file: str = "index.html",
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ReportPushResult:
    """Zip ``local_dir`` and push it to ``path`` under the project."""
    root = Path(local_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"report directory not found: {root}")

    entry = (entry_file or "index.html").lstrip("/")
    if not (root / entry).is_file():
        raise ValueError(f"entry file {entry!r} not found under {root}")

    if not (path or "").strip().strip("/"):
        raise ValueError("path is required")

    api_key = api_key or os.environ.get(EVSYS_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"missing {EVSYS_API_KEY_ENV}")
    project_id = project_id or os.environ.get(EVSYS_PROJECT_ID_ENV)
    if not project_id:
        raise ValueError(f"missing project_id / {EVSYS_PROJECT_ID_ENV}")

    base = (base_url or os.environ.get(EVSYS_API_URL_ENV) or DEFAULT_API_URL).rstrip("/")
    url = f"{base}{API_PREFIX}{EP_REPORT_PUSH}"

    zip_bytes = _zip_dir(root)
    data: dict[str, Any] = {
        "project_id": project_id,
        "path": path.strip(),
        "entry_file": entry,
    }
    if name:
        data["name"] = name

    r = requests.post(
        url,
        headers={"Authorization": bearer(api_key)},
        data=data,
        files={"file": ("report.zip", zip_bytes, "application/zip")},
        timeout=max(timeout_s, 120.0),
    )
    if r.status_code >= 400:
        raise RuntimeError(f"report push → HTTP {r.status_code}: {r.text[:400]}")
    body = r.json() or {}
    return ReportPushResult(
        report_id=str(body["report_id"]),
        path=str(body["path"]),
        content_hash=str(body.get("content_hash") or ""),
        n_files=int(body.get("n_files") or 0),
        size_bytes=int(body.get("size_bytes") or 0),
        entry_file=str(body.get("entry_file") or entry),
        url=str(body.get("url") or ""),
    )

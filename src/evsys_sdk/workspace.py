"""Workspace — local cache for remote datasets/benchmarks.

Remote-first: datasets live in the backend (D20, accessed via ``EvsysStore``
over the gateway). Streaming every row over HTTP during training is slow, so the
agent materializes a dataset to a local JSONL **once** and trains from the local
file. On ``pull_dataset`` the local copy is reused if present and complete;
otherwise it's fetched from remote, written, and cached.

Safe to cache: datasets are versioned and immutable per version, so a given
``dataset_id`` never changes — the only risk is a partial pull, guarded by a
``.meta.json`` manifest (atomic rename + ``complete`` flag + n_rows match).

The workspace root (``$EVSYS_WORKSPACE`` or ``./.evsys``) writes a
self-ignoring ``.gitignore`` (``*``) on init, so nothing in it is ever tracked.
Rows are written **raw** (D17); ``MaterializedDataset`` carries the dataset's
``format`` + ``transform`` so the trainer can render typed rows on read.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .store import EvsysStore

_WORKSPACE_ENV = "EVSYS_WORKSPACE"
_DEFAULT_ROOT = "./.evsys"
_PAGE = 500


@dataclass
class MaterializedDataset:
    path: str               # local JSONL of raw payloads (one per line, idx order)
    format: str | None      # target format (chat_messages / harbor_task / …)
    transform: Any          # transform spec(s) to render raw → typed
    n_rows: int
    cached: bool            # True if served from the local cache (no remote pull)


class Workspace:
    def __init__(self, store: EvsysStore | None = None, *, root: str | None = None) -> None:
        self.store = store or EvsysStore()
        self.root = Path(root or os.environ.get(_WORKSPACE_ENV) or _DEFAULT_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)
        gi = self.root / ".gitignore"
        if not gi.exists():
            gi.write_text("*\n")  # self-ignoring: workspace is never tracked
        for sub in ("datasets", "benchmarks", "validation", "scripts", "outputs"):
            (self.root / sub).mkdir(exist_ok=True)

    # -- materialization ------------------------------------------------------

    def pull_dataset(self, dataset_id: str, *, force: bool = False) -> MaterializedDataset:
        return self._materialize(
            "datasets", dataset_id, force,
            get_meta=self.store.get_dataset,
            get_rows=self.store.get_dataset_rows,
        )

    def pull_benchmark(self, benchmark_id: str, *, force: bool = False) -> MaterializedDataset:
        return self._materialize(
            "benchmarks", benchmark_id, force,
            get_meta=self.store.get_benchmark,
            get_rows=self.store.get_benchmark_rows,
        )

    def pull_validation_dataset(self, validation_dataset_id: str, *, force: bool = False) -> MaterializedDataset:
        return self._materialize(
            "validation", validation_dataset_id, force,
            get_meta=self.store.get_validation_dataset,
            get_rows=self.store.get_validation_dataset_rows,
        )

    # -- name → id resolution (latest version wins) ---------------------------

    def dataset_id_for_name(self, name: str) -> str:
        return self._id_for_name(self.store.list_datasets, name, "dataset")

    def benchmark_id_for_name(self, name: str) -> str:
        return self._id_for_name(self.store.list_benchmarks, name, "benchmark")

    def validation_dataset_id_for_name(self, name: str) -> str:
        return self._id_for_name(
            self.store.list_validation_datasets, name, "validation dataset"
        )

    def _id_for_name(self, list_fn: Callable[[], list[dict] | None], name: str, kind: str) -> str:
        """Resolve a name to the highest-version record's id for this project."""
        records = list_fn() or []
        matching = [r for r in records if r.get("name") == name]
        if not matching:
            raise FileNotFoundError(f"no {kind} named {name!r} in this project")
        return str(max(matching, key=lambda r: int(r.get("version") or 1))["id"])

    def _materialize(self, sub: str, obj_id: str, force: bool, *,
                     get_meta: Callable[[str], dict | None],
                     get_rows: Callable[..., list[dict]]) -> MaterializedDataset:
        meta = get_meta(obj_id)
        if meta is None:
            raise FileNotFoundError(f"{sub[:-1]} {obj_id} not found")
        n_rows = int(meta.get("n_rows") or 0)
        fmt, transform = meta.get("format"), meta.get("transform")

        path = self.root / sub / f"{obj_id}.jsonl"
        man_path = self.root / sub / f"{obj_id}.meta.json"

        if force:
            for p in (path, man_path):
                p.unlink(missing_ok=True)
        elif path.exists() and man_path.exists():
            try:
                man = json.loads(man_path.read_text())
                if man.get("complete") and int(man.get("n_rows", -1)) == n_rows:
                    return MaterializedDataset(str(path), fmt, transform, n_rows, cached=True)
            except Exception:
                pass  # corrupt manifest → re-pull

        # Pull pages → temp file → atomic rename (no half-written cache is trusted).
        tmp = self.root / sub / f".{obj_id}.tmp.jsonl"
        written = 0
        with tmp.open("w") as f:
            offset = 0
            while True:
                page = get_rows(obj_id, limit=_PAGE, offset=offset)
                if not page:
                    break
                for row in page:
                    f.write(json.dumps(row.get("payload", row), default=str) + "\n")
                    written += 1
                if len(page) < _PAGE:
                    break
                offset += _PAGE
        tmp.replace(path)
        man_path.write_text(json.dumps({
            "id": obj_id, "version": meta.get("version"), "n_rows": written,
            "format": fmt, "transform": transform, "complete": True,
        }))
        return MaterializedDataset(str(path), fmt, transform, written, cached=False)

    # -- standard locations for the generated training script -----------------

    def script_path(self, exp_id: str) -> str:
        return str(self.root / "scripts" / f"exp_{exp_id}.py")

    def outputs_dir(self, run_id: str) -> str:
        d = self.root / "outputs" / str(run_id)
        d.mkdir(parents=True, exist_ok=True)
        return str(d)


def read_jsonl_rows(path: str) -> list[dict[str, Any]]:
    """Read a materialized ``.evsys/`` JSONL (one payload per line) to dicts."""
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


__all__ = ["Workspace", "MaterializedDataset", "read_jsonl_rows"]

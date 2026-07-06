"""LocalDataStore — filesystem JSONL/JSON, no network."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import register_data_store


class LocalDataStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root: str = "."
    """Path resolution root. Relative paths are resolved against this."""


@register_data_store("local")
class LocalDataStore:
    name: ClassVar[str] = "local"
    Config: ClassVar[type] = LocalDataStoreConfig

    def __init__(self, *, root: str | os.PathLike[str] = ".") -> None:
        self.root = Path(root).expanduser().resolve()

    def _resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        if p.is_absolute():
            return p
        return self.root / p

    def read_jsonl(self, path: str) -> list[dict[str, Any]]:
        with self._resolve(path).open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def write_jsonl(self, path: str, rows: Iterable[dict[str, Any]]) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def read_json(self, path: str) -> Any:
        with self._resolve(path).open() as f:
            return json.load(f)

    def write_json(self, path: str, value: Any) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as f:
            json.dump(value, f, indent=2, default=str)

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def list(self, prefix: str) -> list[str]:
        base = self._resolve(prefix)
        if base.is_dir():
            return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file())
        return [str(p.relative_to(self.root)) for p in self.root.glob(prefix) if p.is_file()]

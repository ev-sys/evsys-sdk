"""InMemoryDataStore — for tests."""

from __future__ import annotations

from typing import Any, ClassVar, Iterable

from pydantic import BaseModel, ConfigDict

from ..registry import register_data_store


class InMemoryDataStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


@register_data_store("in_memory")
class InMemoryDataStore:
    name: ClassVar[str] = "in_memory"
    Config: ClassVar[type] = InMemoryDataStoreConfig

    def __init__(self) -> None:
        self._jsonl: dict[str, list[dict[str, Any]]] = {}
        self._json: dict[str, Any] = {}

    def read_jsonl(self, path: str) -> list[dict[str, Any]]:
        if path not in self._jsonl:
            raise FileNotFoundError(path)
        return list(self._jsonl[path])

    def write_jsonl(self, path: str, rows: Iterable[dict[str, Any]]) -> None:
        self._jsonl[path] = list(rows)

    def read_json(self, path: str) -> Any:
        if path not in self._json:
            raise FileNotFoundError(path)
        return self._json[path]

    def write_json(self, path: str, value: Any) -> None:
        self._json[path] = value

    def exists(self, path: str) -> bool:
        return path in self._jsonl or path in self._json

    def list(self, prefix: str) -> list[str]:
        return sorted(p for p in (*self._jsonl, *self._json) if p.startswith(prefix))

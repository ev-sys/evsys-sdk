"""JSONLLogStore — append-only metrics.jsonl + hyperparams.json + artifacts.json."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import register_log_store


class JSONLLogStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    log_dir: str
    metrics_file: str = "metrics.jsonl"
    hyperparams_file: str = "hyperparams.json"
    artifacts_file: str = "artifacts.json"


@register_log_store("jsonl")
class JSONLLogStore:
    name: ClassVar[str] = "jsonl"
    Config: ClassVar[type] = JSONLLogStoreConfig

    def __init__(
        self,
        *,
        log_dir: str,
        metrics_file: str = "metrics.jsonl",
        hyperparams_file: str = "hyperparams.json",
        artifacts_file: str = "artifacts.json",
    ) -> None:
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.log_dir / metrics_file
        self._hyperparams_path = self.log_dir / hyperparams_file
        self._artifacts_path = self.log_dir / artifacts_file
        self._closed = False
        self._artifacts: list[dict[str, Any]] = []
        if self._artifacts_path.exists():
            try:
                self._artifacts = json.loads(self._artifacts_path.read_text())
            except json.JSONDecodeError:
                self._artifacts = []

    def _append(self, payload: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("JSONLLogStore is closed")
        with self._metrics_path.open("a") as f:
            f.write(json.dumps(payload) + "\n")

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self._append({"ts": time.time(), "step": step, "metrics": {key: value}})

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._append({"ts": time.time(), "step": step, "metrics": dict(metrics)})

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        existing: dict[str, Any] = {}
        if self._hyperparams_path.exists():
            try:
                existing = json.loads(self._hyperparams_path.read_text())
            except json.JSONDecodeError:
                existing = {}
        existing.update(params)
        self._hyperparams_path.write_text(json.dumps(existing, indent=2, default=str))

    def log_artifact(self, name: str, path: str, *, kind: str = "file") -> None:
        self._artifacts.append({"name": name, "path": path, "kind": kind, "ts": time.time()})
        self._artifacts_path.write_text(json.dumps(self._artifacts, indent=2))

    def close(self) -> None:
        self._closed = True

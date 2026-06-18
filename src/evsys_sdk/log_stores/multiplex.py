"""MultiplexLogStore — fan out to several log stores at once.

Useful for ``log_store.kind: multiplex`` configs:

    log_store:
      kind: multiplex
      params:
        children:
          - kind: jsonl
            params: { log_dir: ./logs }
          - kind: tensorboard
            params: { log_dir: ./tb }
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..registry import get_log_store, register_log_store


class MultiplexLogStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    children: list[dict[str, Any]] = Field(default_factory=list)
    """Each child: {kind: <log-store-name>, params: {...}}."""


@register_log_store("multiplex")
class MultiplexLogStore:
    name: ClassVar[str] = "multiplex"
    Config: ClassVar[type] = MultiplexLogStoreConfig

    def __init__(self, *, children: list[dict[str, Any]] | None = None) -> None:
        self._children = []
        for spec in children or []:
            cls = get_log_store(spec["kind"])
            self._children.append(cls(**spec.get("params", {})))

    def log_scalar(self, key: str, value: float, step: int) -> None:
        for c in self._children:
            c.log_scalar(key, value, step)

    def log_metrics(self, metrics: dict[str, float], step: int, *, split: str = "train") -> None:
        for c in self._children:
            c.log_metrics(metrics, step, split=split)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        for c in self._children:
            c.log_hyperparams(params)

    def log_artifact(self, name: str, path: str, *, kind: str = "file") -> None:
        for c in self._children:
            c.log_artifact(name, path, kind=kind)

    def close(self) -> None:
        for c in self._children:
            try:
                c.close()
            except Exception:
                pass

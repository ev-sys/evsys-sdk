"""TensorBoardLogStore — optional, depends on tensorboard."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import register_log_store

# raise ImportError at module load if tensorboard is missing — caller handles it
from torch.utils.tensorboard import SummaryWriter  # noqa: E402


class TensorBoardLogStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    log_dir: str
    flush_secs: int = 30


@register_log_store("tensorboard")
class TensorBoardLogStore:
    name: ClassVar[str] = "tensorboard"
    Config: ClassVar[type] = TensorBoardLogStoreConfig

    def __init__(self, *, log_dir: str, flush_secs: int = 30) -> None:
        self._writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self._writer.add_scalar(key, value, global_step=step)

    def log_metrics(
        self, metrics: dict[str, float], step: int, *, split: str = "train"
    ) -> None:
        # `split` is already encoded in the metric keys (e.g. "val/..."); kept
        # for interface parity with the other log stores.
        for k, v in metrics.items():
            self._writer.add_scalar(k, v, global_step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        # add_hparams is finicky; persist as text instead
        self._writer.add_text("hparams", str(params))

    def log_artifact(self, name: str, path: str, *, kind: str = "file") -> None:
        self._writer.add_text(f"artifact/{name}", f"[{kind}] {path}")

    def close(self) -> None:
        self._writer.close()

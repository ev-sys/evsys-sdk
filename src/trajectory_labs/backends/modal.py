"""ModalBackend — dispatch training to a deployed Modal app (miles / unsloth).

Unlike :class:`TinkerBackend` (which drives an in-process API client), the
Modal backend launches a **remote GPU job**: the actual training loop runs
inside a Modal app deployed from ``modal_app.py`` (the ``trajectory-training``
app) on the ``radixark/miles`` image. This backend only carries the launch
config — the ``modal_miles`` algorithm does the dispatch (stage data → spawn
remote fn → poll → ingest checkpoint).

``modal`` is imported lazily (inside the algorithm), so importing this backend
needs neither the ``modal`` package nor Modal credentials.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..registry import register_backend


class ModalBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    app_name: str = "trajectory-training"
    """Name of the deployed Modal app to dispatch to."""
    gpu: str = "H200"
    timeout_hours: float = 12.0
    retries: int = 3
    image_tag: str = "radixark/miles:dev"
    data_volume: str = "trajectory-data"
    checkpoints_volume: str = "trajectory-checkpoints"
    engine: Literal["auto", "miles", "unsloth"] = "auto"
    """'auto' picks unsloth for small SFT/SDFT, miles for RL / large models."""
    secrets: list[str] = Field(default_factory=lambda: ["wandb-secret", "hf-token"])


@register_backend("modal")
class ModalBackend:
    name: ClassVar[str] = "modal"
    Config: ClassVar[type] = ModalBackendConfig

    def __init__(
        self,
        *,
        app_name: str = "trajectory-training",
        gpu: str = "H200",
        timeout_hours: float = 12.0,
        retries: int = 3,
        image_tag: str = "radixark/miles:dev",
        data_volume: str = "trajectory-data",
        checkpoints_volume: str = "trajectory-checkpoints",
        engine: str = "auto",
        secrets: list[str] | None = None,
    ) -> None:
        self.cfg: dict[str, Any] = {
            "app_name": app_name,
            "gpu": gpu,
            "timeout_hours": timeout_hours,
            "retries": retries,
            "image_tag": image_tag,
            "data_volume": data_volume,
            "checkpoints_volume": checkpoints_volume,
            "engine": engine,
            "secrets": list(secrets) if secrets is not None else ["wandb-secret", "hf-token"],
        }

    def prepare(self, *, model: dict[str, Any], run_dir: str) -> dict[str, Any]:
        # No persistent client to build: Modal functions are looked up by name
        # at dispatch time. We just thread the launch config + model info.
        return {
            "backend": "modal",
            "model_name": model["name"],
            "load_checkpoint_path": model.get("load_checkpoint_path"),
            "run_dir": run_dir,
            **self.cfg,
        }

    def teardown(self, handles: dict[str, Any]) -> None:
        return None

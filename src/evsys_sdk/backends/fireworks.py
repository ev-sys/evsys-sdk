"""FireworksBackend (registry) — Fireworks via the tinker-compatible Firetitan
service client.

Mirrors the ``tinker`` registry backend: ``prepare()`` validates the API key and
returns the model name + a service client handle. The actual training-client
allocation is left to the algorithm, which dispatches to
:class:`evsys_sdk.training.fireworks_backend.FireworksBackend` (the training
allocator) for ``backend.kind == "fireworks"``.

``fireworks-ai`` is an OPTIONAL extra (``pip install 'evsys-sdk[fireworks]'``);
it is imported lazily so this module — and the registry — load without it.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import register_backend


class FireworksBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key_env: str = "FIREWORKS_API_KEY"
    """Env var holding the Fireworks API key. Read at prepare-time."""
    base_url: str | None = None
    """Override the Firetitan training service URL (rare)."""
    inference_url: str | None = None
    """Override the Firetitan inference/sampling URL (rare)."""


@register_backend("fireworks")
class FireworksBackend:
    name: ClassVar[str] = "fireworks"
    Config: ClassVar[type] = FireworksBackendConfig

    def __init__(
        self,
        *,
        api_key_env: str = "FIREWORKS_API_KEY",
        base_url: str | None = None,
        inference_url: str | None = None,
    ) -> None:
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.inference_url = inference_url

    def prepare(self, *, model: dict[str, Any], run_dir: str) -> dict[str, Any]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} not set in env — needed for FireworksBackend.prepare()"
            )
        try:
            from fireworks.training.sdk.tinker_compat import FiretitanServiceClient
        except ImportError as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "FireworksBackend needs the fireworks-ai SDK. Install the extra: "
                "pip install 'evsys-sdk[fireworks]'."
            ) from e
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.inference_url:
            kwargs["inference_url"] = self.inference_url
        service_client = FiretitanServiceClient.from_firetitan_config(**kwargs)
        return {
            "backend": "fireworks",
            "service_client": service_client,
            "model_name": model["name"],
            "load_checkpoint_path": model.get("load_checkpoint_path"),
            "init_from_checkpoint": model.get("init_from_checkpoint"),
            "renderer_name": model.get("renderer_name"),
            "api_key_env": self.api_key_env,
            "run_dir": run_dir,
        }

    def teardown(self, handles: dict[str, Any]) -> None:
        return None

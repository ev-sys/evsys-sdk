"""TinkerBackend — thin wrapper around the tinker SDK.

prepare() lazily creates a ServiceClient and returns it as a handle along with
the chosen model name + a tokenizer. The actual training-client creation is
left to the algorithm so it can pick LoraConfig / lr / etc.

Most of the heavy lifting (SFT loop, RL loop, checkpointing, eval cadence) is
in tinker_cookbook — algorithms that target this backend should call into the
cookbook recipes rather than reinventing the loop.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import register_backend

# raise ImportError at module load if tinker isn't installed
import tinker  # noqa: E402


class TinkerBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key_env: str = "TINKER_API_KEY"
    """Env var holding the API key. Read at prepare-time."""
    base_url: str | None = None
    """Override the Tinker service URL (rare)."""


@register_backend("tinker")
class TinkerBackend:
    name: ClassVar[str] = "tinker"
    Config: ClassVar[type] = TinkerBackendConfig

    def __init__(self, *, api_key_env: str = "TINKER_API_KEY", base_url: str | None = None) -> None:
        self.api_key_env = api_key_env
        self.base_url = base_url

    def prepare(self, *, model: dict[str, Any], run_dir: str) -> dict[str, Any]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} not set in env — needed for TinkerBackend.prepare()"
            )
        # tinker.ServiceClient picks up the API key from env automatically;
        # we set it explicitly so the active env wins over any earlier export.
        os.environ[self.api_key_env] = api_key
        # make_service_client resolves base_url (arg → TINKER_BASE_URL env) and
        # re-exports it so harbor's rollout client targets the same backend.
        from ..tinker_service import make_service_client
        service_client = make_service_client(self.base_url)
        return {
            "backend": "tinker",
            "service_client": service_client,
            "model_name": model["name"],
            "load_checkpoint_path": model.get("load_checkpoint_path"),
            "init_from_checkpoint": model.get("init_from_checkpoint"),
            "renderer_name": model.get("renderer_name"),
            "run_dir": run_dir,
        }

    def teardown(self, handles: dict[str, Any]) -> None:
        return None

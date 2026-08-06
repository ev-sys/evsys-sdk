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

# raise ImportError at module load if tinker isn't installed
import tinker
from pydantic import BaseModel, ConfigDict

from ..registry import register_backend


class TinkerBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key_env: str = "TINKER_API_KEY"
    """Env var holding the API key. Read at prepare-time."""
    base_url: str | None = None
    """Override the Tinker service URL (rare)."""


PROTOCOL_TINKER = "tinker"
"""The training-service protocol these algorithms speak.

What the RL/SFT loops actually require is the *protocol* — forward_backward,
optim_step, save_weights_for_sampler — not this particular vendor. Backends
declare it as a ClassVar so a server that implements the same surface (SkyRL,
on your own GPUs) is accepted on its capability rather than on its name."""

CONNECT_TIMEOUT_S = 90.0
"""How long to wait for the service client before calling it a failure.

``ServiceClient()`` authenticates in its constructor and retries internally, so
a transport that cannot connect does not raise — it blocks forever. A run then
looks hung with an empty log, which is exactly how a broken transport cost us
an afternoon: the real error was ``invalid peer certificate: UnknownIssuer``
from ``pyqwest`` 0.7.0, invisible behind the retry loop."""


def _connect(kwargs: dict[str, Any]) -> Any:
    """Build the service client, turning a silent stall into a real error."""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(tinker.ServiceClient, **kwargs).result(CONNECT_TIMEOUT_S)
    except FuturesTimeout as e:
        raise RuntimeError(
            f"tinker.ServiceClient() did not connect within {CONNECT_TIMEOUT_S:.0f}s. "
            "It retries auth internally, so a transport failure hangs instead of "
            "raising. The usual cause is the HTTP transport: pyqwest 0.7.0 fails "
            "TLS verification ('invalid peer certificate: UnknownIssuer'). "
            f"Installed: pyqwest {_version('pyqwest')} — pin pyqwest==0.6.1. "
            "Check reachability and TINKER_API_KEY too."
        ) from e
    finally:
        # the thread is a daemon inside a stuck client; do not block teardown
        pool.shutdown(wait=False)


def _version(pkg: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not installed"


@register_backend("tinker")
class TinkerBackend:
    name: ClassVar[str] = "tinker"
    protocol: ClassVar[str] = PROTOCOL_TINKER
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
        kwargs: dict[str, Any] = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        # tinker.ServiceClient picks up the API key from env automatically;
        # we set it explicitly so the active env wins over any earlier export.
        os.environ[self.api_key_env] = api_key
        service_client = _connect(kwargs)
        return {
            "backend": "tinker",
            "service_client": service_client,
            "model_name": model["name"],
            "load_checkpoint_path": model.get("load_checkpoint_path"),
            "resume_step": model.get("resume_step"),
            "init_from_checkpoint": model.get("init_from_checkpoint"),
            "renderer_name": model.get("renderer_name"),
            "run_dir": run_dir,
        }

    def teardown(self, handles: dict[str, Any]) -> None:
        return None

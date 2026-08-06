"""SkyRLBackend — run the same training on your own compute.

`SkyRL <https://github.com/NovaSky-AI/SkyRL>`_ serves the **Tinker protocol**:
the same ``/api/v1/forward_backward``, ``/optim_step``, ``/save_weights_for_sampler``
… surface the hosted service exposes, backed by FSDP / Megatron / JAX on
hardware you control. It is a server for the *official* ``tinker`` client, not
a lookalike, so nothing in this SDK needs a second client implementation.

Which is why this backend is thin. The whole integration is one environment
variable::

    # tinker/_client.py
    if base_url is None:
        base_url = os.environ.get("TINKER_BASE_URL")

Every client the run creates — the training client, each sampling client, and
harbor's ``TinkerLLM`` inside the rollout engine — constructs ``ServiceClient()``
with no arguments and therefore reads that variable. Setting it once in
:meth:`prepare` redirects the entire run. Point it at a SkyRL server and the
identical ``config.yaml`` trains on your GPUs::

    backend: {kind: skyrl, params: {base_url: "http://localhost:8000"}}

**Authentication**: SkyRL performs none. It accepts any non-empty key because
the ``tinker`` client refuses to construct without one, so ``api_key_default``
exists purely to spare users inventing a value. The corollary matters: a SkyRL
server must not be exposed publicly — bind it to localhost or keep it inside
your VPC.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..logger import get_logger
from ..registry import register_backend
from .tinker import PROTOCOL_TINKER, _connect

log = get_logger(__name__)

HEALTH_PATH = "/api/v1/get_server_capabilities"
"""Probed before handing back a client — a refused connection here is a clear
"no server there", where the Tinker client would instead retry and stall."""


class SkyRLBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://localhost:8000"
    """Where the SkyRL Tinker server is listening."""
    api_key_env: str = "TINKER_API_KEY"
    """Env var read for the key. SkyRL ignores its value."""
    api_key_default: str = "tml-dummy"
    """Used when ``api_key_env`` is unset — SkyRL authenticates nobody, and the
    tinker client will not construct without *some* key."""
    health_check: bool = True
    """Probe the server before returning. Off for a server still warming up."""
    health_timeout_s: float = Field(default=30.0, gt=0)
    """A cold server loads weights and warms an engine; allow for it."""


@register_backend("skyrl")
class SkyRLBackend:
    """The Tinker protocol, served from your own hardware."""

    name: ClassVar[str] = "skyrl"
    protocol: ClassVar[str] = PROTOCOL_TINKER
    Config: ClassVar[type] = SkyRLBackendConfig

    def __init__(self, *, base_url: str = "http://localhost:8000",
                 api_key_env: str = "TINKER_API_KEY",
                 api_key_default: str = "tml-dummy",
                 health_check: bool = True,
                 health_timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key_default = api_key_default
        self.health_check = health_check
        self.health_timeout_s = health_timeout_s

    def _probe(self) -> None:
        url = f"{self.base_url}{HEALTH_PATH}"
        try:
            urllib.request.urlopen(url, timeout=self.health_timeout_s)
        except urllib.error.HTTPError:
            return  # it answered; a non-200 status still proves a server is there
        except Exception as e:
            raise RuntimeError(
                f"no SkyRL server reachable at {self.base_url} ({type(e).__name__}: {e}). "
                "Start one with `python -m skyrl.tinker.api --base-model <model> "
                "--backend jax --port 8000`, or set health_check: false if it is "
                "still warming up."
            ) from e

    def prepare(self, *, model: dict[str, Any], run_dir: str) -> dict[str, Any]:
        # The client reads TINKER_BASE_URL whenever it is constructed without
        # arguments, which is how every client in this SDK (and in harbor) is
        # built. Setting it here redirects the whole run in one place.
        os.environ["TINKER_BASE_URL"] = self.base_url
        os.environ.setdefault(self.api_key_env, self.api_key_default)
        if self.health_check:
            self._probe()
        log.info("[skyrl] training against %s", self.base_url)
        service_client = _connect({"base_url": self.base_url})
        return {
            "backend": self.name,
            "protocol": PROTOCOL_TINKER,
            "service_client": service_client,
            "base_url": self.base_url,
            "model_name": model["name"],
            "load_checkpoint_path": model.get("load_checkpoint_path"),
            "resume_step": model.get("resume_step"),
            "init_from_checkpoint": model.get("init_from_checkpoint"),
            "renderer_name": model.get("renderer_name"),
            "run_dir": run_dir,
        }

    def teardown(self, handles: dict[str, Any]) -> None:
        return None


__all__ = ["SkyRLBackend", "SkyRLBackendConfig"]

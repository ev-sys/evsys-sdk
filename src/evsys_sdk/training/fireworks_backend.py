"""FireworksBackend — Fireworks training over the tinker-compatible
``FiretitanServiceClient``.

Fireworks' SDK (``fireworks-ai >= 1.2.0a``) ships an API-compatible service
client at ``fireworks.training.sdk.tinker_compat.FiretitanServiceClient`` whose
training/sampling methods match tinker's one-for-one
(``create_lora_training_client_async``, ``create_training_client_from_state*``,
``create_sampling_client``, …). So the ONLY thing that differs from
:class:`~evsys_sdk.training.tinker_backend.TinkerBackend` is how the service
client is constructed — this subclass overrides that single hook and inherits
everything else (LoRA client creation, forward/backward, optim step,
save-for-sampler, sampling).

``fireworks-ai`` is an OPTIONAL extra (``pip install 'evsys-sdk[fireworks]'``);
the import is lazy so this module loads without it.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

from .tinker_backend import TinkerBackend


class FireworksBackend(TinkerBackend):
    """Training backend over Fireworks' tinker-compatible Firetitan client.

    Identical to :class:`TinkerBackend` except the service client is a
    ``FiretitanServiceClient`` (constructed from ``FIREWORKS_API_KEY``).
    """

    DEFAULT_API_KEY_ENV: ClassVar[str] = "FIREWORKS_API_KEY"

    @classmethod
    def _make_service_client(cls, *, api_key_env: str) -> Any:
        try:
            from fireworks.training.sdk.tinker_compat import FiretitanServiceClient
        except ImportError as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "FireworksBackend needs the fireworks-ai SDK. Install the extra: "
                "pip install 'evsys-sdk[fireworks]' (it provides the "
                "tinker-compatible FiretitanServiceClient)."
            ) from e
        return FiretitanServiceClient.from_firetitan_config(
            api_key=os.environ[api_key_env],
        )


__all__ = ["FireworksBackend"]

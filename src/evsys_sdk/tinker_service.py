"""Single entry point for creating a tinker ``ServiceClient``.

Centralizes *how the SDK points at a tinker backend* so every tinker client in
the process resolves the same service URL:

  * the native training backend (``training/tinker_backend.py``),
  * the inference client (``inference/tinker.py``),
  * the backend spec (``backends/tinker.py``), and
  * harbor's in-rollout ``TinkerLLM`` — which builds its *own* ``ServiceClient``
    inside the rollout, out of our reach, and only ever reads the env var.

Set ``TINKER_BASE_URL`` once and training + inference + rollouts all target that
backend out of the box — compatible with any backend that speaks the tinker SDK.

Resolution precedence mirrors the tinker SDK itself: an explicit ``base_url``
arg wins, else ``TINKER_BASE_URL`` from the env, else tinker's built-in default.
When a URL is resolved it is **re-exported** to ``TINKER_BASE_URL`` so harbor's
rollout client (and any subprocess) inherits the exact same target.
"""

from __future__ import annotations

import os
from typing import Any

TINKER_BASE_URL_ENV = "TINKER_BASE_URL"


def resolve_base_url(base_url: str | None = None) -> str | None:
    """Effective tinker service URL: explicit ``base_url`` arg, else
    ``TINKER_BASE_URL`` from the env, else ``None`` (tinker's own default)."""
    return base_url or os.environ.get(TINKER_BASE_URL_ENV) or None


def make_service_client(base_url: str | None = None, **kwargs: Any) -> Any:
    """Create a ``tinker.ServiceClient`` pointed at the resolved base URL.

    Re-exports the resolved URL to ``TINKER_BASE_URL`` so every other tinker
    client in the process — notably harbor's rollout ``TinkerLLM`` — targets the
    same backend. With neither an arg nor the env set, constructs a bare client
    (tinker falls back to its built-in default URL).
    """
    import tinker

    url = resolve_base_url(base_url)
    if url:
        os.environ[TINKER_BASE_URL_ENV] = url
        kwargs.setdefault("base_url", url)
    return tinker.ServiceClient(**kwargs)


__all__ = ["TINKER_BASE_URL_ENV", "resolve_base_url", "make_service_client"]

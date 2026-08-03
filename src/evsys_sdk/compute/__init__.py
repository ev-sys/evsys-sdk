"""Compute targets — where a training service runs.

Importing this package registers the built-ins, so ``@register_compute`` has
fired by the time YAML is resolved. Vendor SDKs stay lazy: ``sky`` is imported
only when a SkyPilot target is actually brought up.
"""

from __future__ import annotations

from typing import Any

from ..registry import get_compute
from . import credentials, liveness, pricing, snapshot
from .base import BaseCompute, ComputeError

# Side-effect import: register the built-in provider.
from . import skypilot as _skypilot  # noqa: F401,E402


def build_compute(spec: Any) -> BaseCompute:
    """Resolve ``{kind, params}`` (or a bare kind string) into a compute target.

    Mirrors ``build_sandbox``: look the class up by name, validate ``params``
    against its ``Config`` so a typo is loud, then construct.
    """
    kind = getattr(spec, "kind", None) or (spec.get("kind") if isinstance(spec, dict) else spec)
    params = getattr(spec, "params", None)
    if params is None:
        params = spec.get("params", {}) if isinstance(spec, dict) else {}
    cls = get_compute(str(kind))
    return cls(**dict(params or {}))


__all__ = ["BaseCompute", "ComputeError", "build_compute", "credentials",
           "pricing", "snapshot"]

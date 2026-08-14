"""Factory: ``{kind, params}`` → an :class:`EvsysFunction`.

Mirrors :func:`evsys_sdk.sandboxes.build_sandbox` — look the class up in the
function registry by ``kind``, validate ``params`` against its ``Config``,
then construct. A YAML typo fails loudly here rather than mid-run.
"""

from __future__ import annotations

from typing import Any

from ..registry import get_function, list_functions
from .base import EvsysFunction


def build_function(spec: Any, **overrides: Any) -> EvsysFunction:
    """Resolve a function spec into a function instance.

    ``spec`` may be a :class:`~evsys_sdk.config.FunctionSpec`, any object with
    ``kind``/``params``, a plain dict, or a bare function-name string.
    """
    if isinstance(spec, str):
        kind, params = spec, {}
    elif isinstance(spec, dict):
        kind, params = spec.get("kind"), spec.get("params") or {}
    else:
        kind = getattr(spec, "kind", None)
        params = getattr(spec, "params", None) or {}
    if not kind:
        raise ValueError("function spec has no `kind`")

    cls = get_function(kind)
    return cls(**{**params, **overrides})


def available_functions() -> list[str]:
    """Registered function names (built-ins plus anything a project registered)."""
    return list_functions()


__all__ = ["available_functions", "build_function"]

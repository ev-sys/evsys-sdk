"""Factory: ``{kind, params}`` → a live :class:`BaseSandbox`.

Mirrors :func:`evsys_sdk.training.callbacks.build_callbacks` and
:func:`evsys_sdk.trace_sources.runtime.build_trace_sources` — look the class up
in the registry by ``kind``, validate ``params`` against its ``Config``, then
construct. A YAML typo fails loudly here rather than halfway through a run.
"""

from __future__ import annotations

import os
from typing import Any

from ..registry import get_sandbox, list_sandboxes
from .base import BaseSandbox


def build_sandbox(spec: Any, *, envs: dict[str, str] | None = None,
                  timeout_s: float = 1800.0, start: bool = True) -> BaseSandbox:
    """Resolve a sandbox spec into a started sandbox.

    ``spec`` may be a :class:`~evsys_sdk.config.SandboxSpec`, any object with
    ``kind``/``params``, a plain dict, or a bare provider-name string.
    """
    if isinstance(spec, str):
        kind, params = spec, {}
    elif isinstance(spec, dict):
        kind, params = spec.get("kind", "e2b"), spec.get("params") or {}
    else:
        kind = getattr(spec, "kind", None) or "e2b"
        params = getattr(spec, "params", None) or {}

    cls = get_sandbox(kind)
    sbx = cls(envs=envs, timeout_s=timeout_s, **params)
    if start:
        # ensure_started, not start: the result is commonly used as a context
        # manager too, and starting twice would orphan the first sandbox.
        sbx.ensure_started()
    return sbx


def resolve_envs(names: list[str] | None) -> dict[str, str]:
    """The host env vars named in ``env_passthrough`` that are actually set.

    Sandboxes get nothing from the host except what is listed here (a headless
    ``claude`` in a sandbox authenticates via ``ANTHROPIC_API_KEY``; there is no
    OAuth in there).
    """
    return {k: os.environ[k] for k in (names or []) if os.environ.get(k)}


def available_sandboxes() -> list[str]:
    """Registered provider names (built-ins plus anything a project registered)."""
    return list_sandboxes()


__all__ = ["available_sandboxes", "build_sandbox", "resolve_envs"]

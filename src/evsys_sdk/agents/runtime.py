"""Factory: ``{kind, params}`` → an :class:`EvsysAgent`.

Mirrors :func:`evsys_sdk.sandboxes.build_sandbox` — look the class up in the
agent registry by ``kind``, validate ``params`` against its ``Config``, then
construct. A YAML typo fails loudly here rather than mid-spawn.
"""

from __future__ import annotations

from typing import Any

from ..registry import get_agent, list_agents
from .base import EvsysAgent


def build_agent(spec: Any, **overrides: Any) -> EvsysAgent:
    """Resolve an agent spec into an agent instance.

    ``spec`` may be an :class:`~evsys_sdk.config.AgentSpec`, any object with
    ``kind``/``params``, a plain dict, or a bare agent-name string.
    ``overrides`` are merged over the spec's params (how a caller injects the
    environment or a model choice without editing the spec).
    """
    if isinstance(spec, str):
        kind, params = spec, {}
    elif isinstance(spec, dict):
        kind, params = spec.get("kind", "trigger"), spec.get("params") or {}
    else:
        kind = getattr(spec, "kind", None) or "trigger"
        params = getattr(spec, "params", None) or {}

    cls = get_agent(kind)
    return cls(**{**params, **overrides})


def available_agents() -> list[str]:
    """Registered agent names (built-ins plus anything a project registered)."""
    return list_agents()


__all__ = ["available_agents", "build_agent"]

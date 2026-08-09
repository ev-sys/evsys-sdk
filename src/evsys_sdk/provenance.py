"""Trigger provenance — which escalation caused this experiment.

Without this, an experiment on disk is an orphan: you can see that a run
happened, but not that the autoresearch agent ran it in response to
escalation-000015, and there is no way to answer "show me what the agent
actually tried". The UI cannot scope anything to an agent run.

The link travels as **environment variables**, deliberately. The agent may be
a subprocess on this host or a headless ``claude -p`` inside an E2B / Modal
sandbox; env vars are the one channel that works identically for both (the
sandbox providers already inject ``envs`` into every command), and they need
no change to any SDK call signature — an experiment launched by a script the
agent wrote five directories deep still gets stamped.

Set by :func:`trigger_env` at spawn time, read by :func:`current_trigger`
whenever an experiment record is created.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ENV_ESCALATION = "EVSYS_TRIGGER_ESCALATION"
"""Escalation id (the event's file stem, e.g. ``escalation-00000015``)."""
ENV_AGENT = "EVSYS_TRIGGER_AGENT"
"""Which agent is running: ``trigger`` or ``autoresearch``."""
ENV_AGENT_RUN = "EVSYS_TRIGGER_AGENT_RUN"
"""Id for one agent invocation, so several experiments from a single
autoresearch run group together (an agent may try three things)."""
ENV_SANDBOX = "EVSYS_TRIGGER_SANDBOX"
"""Sandbox provider the agent runs in (``e2b``/``modal``/``local``), or unset
when it runs directly on the host."""

AGENT_TRIGGER = "trigger"
AGENT_AUTORESEARCH = "autoresearch"


def trigger_env(
    escalation_path: str | Path | None,
    *,
    agent: str,
    agent_run: str | None = None,
    sandbox: str | None = None,
) -> dict[str, str]:
    """Build the env vars that stamp everything this agent launches.

    Merge into the environment of a spawned agent (and into a sandbox's
    ``env_passthrough`` set, which is why these are plain strings).
    """
    env: dict[str, str] = {ENV_AGENT: agent}
    if escalation_path:
        env[ENV_ESCALATION] = Path(str(escalation_path)).stem
    if agent_run:
        env[ENV_AGENT_RUN] = agent_run
    if sandbox:
        env[ENV_SANDBOX] = sandbox
    return env


def current_trigger(environ: dict[str, str] | None = None) -> dict[str, Any] | None:
    """The trigger context this process is running under, or ``None`` when it
    was not spawned by an agent (a human running ``evsys run`` by hand)."""
    src = environ if environ is not None else os.environ
    escalation = src.get(ENV_ESCALATION)
    agent = src.get(ENV_AGENT)
    if not escalation and not agent:
        return None
    out: dict[str, Any] = {}
    for key, name in (
        (ENV_ESCALATION, "escalation"), (ENV_AGENT, "agent"),
        (ENV_AGENT_RUN, "agent_run"), (ENV_SANDBOX, "sandbox"),
    ):
        value = src.get(key)
        if value:
            out[name] = value
    return out or None


def trigger_tags(context: dict[str, Any] | None) -> list[str]:
    """Filterable tags for the experiment record — so "everything the
    autoresearch agent did for escalation X" is one tag query, locally or on
    the dashboard, without a schema change."""
    if not context:
        return []
    tags = []
    if context.get("agent"):
        tags.append(str(context["agent"]))
    if context.get("escalation"):
        tags.append(f"escalation:{context['escalation']}")
    return tags


__all__ = [
    "AGENT_AUTORESEARCH",
    "AGENT_TRIGGER",
    "ENV_AGENT",
    "ENV_AGENT_RUN",
    "ENV_ESCALATION",
    "ENV_SANDBOX",
    "current_trigger",
    "trigger_env",
    "trigger_tags",
]

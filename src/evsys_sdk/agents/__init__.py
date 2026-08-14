"""The agent extension point — LLM agents the SDK spawns (see ``base.py``)."""

from .base import (
    AUTORESEARCH_OFF,
    AUTORESEARCH_ON,
    DEFAULT_PROMPT,
    DISTILL_PROMPT,
    REMOTE_AUTORESEARCH_PROMPT,
    AutoresearchAgent,
    EvsysAgent,
    TriggerAgent,
)
from .runtime import available_agents, build_agent

__all__ = [
    "AUTORESEARCH_OFF",
    "AUTORESEARCH_ON",
    "DEFAULT_PROMPT",
    "DISTILL_PROMPT",
    "REMOTE_AUTORESEARCH_PROMPT",
    "AutoresearchAgent",
    "EvsysAgent",
    "TriggerAgent",
    "available_agents",
    "build_agent",
]

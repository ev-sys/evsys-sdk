"""Agent harnesses as a registry extension — the same ``{kind, params}`` plugin
pattern used for algorithms, verifiers, transforms, etc.

An *agent* is the rollout harness that drives the model through a task: it decides
how the model is prompted, how many turns it takes, and (for future agents) how it
calls tools and consumes their results. The harness is used for BOTH training
rollouts (RL/SDFT) and val/test eval, so a model is trained and evaluated through
the *same* harness — and the eval harness can be overridden per benchmark.

A plugin is a lightweight descriptor (NOT the harbor agent class itself — that lives
in ``training/harbor_agents.py`` and pulls in tinker/harbor): it carries

  * ``name``       — the ``kind`` used in YAML.
  * ``agent_path`` — ``"module:Class"`` import path of the harbor ``BaseAgent`` to run.
  * ``Config``     — a Pydantic model (``extra="forbid"``) of the agent-behaviour params
                     a researcher sets under ``params:`` (model/sampling params are
                     injected by the rollout, not declared here).

Resolution to ``(import_path, params)`` happens in
:func:`evsys_sdk.training.harbor_engine.resolve_agent`. Researchers register their own
multi-turn / tool-executing harness with ``@register_agent("my_agent")`` + a custom
``agent_import_path`` — no SDK edit.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from .registry import register_agent


class _AgentParams(BaseModel):
    """Base for an agent plugin's ``Config`` — forbids unknown params so typos in
    ``params:`` fail loudly, matching the rest of the SDK's spec models."""

    model_config = ConfigDict(extra="forbid")


@register_agent("basic_loop")
class BasicLoopAgentPlugin:
    """The default harness: one ``Chat(LLM)`` turn per task (single-turn). Backs
    RL scored rollouts, SDFT student generation, and benchmark eval.

    ``params`` only carries agent-behaviour overrides; model/sampling params
    (model_name, model_path, renderer_name, max_tokens, temperature, model_client)
    are injected by the rollout. Overrides are applied only when explicitly set, so
    they don't clobber the algorithm's own ``max_turns`` / ``system_prompt``."""

    name: ClassVar[str] = "basic_loop"
    agent_path: ClassVar[str] = "evsys_sdk.training.harbor_agents:BasicLoopAgent"

    class Config(_AgentParams):
        max_turns: int = 1
        system_prompt: str | None = None

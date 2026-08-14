"""The agent — an LLM agent the SDK spawns, as an extension point.

The SDK already spawns two headless Claude Code runs: the **trigger agent**
(the gatekeeper that judges an escalation and writes a verdict) and the
**autoresearch agent** (the researcher that improves artifacts / runs
experiments). Their mechanics were spread across ``triggers/agent.py``
(mission templates + argv assembly) and ``triggers/remote.py`` (a second,
near-identical argv assembly for the sandboxed stages). This module is that
shape factored out once:

  * **mission** — :meth:`EvsysAgent.build_prompt` renders what the agent is
    told to do (each subclass owns its templates);
  * **invocation** — :meth:`EvsysAgent.build_argv` turns a mission into the
    ``claude -p …`` argv (binary, permission mode, model, plugin dir, extra
    args — identical across agents, so it lives on the base exactly once);
  * **environment** — the first-class *where does it run* field: ``None``
    means a subprocess on this host; anything else is a sandbox
    ``{kind, params}`` spec resolved through the ``sandbox`` registry
    (:func:`~evsys_sdk.sandboxes.build_sandbox`).

    Why the sandbox registry and not harbor's ``BaseEnvironment``: harbor's
    environments are rollout-*task* environments — constructed from a task's
    ``environment_dir`` + docker-compose definition, a trial's ``TrialPaths``
    and session id, with async exec and per-service compose operations. An
    SDK-spawned agent has none of that context; it needs the five-method
    throwaway-box contract :class:`~evsys_sdk.sandboxes.base.BaseSandbox`
    already provides (and whose docstring already mirrors harbor's split
    deliberately). Composing with the sandbox registry keeps local/e2b/modal
    and user-registered providers all valid values of ``environment``.

Agents follow the repo-wide extension convention — two ClassVars (``name``,
``Config``) and a ``@register_agent`` decorator — so a project can define its
own::

    from evsys_sdk import EvsysAgent, register_agent

    @register_agent("red_team")
    class RedTeamAgent(EvsysAgent):
        class Config(EvsysAgent.Config):
            attack_budget: int = 5

        def build_prompt(self, **mission):
            return f"Probe the system. Budget: {self.cfg.attack_budget}"

The existing entry points (``triggers.agent.build_command``,
``triggers.remote.run_remote`` / ``run_prompt``) construct these classes and
delegate — their signatures and behavior are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from ..config import SandboxSpec
from ..logger import get_logger
from ..registry import register_agent

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Mission templates (moved verbatim from triggers/agent.py + triggers/remote.py;
# those modules re-export them, so existing imports keep working)
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = """You are the evsys **trigger agent**. The cheap deterministic gate just escalated \
a batch of production agent traces and it is your job to decide whether they are worth spending \
autoresearch budget on.

Escalation event: {escalation_path}
Ingested traces:  {traces_dir}
Live gate policy: {policy_path}
Write your verdict to: {verdict_path}

Do this:
1. Read the escalation event and the implicated traces. Use the `assess-traces` skill to judge whether \
this batch reflects a real, learnable failure mode (not noise).
2. Write a verdict JSON to the path above: \
{{"escalation": "<event file>", "worth_autoresearch": <bool>, "reasoning": "<why>", "hypothesis": "<what to try>"}}.
3. Consider retuning the gate: if it fired on noise (or missed obvious failures), edit {policy_path} \
using the `tune-trigger` skill — this is the self-improving gate.
4. {autoresearch_clause}

Keep it cheap and decisive: you are the gatekeeper, not the researcher."""

DISTILL_PROMPT = """You are the evsys **distiller agent**. The cheap deterministic gate just \
escalated a batch of coding-agent traces (Claude Code sessions). Your job is to convert them into \
evaluation + training data and launch the PRESET experiment — nothing more.

Escalation event: {escalation_path}
Ingested traces:  {traces_dir}
Live gate policy: {policy_path}
Write your verdict to: {verdict_path}
Preset experiment template: {experiment_template}
Holdout fraction: {holdout_fraction}
Benchmark dir: {benchmark_dir}   Train dir: {train_dir}

Hard rules:
- Do NOT invoke the `training-decider` agent. Do NOT design, choose, or tune a training \
algorithm — the algorithm is FIXED by the experiment template.
- Never let an eval session's data into the training rows (the holdout split is the \
contamination boundary).

Do this, following the `distill-traces` skill:
1. Assess the escalated traces (`assess-traces` skill) and write the verdict JSON \
({{"escalation", "worth_autoresearch", "reasoning", "hypothesis"}}) to the path above. If the \
batch is noise, stop here (you may retune {policy_path} via `tune-trigger`).
2. Split sessions chronologically: newest {holdout_fraction} of sessions -> eval, rest -> train. \
Write the eval set as a benchmark dir under {benchmark_dir} and training rows under {train_dir}.
3. Materialize the experiment FROM THE TEMPLATE (copy {experiment_template}, fill only names/paths), \
launch it, and monitor: poll its run outputs on a sensible cadence, abort on NaN/stalled loss, and \
write a short report next to the verdict when training ends.

Be decisive and cheap."""

AUTORESEARCH_ON = (
    "If (and only if) the batch is worth it, launch the autoresearch agent: invoke the "
    "`training-decider` agent with your hypothesis + the implicated trace ids so it designs and runs "
    "the next experiment."
)
AUTORESEARCH_OFF = (
    "Do NOT launch autoresearch — stop after writing the verdict (a later step consumes it)."
)

REMOTE_AUTORESEARCH_PROMPT = """You are the evsys **autoresearch agent**, running remotely. The \
trigger agent already judged this escalation worth fixing — your job is to actually improve the \
system's artifacts.

- Escalation event:  {escalation_path}
- Trigger-agent verdict (read its hypothesis): {verdict_path}
- Ingested traces (JSONL per source under this dir): {traces_dir}
- The artifacts you may improve (project-relative; ONLY these leave this sandbox): {artifacts}
- Skills: ./skills/  — the project's own improvement playbooks; follow them.

Do this:
1. Read the verdict's hypothesis and the implicated traces.
2. Design the smallest change to the listed artifacts that addresses the failure mode. You MAY run
   experiments with the evsys SDK (installed here): evaluations, and training/weight updates through
   hosted backends (e.g. tinker) — compute happens on the backend, not in this sandbox.
3. Rewrite the artifact(s). Anything you write outside the listed artifacts is discarded.

Be decisive; validate before you overwrite."""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class EvsysAgent:
    """One spawnable LLM agent: mission + invocation + environment.

    Construct with keyword params (validated against this class's ``Config``),
    via :meth:`from_config` (lifting the shared fields off a
    ``TriggerAgentConfig``-shaped object), or through the registry with
    :func:`~evsys_sdk.agents.build_agent`.
    """

    name: ClassVar[str] = ""

    class Config(BaseModel):
        model_config = {"extra": "forbid"}

        claude_bin: str = "claude"
        """The Claude Code executable to invoke."""
        model: str | None = None
        """Optional model override passed as ``--model``."""
        permission_mode: str = "acceptEdits"
        """Headless permission mode (non-interactive)."""
        plugin_dir: str | None = None
        """Path to the evsys-sdk plugin (``--plugin-dir``)."""
        extra_args: list[str] = Field(default_factory=list)
        """Extra argv appended to the ``claude`` command."""
        environment: SandboxSpec | str | None = None
        """WHERE the agent runs. ``None`` → a subprocess on this host; a
        sandbox ``{kind, params}`` spec (or bare provider name) → that
        provider, resolved through the sandbox registry."""

    def __init__(self, **params: Any) -> None:
        # Validate params against the subclass's Config (loud on a typo).
        self.cfg = self.Config(**params)

    # -- what the agent is told to do (subclass responsibility) ------------

    def build_prompt(self, **mission: Any) -> str:
        """Render the mission prompt. Subclasses own their templates."""
        raise NotImplementedError

    # -- how it is invoked (shared, exactly once) --------------------------

    def build_argv(self, prompt: str) -> list[str]:
        """The ``claude -p …`` argv for one mission. Pure — no side effects."""
        argv = [self.cfg.claude_bin, "-p", prompt,
                "--permission-mode", self.cfg.permission_mode]
        if self.cfg.model:
            argv += ["--model", self.cfg.model]
        if self.cfg.plugin_dir:
            argv += ["--plugin-dir", self.cfg.plugin_dir]
        argv += list(self.cfg.extra_args or [])
        return argv

    def build_command(self, *mission_args: Any, **mission: Any) -> list[str]:
        """Mission → argv, in one step."""
        return self.build_argv(self.build_prompt(*mission_args, **mission))

    # -- where it runs -----------------------------------------------------

    @property
    def environment(self) -> SandboxSpec | str | None:
        """The agent's declared execution environment (``None`` = this host)."""
        return self.cfg.environment

    def resolve_environment(self, *, envs: dict[str, str] | None = None,
                            timeout_s: float = 1800.0) -> Any:
        """``environment`` → an UNSTARTED :class:`BaseSandbox`, or ``None``
        for a host-process agent. Unstarted so callers can fix up ``envs``
        before the provider boots (see ``triggers/remote.py``)."""
        if self.cfg.environment is None:
            return None
        from ..sandboxes import build_sandbox

        return build_sandbox(self.cfg.environment, envs=envs,
                             timeout_s=timeout_s, start=False)

    # -- construction from the existing config surface ---------------------

    @classmethod
    def from_config(cls, agent_cfg: Any, **extra: Any) -> EvsysAgent:
        """Lift the shared invocation fields off a ``trigger.agent`` config.

        ``getattr`` with the historical defaults, so any agent_cfg-shaped
        object (tests pass plain namespaces) works exactly as before. The
        environment comes from ``remote.sandbox`` when ``remote.enabled``,
        mirroring the spawn dispatch.
        """
        remote = getattr(agent_cfg, "remote", None)
        environment = (getattr(remote, "sandbox", None)
                       if remote is not None and getattr(remote, "enabled", False)
                       else None)
        return cls(
            claude_bin=getattr(agent_cfg, "claude_bin", "claude"),
            model=getattr(agent_cfg, "model", None),
            permission_mode=getattr(agent_cfg, "permission_mode", "acceptEdits"),
            plugin_dir=getattr(agent_cfg, "plugin_dir", None),
            extra_args=list(getattr(agent_cfg, "extra_args", None) or []),
            environment=environment,
            **extra,
        )


# ---------------------------------------------------------------------------
# Built-ins: the two agents the SDK spawns today
# ---------------------------------------------------------------------------


@register_agent("trigger")
class TriggerAgent(EvsysAgent):
    """The gatekeeper: judges an escalation, writes a verdict, may retune the
    gate. ``mode: distill`` is the same agent on the distiller mission (convert
    traces to data, launch the preset experiment) — a config, not a class,
    because the invocation and verdict contract are identical."""

    class Config(EvsysAgent.Config):
        autoresearch: bool = True
        """When true, the mission allows launching ``training-decider`` on YES."""
        mode: Literal["verdict", "distill"] = "verdict"
        """Which mission template: gatekeeper or distiller."""
        prompt_template: str | None = None
        """Override the mission template (``{escalation_path}`` etc. formatted in)."""
        distill: Any = None
        """Distill-mode knobs (a :class:`~evsys_sdk.config.DistillConfig`);
        read with the historical fallbacks, so ``None`` is fine in verdict mode."""

    def build_prompt(self, escalation_path: str | Path = "", *,  # type: ignore[override]
                     root: str | Path = "", verdict_path: str | Path = "") -> str:
        escalation_path = Path(escalation_path)
        root = Path(root)
        default = DISTILL_PROMPT if self.cfg.mode == "distill" else DEFAULT_PROMPT
        template = self.cfg.prompt_template or default
        distill = self.cfg.distill
        return template.format(
            escalation_path=escalation_path,
            traces_dir=(root.parent / "traces"),
            policy_path=(root / "policy.json"),
            verdict_path=verdict_path,
            autoresearch_clause=(AUTORESEARCH_ON if self.cfg.autoresearch else AUTORESEARCH_OFF),
            experiment_template=getattr(distill, "experiment_template", ""),
            holdout_fraction=getattr(distill, "holdout_fraction", 0.2),
            benchmark_dir=getattr(distill, "benchmark_dir", "data/benchmark"),
            train_dir=getattr(distill, "train_dir", "data/train"),
        )

    @classmethod
    def from_config(cls, agent_cfg: Any, **extra: Any) -> TriggerAgent:
        return super().from_config(  # type: ignore[return-value]
            agent_cfg,
            autoresearch=getattr(agent_cfg, "autoresearch", True),
            mode=getattr(agent_cfg, "mode", "verdict"),
            prompt_template=getattr(agent_cfg, "prompt_template", None),
            distill=getattr(agent_cfg, "distill", None),
            **extra,
        )


@register_agent("autoresearch")
class AutoresearchAgent(EvsysAgent):
    """The researcher: improves the declared artifacts / runs experiments.

    Two mission shapes, matching the two entry points that spawn it:

      * a **direct prompt** (``run_prompt`` — an operator hands the agent its
        mission verbatim; nothing is templated);
      * an **escalation mission** (``run_remote`` stage 2 — the template gets
        the escalation, verdict, traces dir and artifact contract).
    """

    class Config(EvsysAgent.Config):
        prompt_template: str | None = None
        """Escalation-mission template override (the ``remote.
        autoresearch_prompt_template`` knob). ``None`` → the built-in
        :data:`REMOTE_AUTORESEARCH_PROMPT`."""

    def build_prompt(self, prompt: str | None = None, *,  # type: ignore[override]
                     escalation_path: str = "", verdict_path: str = "",
                     traces_dir: str = "", artifacts: str = "") -> str:
        if prompt is not None:
            return prompt
        template = self.cfg.prompt_template or REMOTE_AUTORESEARCH_PROMPT
        return template.format(
            escalation_path=escalation_path, verdict_path=verdict_path,
            traces_dir=traces_dir, artifacts=artifacts,
        )


__all__ = [
    "AUTORESEARCH_OFF",
    "AUTORESEARCH_ON",
    "DEFAULT_PROMPT",
    "DISTILL_PROMPT",
    "REMOTE_AUTORESEARCH_PROMPT",
    "AutoresearchAgent",
    "EvsysAgent",
    "TriggerAgent",
]

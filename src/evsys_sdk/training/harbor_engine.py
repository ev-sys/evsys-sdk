"""Harbor rollout runner — hands rollouts to harbor's ``Job`` engine (0.13.2).

Import-safe **without** the ``harbor`` package: all harbor imports are lazy
(inside the runners), and the agent / environment classes are referenced only by
**string import path** (they live in :mod:`evsys_sdk.training.harbor_agents`,
which harbor loads at trial runtime). So ``rl`` / ``sdft`` can import this
module, and tests can mock the runners, with no ``harbor`` install.

Flow (harbor 0.13.2): a producer's **adapter** (:class:`HarborTaskAdapter` for
scored rollouts, :class:`PromptAdapter` for generation) writes each task dir
(``instruction.md`` + ``task.toml`` [+ ``evsys_verifier.json`` spec + a dummy
``tests/test.sh`` when scored]) and returns harbor-native ``TaskConfig``\\s →
:func:`run_harbor_rollouts` builds a ``JobConfig`` over those ``TaskConfig``\\s x
one ``agent``, ``n_attempts = num_samples`` → ``Job.run()`` → harvest each
trial's ``agent_result`` (``rollout_details`` + completion + token/cost usage)
and ``verifier_result`` (reward) into a :class:`Trajectory`.

The reward is produced by harbor running our
:class:`~evsys_sdk.training.harbor_agents.EvsysVerifier` (the job-level verifier)
**host-side, no container**: it wraps the task's registered verifier fn over the
completion the agent wrote. SHARED verifier mode (the default) keeps it in the
agent's no-op environment; the dummy ``tests/test.sh`` only satisfies harbor's
task-load check and is never executed. (Generation-only rollouts disable the
verifier and use ``environment_mode="separate"`` so no ``test.sh`` is needed.)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..data_types import HarborTask, InProcessVerifier
from .trajectory import Trajectory, TrajectoryGroup, Turn

# Where harbor loads our glue classes from (by string, at trial runtime).
_AGENTS_PATH = "evsys_sdk.training.harbor_agents"

# Files the agent writes / the EvsysVerifier reads, host-side (no container).
_COMPLETION_FILE = "completion.txt"        # agent dir: the model's completion
_VERIFIER_SPEC_FILE = "evsys_verifier.json"  # task dir: {fn_name, expected, params}

# Rollout task.toml — SHARED verifier mode (default). Our EvsysVerifier (the
# job-level verifier) runs host-side; a dummy tests/test.sh satisfies harbor's
# load check and is never executed.
_ROLLOUT_TASK_TOML = "[agent]\ntimeout_sec = 600.0\n\n[environment]\n"
# Generation task.toml — verifier-less. environment_mode="separate" skips the
# test.sh load requirement (the job disables the verifier entirely).
_GENERATION_TASK_TOML = (
    '[agent]\ntimeout_sec = 600.0\n\n[environment]\n\n[verifier]\nenvironment_mode = "separate"\n'
)
_DUMMY_TEST_SH = "#!/bin/sh\nexit 0\n"


# ---------------------------------------------------------------------------
# Adapters: our data formats → harbor task dirs + native TaskConfigs
#
# A harbor task IS a directory (TaskConfig points at a path), so an adapter
# writes the dir and returns the harbor-native TaskConfig. The runner below is
# data-model-agnostic — it only ever sees TaskConfigs. Producers own their
# adapter: Benchmark/RL → HarborTaskAdapter (scored), SDFT → PromptAdapter.
# ---------------------------------------------------------------------------


class HarborTaskAdapter:
    """Adapt our ``HarborTask`` rows (the ``harbor_task`` / RL + eval format) to
    harbor task dirs + native ``TaskConfig``\\s — **scored** rollouts.

    ``to_harbor(output_dir)`` writes one task dir per task (``instruction.md`` +
    a SHARED-mode ``task.toml`` + a dummy ``tests/test.sh`` that only satisfies
    harbor's load check + the per-task ``evsys_verifier.json`` spec our host-side
    :class:`~evsys_sdk.training.harbor_agents.EvsysVerifier` reads) and returns
    the ``TaskConfig``\\s pointing at them. Requires an ``InProcessVerifier``.
    Used by ``run_harbor_rollouts(..., outcome_reward=True)`` (RL + benchmark eval).
    """

    def __init__(self, tasks: Sequence[HarborTask]) -> None:
        self._tasks = list(tasks)

    def to_harbor(self, output_dir: Path) -> list[Any]:
        from harbor.models.trial.config import TaskConfig

        configs: list[Any] = []
        for task in self._tasks:
            if not isinstance(task.verifier, InProcessVerifier):
                kind = getattr(task.verifier, "kind", type(task.verifier).__name__)
                raise RuntimeError(
                    f"harbor_engine: task {task.task_id!r} needs an in_process verifier "
                    f"for a scored rollout (got {kind!r}); only 'in_process' is supported."
                )
            name = _safe(task.task_id)
            dest = output_dir / name
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "instruction.md").write_text(task.instruction)
            (dest / "task.toml").write_text(_ROLLOUT_TASK_TOML)
            tests = dest / "tests"
            tests.mkdir(exist_ok=True)
            (tests / "test.sh").write_text(_DUMMY_TEST_SH)  # dummy — satisfies load, never run
            v = task.verifier
            (dest / _VERIFIER_SPEC_FILE).write_text(json.dumps({
                "fn_name": v.fn_name,
                "expected": v.expected,
                "params": dict(getattr(v, "params", None) or {}),
            }))
            configs.append(TaskConfig(path=dest))
        return configs


class PromptAdapter:
    """Adapt raw prompts (the generation / "prompt" format) to harbor task dirs +
    native ``TaskConfig``\\s — **generation-only** rollouts (no verifier, no
    reward). Used by SDFT student rollouts.

    ``to_harbor(output_dir)`` writes ``instruction.md`` + a verifier-less
    ``task.toml`` (``environment_mode="separate"`` skips the test.sh load check)
    per prompt and returns the ``TaskConfig``\\s. Used by
    ``run_harbor_rollouts(..., outcome_reward=False)`` (SDFT student rollouts).
    """

    def __init__(self, prompts: Sequence[str]) -> None:
        self._prompts = list(prompts)

    def to_harbor(self, output_dir: Path) -> list[Any]:
        from harbor.models.trial.config import TaskConfig

        configs: list[Any] = []
        for i, prompt in enumerate(self._prompts):
            name = _safe(f"gen_{i}")
            dest = output_dir / name
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "instruction.md").write_text(prompt)
            (dest / "task.toml").write_text(_GENERATION_TASK_TOML)
            configs.append(TaskConfig(path=dest))
        return configs


# ---------------------------------------------------------------------------
# Runner: Job.run() → TrajectoryGroups
# ---------------------------------------------------------------------------


def _agent_import_and_kwargs(
    model_client: str,
    *,
    agent_import_path: str | None,
    model_name: str,
    model_path: str | None,
    renderer_name: str | None,
    max_tokens: int,
    temperature: float,
    max_turns: int,
    system_prompt: str | None,
) -> tuple[str, dict[str, Any]]:
    """Pick the harbor agent + its kwargs for a rollout. Pure + harbor-free so
    the agent-selection logic is unit-testable.

    An explicit ``agent_import_path`` wins (fully self-configured agent, no
    kwargs). Otherwise it's always :class:`BasicLoopAgent`, parameterized by
    ``model_client``: ``"tinker"`` (on-policy ``TinkerLLM``, needs ``model_path``)
    or ``"litellm"`` (closed/API model; ``model_name`` is a litellm string, the
    tinker-only ``model_path``/``renderer_name`` are ignored).
    """
    if agent_import_path:
        return agent_import_path, {}
    return f"{_AGENTS_PATH}:BasicLoopAgent", {
        "model_name": model_name,
        "model_path": model_path,
        "renderer_name": renderer_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "max_turns": max_turns,
        "system_prompt": system_prompt,
        "model_client": model_client,
    }


def _to_agent_config(AgentConfig: Any, import_path: str, kwargs: dict[str, Any]) -> Any:
    """Map ``(import_path, agent kwargs)`` → a harbor ``AgentConfig``.

    harbor 0.13.2 passes ``model_name`` to the agent constructor from the
    top-level ``AgentConfig.model_name`` field, so it must NOT also live in
    ``kwargs`` (else the agent gets ``model_name`` twice). Lift it out here so
    the agent-selection logic above can stay a flat kwargs dict."""
    kwargs = dict(kwargs)
    return AgentConfig(
        import_path=import_path,
        model_name=kwargs.pop("model_name", None),
        kwargs=kwargs,
    )


async def run_harbor_rollouts(
    items: Sequence[Any],
    *,
    outcome_reward: bool = True,
    model_name: str,
    model_path: str | None,
    workspace_dir: Path,
    model_client: str = "tinker",
    renderer_name: str | None = None,
    num_samples: int = 1,
    max_turns: int = 1,
    max_tokens: int = 512,
    temperature: float = 1.0,
    system_prompt: str | None = None,
    agent_import_path: str | None = None,
    n_concurrent: int = 4,
    max_retries: int = 2,
    _job_factory: Any | None = None,
) -> list[TrajectoryGroup]:
    """Roll out ``items`` (x ``num_samples``) through harbor's ``Job`` engine —
    one :class:`TrajectoryGroup` per item, in order.

    ``outcome_reward`` is the agent-meaningful knob — does the rollout get scored
    by an outcome verifier? The runner is adapter-aware (it runs the matching
    adapter to write the task dirs, where ``materialize_task`` used to be), so no
    caller ever touches an adapter:

    * ``outcome_reward=True`` (default) — ``items`` are :class:`HarborTask`\\s;
      :class:`HarborTaskAdapter` writes scored task dirs and the host-side
      :class:`EvsysVerifier` produces each outcome reward. (RL + benchmark eval.)
    * ``outcome_reward=False`` — ``items`` are prompt strings; :class:`PromptAdapter`
      writes generation-only dirs (no verifier, ``reward=0``). (SDFT students.)

    ``model_client`` — ``"tinker"`` (on-policy ``TinkerLLM``, needs ``model_path``)
    or ``"litellm"`` (closed/API model; ``model_name`` a litellm string, e.g.
    ``"anthropic/claude-opus-4-1"``).

    ``_job_factory`` is the test seam: ``async (job_config) -> job_result``.
    When ``None``, harbor is imported and ``Job.create(...).run()`` is used.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Adapter-aware: our data → harbor task dirs + native TaskConfigs. The
    # outcome-reward mode picks the adapter (scored task vs generation prompt).
    adapter = (HarborTaskAdapter if outcome_reward else PromptAdapter)(items)
    task_configs = adapter.to_harbor(workspace_dir / "tasks")

    # Lazy harbor imports — keep this module importable without the extra.
    from harbor import Job
    from harbor.models.job.config import JobConfig, RetryConfig
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        VerifierConfig,
    )

    import_path, agent_kwargs = _agent_import_and_kwargs(
        model_client,
        agent_import_path=agent_import_path,
        model_name=model_name,
        model_path=model_path,
        renderer_name=renderer_name,
        max_tokens=max_tokens,
        temperature=temperature,
        max_turns=max_turns,
        system_prompt=system_prompt,
    )
    agent = _to_agent_config(AgentConfig, import_path, agent_kwargs)
    config = JobConfig(
        tasks=task_configs,
        agents=[agent],
        environment=EnvironmentConfig(import_path=f"{_AGENTS_PATH}:NoOpEnvironment"),
        # outcome_reward: host-side EvsysVerifier wraps the registered fn (SHARED
        # mode, no container) → reward per trajectory. Else: no verifier, reward 0.
        verifier=(VerifierConfig(import_path=f"{_AGENTS_PATH}:EvsysVerifier")
                  if outcome_reward else VerifierConfig(disable=True)),
        jobs_dir=workspace_dir / "jobs",
        n_concurrent_trials=n_concurrent,
        n_attempts=num_samples,                    # repeats per task = samples
        retry=RetryConfig(max_retries=max_retries),
    )

    result = await (_job_factory(config) if _job_factory is not None
                    else _run_job(Job, config))
    return _harvest(result, task_configs)


async def _run_job(Job: Any, config: Any) -> Any:
    job = await Job.create(config)
    return await job.run()


# ---------------------------------------------------------------------------
# Harvest: JobResult → TrajectoryGroups (one per task); reward from the verifier
# ---------------------------------------------------------------------------


def _trials_by_task(job_result: Any) -> dict[str, list[Any]]:
    """Group a job's trial results by ``task_name`` (the materialized dir's
    basename = ``_safe(task_id)``). ``n_attempts`` trials share a task_name."""
    out: dict[str, list[Any]] = {}
    for tr in (getattr(job_result, "trial_results", None) or []):
        out.setdefault(getattr(tr, "task_name", None), []).append(tr)
    return out


def _harvest(job_result: Any, task_configs: Sequence[Any]) -> list[TrajectoryGroup]:
    by_task = _trials_by_task(job_result)
    groups: list[TrajectoryGroup] = []
    for tc in task_configs:
        # harbor derives a trial's ``task_name`` from the task dir basename.
        task_name = Path(tc.path).name
        trajs = [
            traj for tr in by_task.get(task_name, [])
            if (traj := _trial_to_trajectory(tr)) is not None
        ]
        groups.append(TrajectoryGroup(trajectories=trajs))
    return groups


def _trial_to_trajectory(tr: Any) -> Trajectory | None:
    """Convert a harbor ``TrialResult`` → our multi-turn :class:`Trajectory`,
    reading the rollout off ``agent_result`` (``AgentContext``).

    Token-level turns come from ``rollout_details`` (tinker on-policy rollouts).
    Closed/API models (litellm) return no token ids, so for an eval trial — one
    that produced a verifier reward — we still build a *token-less* Trajectory
    carrying the reward + usage so it isn't dropped from scoring. Errored trials,
    and generation-only trials with neither tokens nor a reward, return ``None``.
    """
    if tr is None or getattr(tr, "exception_info", None):
        return None
    agent_result = getattr(tr, "agent_result", None)
    details = getattr(agent_result, "rollout_details", None) if agent_result else None
    rd = details[0] if details else {}   # main chat history; empty for API models
    prompt_turns = rd.get("prompt_token_ids") or []
    completion_turns = rd.get("completion_token_ids") or []
    logprob_turns = rd.get("logprobs") or []

    rewards = getattr(getattr(tr, "verifier_result", None), "rewards", None)
    if not completion_turns and rewards is None:
        return None  # no tokens and no score → nothing to harvest (generation / failed)

    turns: list[Turn] = []
    for i, completion in enumerate(completion_turns):
        turns.append(Turn(
            prompt_tokens=list(prompt_turns[i]) if i < len(prompt_turns) else [],
            completion_tokens=list(completion),
            logprobs=list(logprob_turns[i]) if i < len(logprob_turns) else [],
        ))

    usage = _trial_usage(tr)
    if usage["prompt_tokens"] is None:
        usage["prompt_tokens"] = sum(len(t.prompt_tokens) for t in turns)
    if usage["completion_tokens"] is None:
        usage["completion_tokens"] = sum(len(t.completion_tokens) for t in turns)

    # Reward from harbor's verifier (our host-side EvsysVerifier); 0.0 for
    # generation-only rollouts (verifier disabled) or when absent.
    reward = float((rewards or {}).get("reward", 0.0))
    return Trajectory(turns=turns, reward=reward, metadata={"usage": usage})


def _trial_usage(tr: Any) -> dict[str, Any]:
    """Pull harbor's native cost / token / timing info off a trial result.

    Harbor records ``cost_usd`` + token counts on ``agent_result``
    (``AgentContext``) and per-phase wall-clock timing on the trial
    (``agent_execution``, whole-trial span as fallback). Any field harbor didn't
    populate stays ``None`` — on-policy tinker has no API ``cost_usd``, and the
    caller backfills token counts from the turns. Pure + harbor-free."""
    ar = getattr(tr, "agent_result", None)
    return {
        "cost_usd": getattr(ar, "cost_usd", None),
        "prompt_tokens": getattr(ar, "n_input_tokens", None),
        "completion_tokens": getattr(ar, "n_output_tokens", None),
        "cached_tokens": getattr(ar, "n_cache_tokens", None),
        "latency_s": _phase_seconds(getattr(tr, "agent_execution", None))
        or _phase_seconds(tr),
    }


def _phase_seconds(phase: Any) -> float | None:
    """Wall-clock seconds for a harbor timing phase — anything carrying
    ``started_at`` / ``finished_at`` datetimes. ``None`` when either is missing."""
    started = getattr(phase, "started_at", None)
    finished = getattr(phase, "finished_at", None)
    if started is None or finished is None:
        return None
    try:
        return (finished - started).total_seconds()
    except (TypeError, AttributeError):  # pragma: no cover - defensive
        return None


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))


__all__ = ["HarborTaskAdapter", "PromptAdapter", "run_harbor_rollouts"]

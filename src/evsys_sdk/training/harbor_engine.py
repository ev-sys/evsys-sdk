"""Harbor rollout runner — hands rollouts to harbor's ``Job`` engine (0.13.2).

Import-safe **without** the ``harbor`` package: all harbor imports are lazy
(inside the runners), and the agent / environment classes are referenced only by
**string import path** (they live in :mod:`evsys_sdk.training.harbor_agents`,
which harbor loads at trial runtime). So ``rl`` / ``sdft`` can import this
module, and tests can mock the runners, with no ``harbor`` install.

Flow (harbor 0.13.2): materialize each :class:`~evsys_sdk.data_types.HarborTask`
→ a minimal task dir (``instruction.md`` + a ``task.toml`` whose
``environment_mode="separate"`` skips the ``test.sh`` requirement) → build a
``JobConfig`` over ``tasks`` × one ``agent``, ``n_attempts = num_samples``,
verifier **disabled** → ``Job.run()`` → harvest each trial's ``agent_result``
(``rollout_details`` + completion text + token/cost usage) into a
:class:`Trajectory`, then **score the reward in Python** from the completion and
the task's verifier fn.

Why Python scoring: harbor 0.13.2's verifier runs host-side against files synced
out of a *container*. Our agents run in-process with a no-op environment, so
there's nothing to sync — we disable the harbor verifier and score completions
ourselves (same registered verifier fns the rest of the SDK uses).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..data_types import HarborTask, InProcessVerifier
from .trajectory import Trajectory, TrajectoryGroup, Turn

# Where harbor loads our glue classes from (by string, at trial runtime).
_AGENTS_PATH = "evsys_sdk.training.harbor_agents"

# task.toml that loads with no test.sh + no per-task verifier (we score in
# Python, and disable harbor's verifier at the job level).
_TASK_TOML = (
    "[agent]\n"
    "timeout_sec = 600.0\n\n"
    "[environment]\n\n"
    "[verifier]\n"
    'environment_mode = "separate"   # skip the test.sh requirement at load\n'
)


# ---------------------------------------------------------------------------
# Task materializer (minimal dir: task.toml + instruction.md)
# ---------------------------------------------------------------------------


def materialize_task(task: HarborTask, dest: Path) -> Path:
    """Write a minimal harbor task dir for ``task`` at ``dest`` — only
    ``task.toml`` + ``instruction.md``. The reward is scored in Python from the
    task's verifier fn, so the harbor verifier is disabled (not in task.toml)."""
    if not isinstance(task.verifier, InProcessVerifier):
        raise RuntimeError(
            f"harbor_engine: task {task.task_id!r} has a {task.verifier.kind!r} "
            "verifier; only 'in_process' is supported in the rollout path today."
        )
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "instruction.md").write_text(task.instruction)
    (dest / "task.toml").write_text(_TASK_TOML)
    return dest


def _agent_config(
    AgentConfig: Any,
    agent_import_path: str | None,
    *,
    model_name: str,
    model_path: str | None,
    renderer_name: str | None,
    max_tokens: int,
    temperature: float,
    max_turns: int,
    system_prompt: str | None,
) -> Any:
    return AgentConfig(
        import_path=agent_import_path or f"{_AGENTS_PATH}:BasicLoopAgent",
        kwargs={} if agent_import_path else {
            "model_name": model_name,
            "model_path": model_path,
            "renderer_name": renderer_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "max_turns": max_turns,
            "system_prompt": system_prompt,
        },
    )


# ---------------------------------------------------------------------------
# Runner: Job.run() → TrajectoryGroups
# ---------------------------------------------------------------------------


async def run_harbor_rollouts(
    tasks: Sequence[HarborTask],
    *,
    model_name: str,
    model_path: str | None,
    workspace_dir: Path,
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
    """Roll out ``tasks`` (× ``num_samples``) through harbor's ``Job`` engine and
    score each rollout in Python.

    ``_job_factory`` is the test seam: ``async (job_config) -> job_result``.
    When ``None``, harbor is imported and ``Job.create(...).run()`` is used.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Lazy harbor imports — keep this module importable without the extra.
    from harbor import Job
    from harbor.models.job.config import JobConfig, RetryConfig
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        VerifierConfig,
    )

    agent = _agent_config(
        AgentConfig, agent_import_path, model_name=model_name, model_path=model_path,
        renderer_name=renderer_name, max_tokens=max_tokens, temperature=temperature,
        max_turns=max_turns, system_prompt=system_prompt,
    )
    task_cfgs = [
        TaskConfig(path=materialize_task(t, workspace_dir / "tasks" / _safe(t.task_id)))
        for t in tasks
    ]
    config = JobConfig(
        tasks=task_cfgs,
        agents=[agent],
        environment=EnvironmentConfig(import_path=f"{_AGENTS_PATH}:NoOpEnvironment"),
        verifier=VerifierConfig(disable=True),     # we score in Python (below)
        jobs_dir=workspace_dir / "jobs",
        n_concurrent_trials=n_concurrent,
        n_attempts=num_samples,                    # repeats per task = samples
        retry=RetryConfig(max_retries=max_retries),
    )

    result = await (_job_factory(config) if _job_factory is not None
                    else _run_job(Job, config))
    return _harvest(result, tasks, score=True)


async def _run_job(Job: Any, config: Any) -> Any:
    job = await Job.create(config)
    return await job.run()


# ---------------------------------------------------------------------------
# Generation-only rollouts (no reward) — used by SDFT's student rollout
# ---------------------------------------------------------------------------


async def run_harbor_generations(
    prompts: Sequence[str],
    *,
    model_name: str,
    model_path: str | None,
    workspace_dir: Path,
    renderer_name: str | None = None,
    max_turns: int = 1,
    max_tokens: int = 512,
    temperature: float = 1.0,
    system_prompt: str | None = None,
    agent_import_path: str | None = None,
    n_concurrent: int = 4,
    max_retries: int = 2,
    _job_factory: Any | None = None,
) -> list[Trajectory]:
    """One generation per prompt through harbor's engine, **no reward**.

    Returns one :class:`Trajectory` per prompt (``reward=0``); SDFT uses the
    student completion tokens for teacher-forced distillation.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)

    from harbor import Job
    from harbor.models.job.config import JobConfig, RetryConfig
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        VerifierConfig,
    )

    agent = _agent_config(
        AgentConfig, agent_import_path, model_name=model_name, model_path=model_path,
        renderer_name=renderer_name, max_tokens=max_tokens, temperature=temperature,
        max_turns=max_turns, system_prompt=system_prompt,
    )
    task_cfgs = []
    for i, prompt in enumerate(prompts):
        dest = workspace_dir / "tasks" / f"gen_{i}"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "instruction.md").write_text(prompt)
        (dest / "task.toml").write_text(_TASK_TOML)
        task_cfgs.append(TaskConfig(path=dest))

    config = JobConfig(
        tasks=task_cfgs,
        agents=[agent],
        environment=EnvironmentConfig(import_path=f"{_AGENTS_PATH}:NoOpEnvironment"),
        verifier=VerifierConfig(disable=True),
        jobs_dir=workspace_dir / "jobs",
        n_concurrent_trials=n_concurrent,
        n_attempts=1,
        retry=RetryConfig(max_retries=max_retries),
    )

    result = await (_job_factory(config) if _job_factory is not None
                    else _run_job(Job, config))
    by_task = _trials_by_task(result)
    out: list[Trajectory] = []
    for i in range(len(prompts)):
        trs = by_task.get(f"gen_{i}", [])
        traj = _trial_to_trajectory(trs[0]) if trs else None
        out.append(traj if traj is not None else Trajectory(turns=[]))
    return out


# ---------------------------------------------------------------------------
# Harvest: JobResult → TrajectoryGroups (one per task), with Python scoring
# ---------------------------------------------------------------------------


def _trials_by_task(job_result: Any) -> dict[str, list[Any]]:
    """Group a job's trial results by ``task_name`` (the materialized dir's
    basename = ``_safe(task_id)``). ``n_attempts`` trials share a task_name."""
    out: dict[str, list[Any]] = {}
    for tr in (getattr(job_result, "trial_results", None) or []):
        out.setdefault(getattr(tr, "task_name", None), []).append(tr)
    return out


def _harvest(job_result: Any, tasks: Sequence[HarborTask], *, score: bool) -> list[TrajectoryGroup]:
    by_task = _trials_by_task(job_result)
    groups: list[TrajectoryGroup] = []
    for t in tasks:
        trajs: list[Trajectory] = []
        for tr in by_task.get(_safe(t.task_id), []):
            traj = _trial_to_trajectory(tr)
            if traj is None:
                continue
            if score:
                traj.reward = _score_completion(t, traj.metadata.get("completion", ""))
            trajs.append(traj)
        groups.append(TrajectoryGroup(
            trajectories=trajs, tags=list(t.metadata.get("tags") or []),
        ))
    return groups


def _score_completion(task: HarborTask, completion: str) -> float:
    """Reward = the task's registered verifier fn over the completion + expected."""
    v = task.verifier
    if not isinstance(v, InProcessVerifier):
        return 0.0
    from ..verifiers import get_verifier_fn

    try:
        fn = get_verifier_fn(v.fn_name)
        return float(fn(completion, v.expected, dict(getattr(v, "params", None) or {})))
    except Exception:  # pragma: no cover - a bad verifier fn shouldn't crash the batch
        return 0.0


def _trial_to_trajectory(tr: Any) -> Trajectory | None:
    """Convert a harbor ``TrialResult`` → our multi-turn :class:`Trajectory`,
    reading the rollout off ``agent_result`` (``AgentContext``)."""
    if tr is None:
        return None
    agent_result = getattr(tr, "agent_result", None)
    details = getattr(agent_result, "rollout_details", None) if agent_result else None
    if not details:
        return None
    rd = details[0]  # the main linear chat history
    prompt_turns = rd.get("prompt_token_ids") or []
    completion_turns = rd.get("completion_token_ids") or []
    logprob_turns = rd.get("logprobs") or []
    if not completion_turns:
        return None

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

    completion_text = ""
    meta = getattr(agent_result, "metadata", None) or {}
    if isinstance(meta, dict):
        completion_text = meta.get("completion") or ""
    # reward is set by the caller (_harvest) via Python scoring.
    return Trajectory(turns=turns, reward=0.0,
                      metadata={"usage": usage, "completion": completion_text})


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


__all__ = ["materialize_task", "run_harbor_rollouts", "run_harbor_generations"]

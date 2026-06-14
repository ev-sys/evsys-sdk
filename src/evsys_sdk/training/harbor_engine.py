"""Harbor rollout runner — hands rollouts to harbor's ``Job`` engine.

This module is import-safe **without** the ``harbor`` package: all harbor
imports are lazy (inside :func:`run_harbor_rollouts`), and the harbor agent /
environment / verifier classes are referenced only by **string import path**
(they live in :mod:`evsys_sdk.training.harbor_agents`, which harbor loads at
trial runtime). So ``rl`` / ``sdft`` can import this module, and tests can mock
:func:`run_harbor_rollouts`, with no ``harbor`` install.

Flow: materialize each :class:`~evsys_sdk.data_types.HarborTask` → a minimal
task dir (``task.toml`` with ``[verifier] environment_mode="separate"`` so the
``test.sh`` check is skipped + ``instruction.md``) → build a ``JobConfig`` over
the batch → ``Job.run()`` (retries + bounded concurrency + persistence) →
harvest each trial's ``rollout_details`` + reward into a
:class:`~evsys_sdk.training.env.TrajectoryGroup`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import tinker

from ..data_types import HarborTask, InProcessVerifier
from .env import Trajectory, TrajectoryGroup

# Where harbor loads our glue classes from (by string, at trial runtime).
_AGENTS_PATH = "evsys_sdk.training.harbor_agents"
_COMPLETION_FILE = "completion.txt"


# ---------------------------------------------------------------------------
# Task materializer (minimal dir: task.toml + instruction.md, no Dockerfile/test.sh)
# ---------------------------------------------------------------------------


def materialize_task(task: HarborTask, dest: Path) -> Path:
    """Write a minimal harbor task dir for ``task`` at ``dest`` — only
    ``task.toml`` + ``instruction.md``. ``environment_mode = "separate"`` makes
    the harbor ``Task`` load skip the ``test.sh`` requirement; scoring is our
    in-process ``EvsysVerifier``."""
    if not isinstance(task.verifier, InProcessVerifier):
        raise RuntimeError(
            f"harbor_engine: task {task.task_id!r} has a {task.verifier.kind!r} "
            "verifier; only 'in_process' is supported in the rollout path today."
        )
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "instruction.md").write_text(task.instruction)
    (dest / "task.toml").write_text(_task_toml(task))
    return dest


def _task_toml(task: HarborTask) -> str:
    v = task.verifier
    assert isinstance(v, InProcessVerifier)
    expected = json.dumps(v.expected) if v.expected is not None else '""'
    return (
        "[agent]\n"
        "timeout_sec = 600.0\n\n"
        "[environment]\n\n"
        "[verifier]\n"
        'environment_mode = "separate"   # skips the test.sh requirement at load\n'
        f'import_path = "{_AGENTS_PATH}:EvsysVerifier"\n\n'
        "[verifier.kwargs]\n"
        f'fn_name = "{v.fn_name}"\n'
        f"expected = {expected}\n"
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
    """Roll out ``tasks`` (× ``num_samples``) through harbor's ``Job`` engine.

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
        TrialConfig,
        VerifierConfig,
    )

    agent = AgentConfig(
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

    trial_configs = []
    for t in tasks:
        task_dir = materialize_task(t, workspace_dir / "tasks" / _safe(t.task_id))
        for s in range(num_samples):
            trial_configs.append(TrialConfig(
                task=TaskConfig(path=task_dir),
                trial_name=_trial_name(t.task_id, s),
                trials_dir=workspace_dir / "trials",
                agent=agent,
                environment=EnvironmentConfig(import_path=f"{_AGENTS_PATH}:NoOpEnvironment"),
                verifier=VerifierConfig(),  # the task.toml carries import_path + kwargs
            ))

    config = JobConfig(
        trials=trial_configs,
        jobs_dir=workspace_dir / "jobs",
        n_concurrent_trials=n_concurrent,
        retry=RetryConfig(max_retries=max_retries),
    )

    if _job_factory is not None:
        result = await _job_factory(config)
    else:
        job = await Job.create(config)
        result = await job.run()

    return _harvest(result, tasks, num_samples)


# ---------------------------------------------------------------------------
# Harvest: JobResult → TrajectoryGroups (one per task)
# ---------------------------------------------------------------------------


def _harvest(job_result: Any, tasks: Sequence[HarborTask], num_samples: int) -> list[TrajectoryGroup]:
    by_trial = {
        tr.trial_name: tr
        for tr in (getattr(job_result, "trial_results", None) or [])
    }
    groups: list[TrajectoryGroup] = []
    for t in tasks:
        trajs: list[Trajectory] = []
        for s in range(num_samples):
            traj = _trial_to_trajectory(by_trial.get(_trial_name(t.task_id, s)))
            if traj is not None:
                trajs.append(traj)
        groups.append(TrajectoryGroup(
            trajectories=trajs, tags=list(t.metadata.get("tags") or []),
        ))
    return groups


def _trial_to_trajectory(tr: Any) -> Trajectory | None:
    if tr is None:
        return None
    agent_result = getattr(tr, "agent_result", None)
    details = getattr(agent_result, "rollout_details", None) if agent_result else None
    if not details:
        return None
    rd = details[0]  # single linear chat history
    prompt_turns = rd.get("prompt_token_ids") or []
    completion_turns = rd.get("completion_token_ids") or []
    logprob_turns = rd.get("logprobs") or []
    if not completion_turns:
        return None
    # Single-turn Trajectory shape: use the last turn.
    prompt_ids = list(prompt_turns[-1]) if prompt_turns else []
    completion = list(completion_turns[-1])
    logprobs = list(logprob_turns[-1]) if logprob_turns else []

    rewards = getattr(getattr(tr, "verifier_result", None), "rewards", None) or {}
    reward = float(rewards.get("reward", 0.0))

    return Trajectory(
        prompt=tinker.ModelInput.from_ints(prompt_ids),
        completion_tokens=completion,
        completion_logprobs=logprobs,
        reward=reward,
    )


def _trial_name(task_id: str, sample: int) -> str:
    return f"{_safe(task_id)}__s{sample}"


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))


__all__ = ["materialize_task", "run_harbor_rollouts"]

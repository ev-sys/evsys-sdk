"""Benchmark / validation evaluation through harbor's rollout engine.

Eval reuses the *same* engine as training: a benchmark is a set of
:class:`~evsys_sdk.data_types.HarborTask`\\s (instruction + verifier), so
scoring it is just :func:`~evsys_sdk.training.harbor_engine.run_harbor_rollouts`
over those tasks — the verifier reward *is* the eval score.

Unlike training, eval rollouts are **uploaded to the dashboard** (Supabase)
with ``kind='eval'`` via :func:`upload_eval_rollouts`. (Training rollouts stay
on disk in the run workspace and are never uploaded.)

The metrics / prediction builders are pure functions over
:class:`TrajectoryGroup`\\s — harbor-free and directly testable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from ..data_types import HarborTask
from .trajectory import TrajectoryGroup

logger = logging.getLogger(__name__)


async def score_via_harbor(
    tasks: Sequence[HarborTask],
    *,
    model_name: str,
    model_path: str | None,
    workspace_dir: Path,
    num_samples: int = 1,
    max_turns: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.0,
    renderer_name: str | None = None,
    system_prompt: str | None = None,
    agent_import_path: str | None = None,
    n_concurrent: int = 8,
    max_retries: int = 2,
    _job_factory: Any | None = None,
) -> list[TrajectoryGroup]:
    """Score ``tasks`` through harbor (one TrajectoryGroup per task, rewards
    from each task's verifier). Thin wrapper over the shared rollout engine."""
    from .harbor_engine import run_harbor_rollouts

    return await run_harbor_rollouts(
        tasks,
        model_name=model_name,
        model_path=model_path,
        workspace_dir=workspace_dir,
        renderer_name=renderer_name,
        num_samples=num_samples,
        max_turns=max_turns,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        agent_import_path=agent_import_path,
        n_concurrent=n_concurrent,
        max_retries=max_retries,
        _job_factory=_job_factory,
    )


# ---------------------------------------------------------------------------
# Pure: metrics + prediction rows (harbor-free)
# ---------------------------------------------------------------------------


def eval_metrics(
    groups: Sequence[TrajectoryGroup],
    *,
    metrics: Sequence[str] | None = None,
) -> dict[str, float]:
    """Reduce per-task rollout rewards to the benchmark's declared metrics.

    ``metrics`` is a list of registered metric names (e.g. ``["pass@3",
    "pass^3", "avg"]``); each is looked up via :func:`get_metric` and applied
    to the per-task sample rewards (one inner list per task, holding that
    task's ``num_samples`` rewards). ``n_tasks`` is always included. When no
    metrics are declared, defaults to ``mean_reward`` + ``pass_rate`` for
    back-compat."""
    from ..registry import get_metric

    task_rewards = [list(g.rewards) for g in groups if g.rewards]
    names = list(metrics) if metrics else ["mean_reward", "pass_rate"]
    out: dict[str, float] = {"n_tasks": float(len(task_rewards))}
    for name in names:
        try:
            out[name] = float(get_metric(name)().compute(task_rewards))
        except Exception:
            logger.warning("eval metric %r failed; skipping", name, exc_info=True)
    return out


def eval_predictions(
    tasks: Sequence[HarborTask],
    groups: Sequence[TrajectoryGroup],
    *,
    eval_id: str | None = None,
    step: int | None = None,
) -> list[dict]:
    """Build dashboard prediction rows (``kind='eval'``) — one per
    (task, sample). Carries the token-level rollout + reward for the eval."""
    rows: list[dict] = []
    for task, group in zip(tasks, groups):
        for sample_idx, traj in enumerate(group.trajectories):
            last = traj.turns[-1] if traj.turns else None
            rows.append({
                "kind": "eval",
                "eval_id": eval_id,
                "task_id": task.task_id,
                "sample_idx": sample_idx,
                "step": step,
                "instruction": task.instruction,
                "expected": getattr(task.verifier, "expected", None),
                "reward": traj.reward,
                "completion_token_ids": last.completion_tokens if last else [],
                "metadata": dict(task.metadata),
            })
    return rows


# ---------------------------------------------------------------------------
# Upload (eval only — training rollouts are never uploaded)
# ---------------------------------------------------------------------------


def upload_eval_rollouts(store: Any, run_id: str, predictions: list[dict]) -> None:
    """Upload eval predictions to the dashboard. Accepts either a
    ``DashboardClient`` (``log_predictions``) or an ``EvsysStore``
    (``add_prediction`` per row). No-op when ``store``/``run_id`` is falsy."""
    if not store or not run_id or not predictions:
        return
    if hasattr(store, "log_predictions"):
        store.log_predictions(run_id, predictions)
        return
    if hasattr(store, "add_prediction"):
        for p in predictions:
            store.add_prediction(
                run_id=run_id,
                kind=p.get("kind", "eval"),
                eval_id=p.get("eval_id"),
                task_id=p.get("task_id"),
                instruction=p.get("instruction"),
                expected=p.get("expected"),
                reward=p.get("reward"),
                step=p.get("step"),
                sample_idx=p.get("sample_idx", 0),
                metadata={
                    **(p.get("metadata") or {}),
                    "completion_token_ids": p.get("completion_token_ids", []),
                },
            )


__all__ = ["score_via_harbor", "eval_metrics", "eval_predictions", "upload_eval_rollouts"]

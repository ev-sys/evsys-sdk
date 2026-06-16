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

from pathlib import Path
from typing import Any, Sequence

from ..data_types import HarborTask
from .trajectory import TrajectoryGroup


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


def eval_metrics(groups: Sequence[TrajectoryGroup]) -> dict[str, float]:
    """Aggregate reward + economics stats over the eval rollouts. Always
    reports ``{mean_reward, pass_rate, n_tasks}`` (a completion passes when its
    reward >= 1.0); adds ``{time_per_task, tokens_per_task, cost_per_task}``
    whenever harbor reported the underlying usage (cost is omitted for runs
    with no API price, e.g. on-policy tinker). Every stat is a mean over the
    per-task mean (so ``num_samples`` > 1 averages within a task first)."""
    task_means: list[float] = []
    passes = 0
    total = 0
    times: list[float] = []
    tokens: list[float] = []
    costs: list[float] = []
    for g in groups:
        rewards = g.rewards
        if not rewards:
            continue
        task_means.append(sum(rewards) / len(rewards))
        passes += sum(1 for r in rewards if r >= 1.0)
        total += len(rewards)
        u = _task_usage_means(g)
        if u["latency_s"] is not None:
            times.append(u["latency_s"])
        if u["tokens"] is not None:
            tokens.append(u["tokens"])
        if u["cost_usd"] is not None:
            costs.append(u["cost_usd"])
    n = len(task_means)
    out: dict[str, float] = {
        "mean_reward": (sum(task_means) / n) if n else 0.0,
        "pass_rate": (passes / total) if total else 0.0,
        "n_tasks": float(n),
    }
    if times:
        out["time_per_task"] = sum(times) / len(times)
    if tokens:
        out["tokens_per_task"] = sum(tokens) / len(tokens)
    if costs:
        out["cost_per_task"] = sum(costs) / len(costs)
    return out


def _task_usage_means(group: TrajectoryGroup) -> dict[str, float | None]:
    """Per-task mean latency / token count / cost over the group's
    trajectories, reading the ``metadata['usage']`` harbor_engine stamps on
    each rollout. A field is ``None`` when no trajectory reported it."""
    lat: list[float] = []
    toks: list[float] = []
    cost: list[float] = []
    for t in group.trajectories:
        u = (t.metadata or {}).get("usage") or {}
        if u.get("latency_s") is not None:
            lat.append(float(u["latency_s"]))
        pt, ct = u.get("prompt_tokens"), u.get("completion_tokens")
        if pt is not None or ct is not None:
            toks.append(float(pt or 0) + float(ct or 0))
        if u.get("cost_usd") is not None:
            cost.append(float(u["cost_usd"]))
    return {
        "latency_s": (sum(lat) / len(lat)) if lat else None,
        "tokens": (sum(toks) / len(toks)) if toks else None,
        "cost_usd": (sum(cost) / len(cost)) if cost else None,
    }


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
            usage = (traj.metadata or {}).get("usage") or {}
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
                # Per-task time + tokens are persisted inside ``metadata`` — the
                # one JSON column both the local and remote prediction stores
                # keep. Top-level prediction columns are fixed, so anything
                # outside ``metadata`` is dropped by the remote backend.
                "metadata": {
                    **dict(task.metadata),
                    "latency_s": usage.get("latency_s"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                },
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
                    # ``metadata`` already carries per-task latency_s +
                    # prompt/completion_tokens (built in eval_predictions);
                    # fold in the rollout token ids alongside them.
                    **(p.get("metadata") or {}),
                    "completion_token_ids": p.get("completion_token_ids", []),
                },
            )


__all__ = ["score_via_harbor", "eval_metrics", "eval_predictions", "upload_eval_rollouts"]

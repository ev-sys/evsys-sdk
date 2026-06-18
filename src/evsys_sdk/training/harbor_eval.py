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
from typing import Any, Sequence

from ..data_types import HarborTask
from .trajectory import TrajectoryGroup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure: metrics + prediction rows (harbor-free)
# ---------------------------------------------------------------------------


def eval_metrics(
    groups: Sequence[TrajectoryGroup],
    *,
    metrics: Sequence[str] | None = None,
) -> dict[str, float]:
    """Reduce per-task rollout rewards to the benchmark's declared metrics,
    plus per-task economics.

    ``metrics`` is a list of registered metric names (e.g. ``["pass@3",
    "pass^3", "avg"]``); each is looked up via :func:`get_metric` and applied
    to the per-task sample rewards (one inner list per task, holding that task's
    ``num_samples`` rewards). ``n_tasks`` is always included; when no metrics
    are declared it defaults to ``mean_reward`` + ``pass_rate``.

    Independently, ``{time_per_task, tokens_per_task, cost_per_task}`` are added
    whenever harbor reported the underlying usage (cost is omitted for runs with
    no API price, e.g. on-policy tinker)."""
    from ..registry import get_metric

    task_rewards = [list(g.rewards) for g in groups if g.rewards]
    names = list(metrics) if metrics else ["mean_reward", "pass_rate"]
    out: dict[str, float] = {"n_tasks": float(len(task_rewards))}
    for name in names:
        try:
            out[name] = float(get_metric(name)().compute(task_rewards))
        except Exception:
            logger.warning("eval metric %r failed; skipping", name, exc_info=True)

    # Per-task economics (independent of the reward metrics above).
    times: list[float] = []
    tokens: list[float] = []
    costs: list[float] = []
    for g in groups:
        if not g.rewards:
            continue
        u = _task_usage_means(g)
        if u["latency_s"] is not None:
            times.append(u["latency_s"])
        if u["tokens"] is not None:
            tokens.append(u["tokens"])
        if u["cost_usd"] is not None:
            costs.append(u["cost_usd"])
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


__all__ = ["eval_metrics", "eval_predictions", "upload_eval_rollouts"]

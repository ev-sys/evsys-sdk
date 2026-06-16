"""Tests for the harbor eval helpers' harbor-free parts: metrics, prediction
rows, and the eval-only upload (training rollouts are never uploaded)."""

from __future__ import annotations

import pytest

pytest.importorskip("tinker")
pytest.importorskip("torch")

from evsys_sdk.data_types import HarborTask, InProcessVerifier
from evsys_sdk.training import harbor_eval as he
from evsys_sdk.training.trajectory import Trajectory, TrajectoryGroup, Turn


def _task(task_id, expected="42"):
    return HarborTask(
        task_id=task_id, instruction=f"solve {task_id}",
        verifier=InProcessVerifier(fn_name="exact_match", expected=expected),
        metadata={"tags": ["t"]},
    )


def _group(rewards):
    return TrajectoryGroup(trajectories=[
        Trajectory(turns=[Turn(prompt_tokens=[1], completion_tokens=[2, 3], logprobs=[-0.1, -0.2])], reward=r)
        for r in rewards
    ])


def _group_with_usage(rewards, *, latency, prompt_tokens, completion_tokens, cost_usd):
    return TrajectoryGroup(trajectories=[
        Trajectory(
            turns=[Turn(prompt_tokens=[1], completion_tokens=[2, 3], logprobs=[-0.1, -0.2])],
            reward=r,
            metadata={"usage": {
                "latency_s": latency, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "cost_usd": cost_usd,
                "cached_tokens": None,
            }},
        )
        for r in rewards
    ])


# --- metrics ---------------------------------------------------------------


def test_eval_metrics_mean_and_pass_rate():
    groups = [_group([1.0]), _group([0.0]), _group([1.0])]
    m = he.eval_metrics(groups)
    assert m["n_tasks"] == 3.0
    assert m["mean_reward"] == pytest.approx(2 / 3)
    assert m["pass_rate"] == pytest.approx(2 / 3)


def test_eval_metrics_averages_samples_within_task():
    groups = [_group([1.0, 0.0])]   # one task, two samples
    m = he.eval_metrics(groups)
    assert m["n_tasks"] == 1.0
    assert m["mean_reward"] == pytest.approx(0.5)
    assert m["pass_rate"] == pytest.approx(0.5)   # 1 of 2 samples passed


def test_eval_metrics_empty():
    assert he.eval_metrics([]) == {"mean_reward": 0.0, "pass_rate": 0.0, "n_tasks": 0.0}


def test_eval_metrics_honors_declared_metric_names():
    # 4 tasks × 3 samples: any-of-3 solves 3/4 (pass@3), all-of-3 solves 1/4 (pass^3).
    groups = [
        _group([1.0, 1.0, 0.0]),
        _group([1.0, 0.0, 0.0]),
        _group([0.0, 0.0, 0.0]),
        _group([1.0, 1.0, 1.0]),
    ]
    m = he.eval_metrics(groups, metrics=["pass@3", "pass^3", "avg"])
    assert m["pass@3"] == pytest.approx(0.75)
    assert m["pass^3"] == pytest.approx(0.25)
    assert m["avg"] == pytest.approx(0.5)
    assert m["n_tasks"] == 4.0
    # Declared list replaces the default mean_reward/pass_rate keys.
    assert "mean_reward" not in m


def test_eval_metrics_includes_time_tokens_cost_when_present():
    groups = [
        _group_with_usage([1.0], latency=2.0, prompt_tokens=10, completion_tokens=5, cost_usd=0.01),
        _group_with_usage([0.0], latency=4.0, prompt_tokens=20, completion_tokens=5, cost_usd=0.03),
    ]
    m = he.eval_metrics(groups)
    assert m["time_per_task"] == pytest.approx(3.0)       # mean(2, 4)
    assert m["tokens_per_task"] == pytest.approx(20.0)    # mean(10+5, 20+5)
    assert m["cost_per_task"] == pytest.approx(0.02)      # mean(0.01, 0.03)


def test_eval_metrics_omits_cost_when_no_api_price():
    # tinker-style: latency + tokens present, cost is None → cost_per_task omitted.
    groups = [_group_with_usage([1.0], latency=1.0, prompt_tokens=8, completion_tokens=2, cost_usd=None)]
    m = he.eval_metrics(groups)
    assert "cost_per_task" not in m
    assert m["time_per_task"] == pytest.approx(1.0)
    assert m["tokens_per_task"] == pytest.approx(10.0)


def test_eval_metrics_no_usage_keeps_legacy_shape():
    # Trajectories with no usage metadata → only the reward stats, no econ keys.
    m = he.eval_metrics([_group([1.0])])
    assert set(m) == {"mean_reward", "pass_rate", "n_tasks"}


# --- predictions -----------------------------------------------------------


def test_eval_predictions_rows():
    tasks = [_task("a", expected="42"), _task("b", expected="7")]
    groups = [_group([1.0]), _group([0.0])]
    rows = he.eval_predictions(tasks, groups, eval_id="ev1", step=5)
    assert len(rows) == 2
    assert rows[0]["kind"] == "eval"
    assert rows[0]["eval_id"] == "ev1"
    assert rows[0]["step"] == 5
    assert rows[0]["task_id"] == "a"
    assert rows[0]["expected"] == "42"
    assert rows[0]["reward"] == 1.0
    assert rows[0]["completion_token_ids"] == [2, 3]


def test_eval_predictions_store_time_and_tokens_in_metadata():
    # Per-task time + tokens are stored in metadata (the JSON column both stores
    # persist), not top-level columns the remote backend would drop.
    tasks = [_task("a")]
    groups = [_group_with_usage([1.0], latency=2.5, prompt_tokens=12, completion_tokens=3, cost_usd=0.02)]
    rows = he.eval_predictions(tasks, groups, eval_id="ev", step=None)
    md = rows[0]["metadata"]
    assert md["latency_s"] == pytest.approx(2.5)
    assert md["prompt_tokens"] == 12
    assert md["completion_tokens"] == 3
    assert md["tags"] == ["t"]                 # task metadata preserved
    assert "cost_usd" not in rows[0] and "latency_s" not in rows[0]  # not top-level


# --- upload (eval only) ----------------------------------------------------


class _DashboardLike:
    def __init__(self):
        self.calls = []

    def log_predictions(self, run_id, predictions):
        self.calls.append((run_id, predictions))


class _StoreLike:
    def __init__(self):
        self.rows = []

    def add_prediction(self, **kw):
        self.rows.append(kw)


def test_upload_uses_log_predictions_when_available():
    store = _DashboardLike()
    rows = [{"kind": "eval", "task_id": "a", "reward": 1.0}]
    he.upload_eval_rollouts(store, "run1", rows)
    assert store.calls == [("run1", rows)]


def test_upload_falls_back_to_add_prediction():
    store = _StoreLike()
    rows = [{"kind": "eval", "task_id": "a", "reward": 1.0, "completion_token_ids": [2, 3]}]
    he.upload_eval_rollouts(store, "run1", rows)
    assert len(store.rows) == 1
    assert store.rows[0]["task_id"] == "a"
    assert store.rows[0]["metadata"]["completion_token_ids"] == [2, 3]


def test_upload_add_prediction_preserves_time_and_token_metadata():
    # eval_predictions already put time+tokens in metadata; the add_prediction
    # fallback must round-trip it (and fold in the token ids).
    store = _StoreLike()
    rows = [{"kind": "eval", "task_id": "a", "reward": 1.0,
             "completion_token_ids": [2, 3],
             "metadata": {"latency_s": 2.5, "prompt_tokens": 12, "completion_tokens": 3}}]
    he.upload_eval_rollouts(store, "run1", rows)
    md = store.rows[0]["metadata"]
    assert md["latency_s"] == 2.5
    assert md["prompt_tokens"] == 12
    assert md["completion_tokens"] == 3
    assert md["completion_token_ids"] == [2, 3]


def test_upload_noop_without_store_or_rows():
    he.upload_eval_rollouts(None, "run1", [{"kind": "eval"}])   # no store → no-op
    he.upload_eval_rollouts(_DashboardLike(), "", [{"kind": "eval"}])  # no run_id
    he.upload_eval_rollouts(_DashboardLike(), "run1", [])  # no rows

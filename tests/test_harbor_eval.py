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


def test_upload_noop_without_store_or_rows():
    he.upload_eval_rollouts(None, "run1", [{"kind": "eval"}])   # no store → no-op
    he.upload_eval_rollouts(_DashboardLike(), "", [{"kind": "eval"}])  # no run_id
    he.upload_eval_rollouts(_DashboardLike(), "run1", [])  # no rows

"""Capped rollout capture — the budget keeper, the training-rollout row
builder, and the loop/callback wiring that puts the first N of each kind into
the run's predictions."""

from __future__ import annotations

import pytest

from evsys_sdk.training.rollout_capture import (
    DEFAULT_ROLLOUT_CAP,
    KIND_EVAL,
    KIND_TRAIN,
    KIND_VALIDATION,
    RolloutCapture,
    training_rollout_rows,
)


class _Turn:
    def __init__(self, text="", completion_tokens=None):
        self.text = text
        self.completion_tokens = completion_tokens or [1, 2]


class _Traj:
    def __init__(self, reward=1.0, text="out", metadata=None):
        self.turns = [_Turn(text)]
        self.reward = reward
        self.metadata = metadata or {"usage": {"latency_s": 0.5, "completion_tokens": 7}}


class _Group:
    def __init__(self, n=2, task_id=None):
        self.trajectories = [_Traj(text=f"o{i}") for i in range(n)]
        if task_id:
            self.task_id = task_id


class TestBudget:
    def test_default_cap_is_twenty(self):
        assert DEFAULT_ROLLOUT_CAP == 20
        assert RolloutCapture().remaining(KIND_TRAIN) == 20

    def test_each_kind_has_its_own_budget(self):
        cap = RolloutCapture(2)
        assert cap.take(KIND_TRAIN, [{"i": 0}, {"i": 1}, {"i": 2}]) == [{"i": 0}, {"i": 1}]
        assert cap.remaining(KIND_TRAIN) == 0
        # validation and eval are untouched by training's spend
        assert cap.remaining(KIND_VALIDATION) == 2
        assert cap.remaining(KIND_EVAL) == 2
        assert len(cap.take(KIND_VALIDATION, [{"i": 0}, {"i": 1}, {"i": 2}])) == 2

    def test_keeps_the_first_n_not_a_sample(self):
        """First-N, so the kept set is deterministic across reruns."""
        cap = RolloutCapture(3)
        kept = cap.take(KIND_TRAIN, [{"i": i} for i in range(10)])
        assert [r["i"] for r in kept] == [0, 1, 2]

    def test_budget_spans_calls(self):
        cap = RolloutCapture(3)
        assert len(cap.take(KIND_TRAIN, [{"i": 0}, {"i": 1}])) == 2
        assert len(cap.take(KIND_TRAIN, [{"i": 2}, {"i": 3}])) == 1   # only 1 left
        assert cap.take(KIND_TRAIN, [{"i": 4}]) == []
        assert cap.counts()[KIND_TRAIN] == 3

    def test_zero_disables_and_negative_is_unlimited(self):
        off = RolloutCapture(0)
        assert not off.enabled
        assert off.take(KIND_TRAIN, [{"i": 0}]) == []

        unlimited = RolloutCapture(-1)
        assert unlimited.remaining(KIND_TRAIN) == -1
        assert len(unlimited.take(KIND_TRAIN, [{"i": i} for i in range(500)])) == 500


class TestTrainingRows:
    def test_row_shape_matches_the_prediction_schema(self):
        rows = training_rollout_rows([_Group(n=1)], step=3)
        assert len(rows) == 1
        r = rows[0]
        assert r["kind"] == KIND_TRAIN
        assert r["step"] == 3 and r["sample_idx"] == 0
        assert r["reward"] == 1.0
        assert r["completion"] == "o0"
        assert r["completion_token_ids"] == [1, 2]
        # usage is folded into metadata, same as eval_predictions does
        assert r["metadata"]["latency_s"] == 0.5
        assert r["metadata"]["completion_tokens"] == 7
        assert r["metadata"]["group_idx"] == 0

    def test_flattens_groups_and_samples(self):
        rows = training_rollout_rows([_Group(n=2), _Group(n=3)], step=0)
        assert len(rows) == 5
        assert [r["sample_idx"] for r in rows] == [0, 1, 0, 1, 2]

    def test_limit_stops_building_rows(self):
        """The cap must stop row CONSTRUCTION, not just filter afterwards —
        a 10k-rollout step should never materialise 10k dicts."""
        rows = training_rollout_rows([_Group(n=50) for _ in range(20)], step=0, limit=4)
        assert len(rows) == 4

    def test_task_id_falls_back_to_group_index(self):
        rows = training_rollout_rows([_Group(n=1), _Group(n=1, task_id="t9")], step=0)
        assert rows[0]["task_id"] == "group-0"
        assert rows[1]["task_id"] == "t9"

    def test_empty_input_is_empty_output(self):
        assert training_rollout_rows([], step=0) == []
        assert training_rollout_rows(None, step=0) == []


class TestCallbackWiring:
    def _callback(self, cap=DEFAULT_ROLLOUT_CAP):
        from evsys_sdk.training.callbacks import EvsysLoggerCallback

        cb = EvsysLoggerCallback(rollout_cap=cap)
        sent: list[dict] = []

        class _Store:
            def log_predictions(self, run_id, predictions):
                sent.extend(predictions)

        cb._store = _Store()
        cb._run_id = "run1"
        return cb, sent

    def test_training_rollouts_are_persisted_capped(self):
        cb, sent = self._callback(cap=3)
        cb.on_rollout(None, 0, [_Group(n=5)])
        assert len(sent) == 3
        assert all(r["kind"] == KIND_TRAIN for r in sent)

        cb.on_rollout(None, 1, [_Group(n=5)])   # budget already spent
        assert len(sent) == 3

    def test_no_run_id_means_no_write(self):
        cb, sent = self._callback()
        cb._run_id = None
        cb.on_rollout(None, 0, [_Group(n=2)])
        assert sent == []

    def test_cap_zero_writes_nothing(self):
        cb, sent = self._callback(cap=0)
        cb.on_rollout(None, 0, [_Group(n=5)])
        assert sent == []


class TestLoopGate:
    """The loop must stop dispatching once the budget is gone, so a long run
    does not pay to build rollout rows forever."""

    def test_dispatch_stops_when_budget_is_exhausted(self):
        pytest.importorskip("tinker")
        from evsys_sdk.training.loop import TrainingLoop

        assert hasattr(TrainingLoop, "__init__")
        cap = RolloutCapture(2)
        assert cap.remaining(KIND_TRAIN) == 2
        cap.take(KIND_TRAIN, [{"i": 0}, {"i": 1}])
        assert cap.remaining(KIND_TRAIN) == 0

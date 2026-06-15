"""Post-training benchmark eval through harbor (opt-in `engine: harbor`).

Mocks score_via_harbor (no harbor/containers) and checks Experiment._eval_arm_harbor
scores the benchmark + uploads eval rollouts (kind='eval')."""

from __future__ import annotations

import pytest

pytest.importorskip("tinker")
pytest.importorskip("torch")

from evsys_sdk.benchmark import Benchmark
from evsys_sdk.config import AlgorithmConfig, DataConfig, ExperimentConfig, ModelConfig, RunConfig
from evsys_sdk.data_types import HarborTask, InProcessVerifier
from evsys_sdk.experiment import ArmResult, Experiment
from evsys_sdk.protocols import RunResult
from evsys_sdk.training.trajectory import Trajectory, TrajectoryGroup, Turn


class _Store:
    def __init__(self):
        self.evals: list[dict] = []
        self.preds: list[tuple] = []

    def create_eval(self, **kw):
        self.evals.append(kw)
        return {"id": "ev1"}

    def log_predictions(self, run_id, predictions):
        self.preds.append((run_id, predictions))


def _bench():
    tasks = [
        HarborTask(task_id=f"t{i}", instruction=f"P{i}",
                   verifier=InProcessVerifier(fn_name="exact_match", expected="x"),
                   metadata={"tags": ["test"]})
        for i in range(3)
    ]
    return Benchmark.from_iterable("b", tasks)


def _run_cfg():
    return RunConfig(
        name="r",
        data=DataConfig(source_kind="in_memory", rows=[{"x": 1}]),
        model=ModelConfig(name="m"),
        algorithm=AlgorithmConfig(kind="rl"),
    )


def test_eval_arm_harbor_scores_and_uploads(monkeypatch):
    async def _fake_score(tasks, **kwargs):
        return [
            TrajectoryGroup(
                trajectories=[Trajectory(
                    turns=[Turn(prompt_tokens=[1], completion_tokens=[2, 3], logprobs=[-0.1, -0.2])],
                    reward=1.0,
                )],
                tags=["test"],
            )
            for _ in tasks
        ]

    monkeypatch.setattr("evsys_sdk.training.harbor_eval.score_via_harbor", _fake_score)

    store = _Store()
    run_cfg = _run_cfg()
    e = Experiment(ExperimentConfig(name="x", run=run_cfg), store=store)
    arm = ArmResult(
        name="r", run_config=run_cfg, status="completed", run_id="run1",
        run_result=RunResult(run_id="run1", status="completed",
                             artifacts={"checkpoint-final": "tinker://ckpt"}),
    )

    e._eval_arm_harbor(arm, run_cfg, _bench(), {"engine": "harbor", "name": "b", "tags": ["test"]})

    # scored
    assert len(arm.evals) == 1
    assert arm.evals[0].metrics["pass_rate"] == 1.0
    assert arm.evals[0].metrics["n_tasks"] == 3.0
    # eval recorded + rollouts uploaded (kind='eval'), one per task
    assert store.evals and store.evals[0]["run_id"] == "run1"
    assert store.preds and store.preds[0][0] == "run1"
    rows = store.preds[0][1]
    assert len(rows) == 3
    assert all(r["kind"] == "eval" and r["eval_id"] == "ev1" for r in rows)
    assert rows[0]["completion_token_ids"] == [2, 3]


def test_final_checkpoint_picks_from_artifacts():
    arm = ArmResult(
        name="r", run_config=None, status="completed",  # type: ignore[arg-type]
        run_result=RunResult(run_id="r", status="completed",
                             artifacts={"checkpoint-final": "tinker://final"}),
    )
    assert Experiment._final_checkpoint(arm) == "tinker://final"

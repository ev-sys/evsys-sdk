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


def test_eval_arm_harbor_api_model_uses_litellm_and_per_model_eval(monkeypatch):
    # api_model → score that closed model via litellm (not the checkpoint),
    # recorded as its own per-model eval.
    captured: dict = {}

    async def _fake_score(tasks, **kwargs):
        captured.update(kwargs)
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

    run_cfg = _run_cfg()
    e = Experiment(ExperimentConfig(name="x", run=run_cfg), store=_Store())
    arm = ArmResult(
        name="r", run_config=run_cfg, status="completed", run_id="run1",
        run_result=RunResult(run_id="run1", status="completed",
                             artifacts={"checkpoint-final": "tinker://ckpt"}),
    )

    e._eval_arm_harbor(
        arm, run_cfg, _bench(), {"engine": "harbor", "name": "b", "tags": ["test"]},
        api_model="anthropic/claude-opus-4-1",
    )

    assert captured["model_client"] == "litellm"
    assert captured["model_name"] == "anthropic/claude-opus-4-1"
    assert captured["model_path"] is None              # API model, not the checkpoint
    # recorded as a distinct per-model eval (name + tag carry the model)
    ev = arm.evals[0]
    assert ev.name == "b@anthropic/claude-opus-4-1"
    assert "anthropic/claude-opus-4-1" in ev.tags


def test_eval_arm_harbor_persists_rollouts_under_run_dir(monkeypatch, tmp_path):
    # Eval rollouts must land under the run's output dir (harbor_eval/<bench>),
    # not an ephemeral tempdir — so they survive the run like training/val do.
    captured: dict = {}

    async def _fake_score(tasks, **kwargs):
        captured["workspace_dir"] = kwargs.get("workspace_dir")
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

    run_cfg = _run_cfg()
    cfg = ExperimentConfig(name="x", run=run_cfg, output_dir=str(tmp_path))
    e = Experiment(cfg, store=_Store())
    arm = ArmResult(
        name="r", run_config=run_cfg, status="completed", run_id="run1",
        run_result=RunResult(run_id="run1", status="completed",
                             artifacts={"checkpoint-final": "tinker://ckpt"}),
    )

    e._eval_arm_harbor(arm, run_cfg, _bench(), {"engine": "harbor", "name": "b", "tags": ["test"]})

    ws = captured["workspace_dir"]
    assert ws == tmp_path / "r" / "harbor_eval" / "b"   # persisted under the run dir
    assert ws.exists()                                   # created, not a vanished tempdir
    assert str(ws).startswith(str(tmp_path))             # never a system tempdir


def test_final_checkpoint_picks_from_artifacts():
    arm = ArmResult(
        name="r", run_config=None, status="completed",  # type: ignore[arg-type]
        run_result=RunResult(run_id="r", status="completed",
                             artifacts={"checkpoint-final": "tinker://final"}),
    )
    assert Experiment._final_checkpoint(arm) == "tinker://final"

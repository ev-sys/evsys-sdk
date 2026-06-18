"""In-loop validation through harbor (BenchmarkEvaluator engine='harbor').

Mocks run_harbor_rollouts so the evaluator's harbor branch is exercised without
a harbor install."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("tinker")
pytest.importorskip("torch")

from evsys_sdk.benchmark import Benchmark
from evsys_sdk.data_types import HarborTask, InProcessVerifier
from evsys_sdk.training.evaluators import BenchmarkEvaluator
from evsys_sdk.training.trajectory import Trajectory, TrajectoryGroup, Turn


def _bench():
    tasks = [
        HarborTask(task_id=f"t{i}", instruction=f"P{i}",
                   verifier=InProcessVerifier(fn_name="exact_match", expected="x"))
        for i in range(2)
    ]
    return Benchmark.from_iterable("val", tasks)


def test_harbor_engine_scores_via_harbor(monkeypatch):
    async def _fake_score(tasks, **kwargs):
        return [
            TrajectoryGroup(trajectories=[Trajectory(
                turns=[Turn(prompt_tokens=[1], completion_tokens=[2], logprobs=[-0.1])],
                reward=1.0,
            )])
            for _ in tasks
        ]

    monkeypatch.setattr("evsys_sdk.training.harbor_engine.run_harbor_rollouts", _fake_score)

    ev = BenchmarkEvaluator(
        name="val", benchmark=_bench(), tokenizer=None,
        engine="harbor", model_name="m",
    )
    metrics = asyncio.run(ev.evaluate(object(), model_path="tinker://ckpt", step=10))
    assert metrics["pass_rate"] == 1.0
    assert metrics["n_tasks"] == 2.0


def test_harbor_validation_uploads_eval_per_step(monkeypatch):
    """store + run_id present → one create_eval per validation (tagged with the
    training step) and one eval prediction per task."""
    async def _fake_score(tasks, **kwargs):
        return [
            TrajectoryGroup(trajectories=[Trajectory(
                turns=[Turn(prompt_tokens=[1], completion_tokens=[2], logprobs=[-0.1])],
                reward=1.0,
            )])
            for _ in tasks
        ]

    monkeypatch.setattr("evsys_sdk.training.harbor_engine.run_harbor_rollouts", _fake_score)

    class _Store:
        def __init__(self):
            self.evals: list[dict] = []
            self.preds: list[dict] = []

        def create_eval(self, **kw):
            self.evals.append(kw)
            return {"id": f"eval_{kw['step']}"}

        def add_prediction(self, **kw):
            self.preds.append(kw)

    store = _Store()
    ev = BenchmarkEvaluator(
        name="val", benchmark=_bench(), tokenizer=None,
        engine="harbor", model_name="m",
        store=store, run_id="run123", benchmark_id="bench9",
    )

    # Two validations at different training steps.
    for step in (5, 10):
        asyncio.run(ev.evaluate(object(), model_path="tinker://ckpt", step=step))

    # One eval per validation, each tagged with its step + run + benchmark.
    assert [e["step"] for e in store.evals] == [5, 10]
    assert all(e["run_id"] == "run123" for e in store.evals)
    assert all(e["benchmark_id"] == "bench9" for e in store.evals)
    # Two tasks × two validations = 4 eval predictions, eval_id threaded through.
    assert len(store.preds) == 4
    assert {p["eval_id"] for p in store.preds} == {"eval_5", "eval_10"}
    assert all(p["kind"] == "eval" for p in store.preds)


def test_harbor_validation_skips_upload_without_eval_id(monkeypatch):
    """If create_eval yields no id, predictions are NOT uploaded — orphan rows
    couldn't be told apart from other evals on the run."""
    async def _fake_score(tasks, **kwargs):
        return [TrajectoryGroup(trajectories=[Trajectory(
            turns=[Turn(prompt_tokens=[1], completion_tokens=[2], logprobs=[-0.1])],
            reward=1.0,
        )]) for _ in tasks]

    monkeypatch.setattr("evsys_sdk.training.harbor_engine.run_harbor_rollouts", _fake_score)

    class _Store:
        def __init__(self):
            self.preds: list[dict] = []

        def create_eval(self, **kw):
            return {}  # no id

        def add_prediction(self, **kw):
            self.preds.append(kw)

    store = _Store()
    ev = BenchmarkEvaluator(
        name="val", benchmark=_bench(), tokenizer=None,
        engine="harbor", model_name="m", store=store, run_id="r1",
    )
    asyncio.run(ev.evaluate(object(), model_path="tinker://ckpt", step=5))
    assert store.preds == []


def test_non_harbor_engine_uses_sampler_path(monkeypatch):
    """Without engine='harbor', the live-sampler path is used (no harbor call)."""
    called = {"harbor": False}

    async def _boom(*a, **k):
        called["harbor"] = True
        return []

    monkeypatch.setattr("evsys_sdk.training.harbor_engine.run_harbor_rollouts", _boom)

    # benchmark.score is what the sampler path calls — stub it via a fake bench.
    class _Bench:
        tasks: list = []

        def score(self, client, **kw):
            from evsys_sdk.benchmark import BenchmarkScore
            return BenchmarkScore(metrics={"pass_rate": 0.5}, per_task=[])

    class _Sampler:
        async def sample_async(self, **kw):
            class _R:
                sequences = []
            return _R()

    ev = BenchmarkEvaluator(name="val", benchmark=_Bench(), tokenizer=object(), engine="")
    metrics = asyncio.run(ev.evaluate(_Sampler(), model_path="tinker://ckpt"))
    assert called["harbor"] is False
    assert metrics["pass_rate"] == 0.5

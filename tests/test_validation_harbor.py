"""In-loop validation through harbor (BenchmarkEvaluator engine='harbor').

Mocks score_via_harbor so the evaluator's harbor branch is exercised without
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

    monkeypatch.setattr("evsys_sdk.training.harbor_eval.score_via_harbor", _fake_score)

    ev = BenchmarkEvaluator(
        name="val", benchmark=_bench(), tokenizer=None,
        engine="harbor", model_name="m",
    )
    metrics = asyncio.run(ev.evaluate(object(), model_path="tinker://ckpt", step=10))
    assert metrics["pass_rate"] == 1.0
    assert metrics["n_tasks"] == 2.0


def test_non_harbor_engine_uses_sampler_path(monkeypatch):
    """Without engine='harbor', the live-sampler path is used (no harbor call)."""
    called = {"harbor": False}

    async def _boom(*a, **k):
        called["harbor"] = True
        return []

    monkeypatch.setattr("evsys_sdk.training.harbor_eval.score_via_harbor", _boom)

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

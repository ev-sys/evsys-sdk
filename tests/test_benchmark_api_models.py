"""Benchmarks on closed / API models via litellm.

Two harbor-free layers: the agent-selection helper (tinker vs litellm) and the
standalone ``run_benchmark`` (no training). The harbor rollout itself is mocked."""

from __future__ import annotations

import pytest

pytest.importorskip("tinker")
pytest.importorskip("torch")

from evsys_sdk.benchmark import Benchmark
from evsys_sdk.benchmark_run import run_benchmark
from evsys_sdk.data_types import HarborTask, InProcessVerifier
from evsys_sdk.training import harbor_engine as he
from evsys_sdk.training.trajectory import Trajectory, TrajectoryGroup, Turn


# --- agent selection (tinker vs litellm) -----------------------------------


def test_agent_spec_litellm_is_basic_loop_with_litellm_client():
    # No agent copy — the one BasicLoopAgent is parameterized by model_client.
    ip, kw = he._agent_import_and_kwargs(
        "litellm", agent_import_path=None, model_name="anthropic/claude-opus-4-1",
        model_path="tinker://ckpt", renderer_name="qwen3", max_tokens=256,
        temperature=0.0, max_turns=1, system_prompt="sys",
    )
    assert ip.endswith(":BasicLoopAgent")
    assert kw["model_client"] == "litellm"
    assert kw["model_name"] == "anthropic/claude-opus-4-1"
    assert kw["system_prompt"] == "sys"


def test_agent_spec_tinker_is_basic_loop_with_tinker_client():
    ip, kw = he._agent_import_and_kwargs(
        "tinker", agent_import_path=None, model_name="Qwen/Qwen3-4B",
        model_path="tinker://ckpt", renderer_name="qwen3", max_tokens=256,
        temperature=0.0, max_turns=1, system_prompt=None,
    )
    assert ip.endswith(":BasicLoopAgent")
    assert kw["model_client"] == "tinker"
    assert kw["model_path"] == "tinker://ckpt"
    assert kw["renderer_name"] == "qwen3"


def test_benchmark_models_parsing():
    from evsys_sdk.experiment import _benchmark_models
    assert _benchmark_models({}) == []
    assert _benchmark_models({"model": "anthropic/claude-opus-4-1"}) == ["anthropic/claude-opus-4-1"]
    assert _benchmark_models({"models": ["a", "b"]}) == ["a", "b"]
    # explicit empty models → checkpoint-only (no API models)
    assert _benchmark_models({"models": []}) == []


def test_agent_spec_explicit_import_path_wins_with_no_kwargs():
    ip, kw = he._agent_import_and_kwargs(
        "litellm", agent_import_path="my.module:CustomAgent", model_name="m",
        model_path=None, renderer_name=None, max_tokens=256, temperature=0.0,
        max_turns=1, system_prompt=None,
    )
    assert ip == "my.module:CustomAgent"
    assert kw == {}


# --- run_benchmark (standalone, no training) -------------------------------


def _bench():
    tasks = [
        HarborTask(task_id=f"t{i}", instruction=f"P{i}",
                   verifier=InProcessVerifier(fn_name="exact_match", expected="x"),
                   metadata={"tags": ["b"]})
        for i in range(2)
    ]
    return Benchmark.from_iterable("apibench", tasks)


def _usage_group(n, **usage):
    return [
        TrajectoryGroup(
            trajectories=[Trajectory(
                turns=[Turn(prompt_tokens=[1], completion_tokens=[2, 3], logprobs=[-0.1, -0.2])],
                reward=1.0, metadata={"usage": usage} if usage else {},
            )],
            tags=["b"],
        )
        for _ in range(n)
    ]


def test_run_benchmark_routes_to_litellm_and_returns_metrics(monkeypatch):
    captured: dict = {}

    async def _fake_score(tasks, **kwargs):
        captured.update(kwargs)
        captured["n_tasks"] = len(tasks)
        return _usage_group(len(tasks), latency_s=1.0, prompt_tokens=5,
                            completion_tokens=2, cost_usd=0.01)

    monkeypatch.setattr("evsys_sdk.training.harbor_engine.run_harbor_rollouts", _fake_score)
    metrics = run_benchmark(_bench(), model="anthropic/claude-opus-4-1", max_tokens=128)

    assert captured["model_client"] == "litellm"
    assert captured["model_name"] == "anthropic/claude-opus-4-1"
    assert captured["model_path"] is None            # API model is the policy — no checkpoint
    assert captured["n_tasks"] == 2
    assert metrics["pass_rate"] == 1.0
    assert metrics["cost_per_task"] == pytest.approx(0.01)
    assert metrics["tokens_per_task"] == pytest.approx(7.0)


def test_run_benchmark_uploads_when_store_and_run_id(monkeypatch):
    async def _fake_score(tasks, **kwargs):
        return _usage_group(len(tasks))

    monkeypatch.setattr("evsys_sdk.training.harbor_engine.run_harbor_rollouts", _fake_score)

    uploaded: list = []

    class _Store:
        def log_predictions(self, run_id, preds):
            uploaded.append((run_id, preds))

    run_benchmark(_bench(), model="openai/gpt-4o", store=_Store(), run_id="run1", limit=1)
    assert uploaded and uploaded[0][0] == "run1"
    assert len(uploaded[0][1]) == 1                  # limit=1 → one task scored

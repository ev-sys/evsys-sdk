"""End-to-end test of `evsys_sdk.algorithms.native_rl.NativeRL`.

Like the SFT/SDFT composer tests: stub TinkerBackend.create + the sampler
factory so the full RL pipeline (rows → builders → rollout → advantage →
IS-loss CE-shape Datums → loop) runs without a real tinker session.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import evsys_sdk.algorithms.native_rl as native_rl_module
from evsys_sdk.algorithms.native_rl import NativeRL, NativeRLConfig
from evsys_sdk.protocols import RunResult
from evsys_sdk.registry import get_algorithm
from evsys_sdk.training import MockBackend


class _StubLogStore:
    def __init__(self):
        self.hyperparams: dict | None = None
        self.metric_rows: list[dict] = []
        self.artifacts: list[tuple[str, str, str]] = []

    def log_hyperparams(self, hp):
        self.hyperparams = dict(hp)

    def log_metrics(self, metrics, *, step, split="train"):
        self.metric_rows.append({"step": step, "split": split, "metrics": dict(metrics)})

    def log_artifact(self, key, value, *, kind):
        self.artifacts.append((key, value, kind))


class _StubTokenizer:
    def apply_chat_template(self, messages, *, tokenize=True,
                            add_generation_prompt=False, **extra):
        parts = [f"<{m['role'][0]}>{m['content']}" for m in messages]
        text = "$".join(parts)
        if add_generation_prompt:
            text += "?"
        return text

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "T" + "_".join(str(t) for t in tokens)


class _CannedRolloutSampler:
    """Sampler that emits a canned `tokens` list per call."""

    def __init__(self, tokens, name="rl-mock"):
        self.tokens = tokens
        self.name = name

    async def sample_async(self, **kwargs):
        @dataclass
        class _Seq:
            tokens: list[int]
            logprobs: list[float]
        @dataclass
        class _Resp:
            sequences: list[Any]
        return _Resp(sequences=[
            _Seq(tokens=list(self.tokens), logprobs=[-0.5] * len(self.tokens)),
        ])


@pytest.fixture
def patched_tinker_backend(monkeypatch):
    backend = MockBackend(tokenizer=_StubTokenizer())
    backend._sampler_factory = lambda name: _CannedRolloutSampler(  # type: ignore[assignment]
        tokens=[200, 201, 202], name=name,
    )

    async def _factory(**kwargs):
        backend._model_name = kwargs.get("model_name")  # type: ignore[attr-defined]
        return backend

    monkeypatch.setattr(native_rl_module.TinkerBackend, "create", _factory)
    return backend


@pytest.fixture
def ctx(tmp_path: Path):
    rows = [
        {"prompt": f"P{i}", "expected": "T200_201_202", "tags": ["foo"]}
        for i in range(20)
    ]

    class _Backend:
        name = "tinker"

    class _Ctx:
        def __init__(self):
            self.run_id = "run-rl"
            self.output_dir = tmp_path
            self.log_store = _StubLogStore()
            self.backend = _Backend()
            self.extras = {
                "train_rows": rows,
                "backend_handles": {"model_name": "Qwen/Qwen3-4B"},
                "model_name": "Qwen/Qwen3-4B",
            }

    return _Ctx()


# ---------------------------------------------------------------------------
# Registry + Config
# ---------------------------------------------------------------------------


def test_registered_under_native_rl():
    assert get_algorithm("native_rl") is NativeRL


def test_config_defaults():
    cfg = NativeRLConfig()
    assert cfg.batch_size == 4
    assert cfg.num_samples == 1
    assert cfg.drop_constant_reward is True
    assert cfg.learning_rate == 1.0e-5  # lower default than SFT


def test_config_rejects_unknown_kwarg():
    with pytest.raises(Exception):
        NativeRLConfig(bogus_field=True)


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------


def test_rejects_non_tinker_backend(ctx):
    class _M:
        name = "mock"
    ctx.backend = _M()
    with pytest.raises(RuntimeError, match="backend=tinker"):
        NativeRL(max_steps=2, batch_size=4, verifier_name="exact_match").train(ctx)


def test_rejects_missing_train_rows_and_no_env_builders(ctx, patched_tinker_backend):
    ctx.extras["train_rows"] = []
    ctx.extras.pop("env_builders", None)
    with pytest.raises(RuntimeError, match="env_builders.*train_rows|train_rows.*env_builders"):
        NativeRL(max_steps=2, batch_size=4, verifier_name="exact_match").train(ctx)


def test_rejects_unknown_verifier(ctx, patched_tinker_backend):
    with pytest.raises(RuntimeError, match="unknown verifier_name"):
        NativeRL(max_steps=2, batch_size=4, verifier_name="not_real_verifier").train(ctx)


def test_rejects_missing_verifier_when_rows_path_chosen(ctx, patched_tinker_backend):
    with pytest.raises(RuntimeError, match="verifier_name unset"):
        NativeRL(max_steps=2, batch_size=4).train(ctx)


# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------


def test_train_runs_end_to_end(patched_tinker_backend, ctx):
    """drop_constant_reward=False so the rewards (all 1.0 from the canned
    sampler) don't get filtered out — keeps the test's batch non-empty."""
    algo = NativeRL(
        max_steps=2, batch_size=4, num_samples=1,
        verifier_name="exact_match", drop_constant_reward=False,
    )
    result = algo.train(ctx)
    assert isinstance(result, RunResult)
    assert result.status == "completed"
    # The loop ran 2 train steps.
    assert len(patched_tinker_backend.fb_calls) == 2
    assert len(patched_tinker_backend.optim_calls) == 2
    # IS loss name was passed through.
    assert patched_tinker_backend.fb_calls[0]["loss_fn"] == "importance_sampling"
    # checkpoint artifacts surface
    assert result.artifacts.get("checkpoint-final", "").startswith("mock://sampler/")


def test_train_logs_reward_metrics_per_step(patched_tinker_backend, ctx):
    algo = NativeRL(
        max_steps=2, batch_size=4, num_samples=1,
        verifier_name="exact_match", drop_constant_reward=False,
    )
    algo.train(ctx)
    train_rows = [r for r in ctx.log_store.metric_rows if r["split"] == "train"]
    assert len(train_rows) == 2
    for r in train_rows:
        assert "reward/mean" in r["metrics"]
        assert "reward/n_trajectories" in r["metrics"]
        assert "progress/step" in r["metrics"]


def test_train_logs_hyperparams(patched_tinker_backend, ctx):
    NativeRL(
        max_steps=2, batch_size=4, num_samples=1,
        verifier_name="exact_match", drop_constant_reward=False,
    ).train(ctx)
    hp = ctx.log_store.hyperparams
    assert hp is not None
    assert hp["algorithm"] == "native_rl"
    assert hp["n_builders"] == 20
    assert hp["total_steps"] == 2

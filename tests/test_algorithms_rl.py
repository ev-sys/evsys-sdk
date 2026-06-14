"""End-to-end test of `evsys_sdk.algorithms.rl.RL`.

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

pytest.importorskip("tinker")  # optional dep; not installed in base CI
pytest.importorskip("torch")

import evsys_sdk.algorithms.rl as rl_module
from evsys_sdk.algorithms.rl import RL, RLConfig
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

    monkeypatch.setattr(rl_module.TinkerBackend, "create", _factory)
    return backend


@pytest.fixture
def ctx(tmp_path: Path):
    # Standardized HarborTask shape: instruction + per-row in_process verifier.
    rows = [
        {
            "task_id": f"t{i}",
            "instruction": f"P{i}",
            "verifier": {
                "kind": "in_process",
                "fn_name": "exact_match",
                "expected": "T200_201_202",
            },
            "metadata": {"tags": ["foo"]},
        }
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


def test_registered_under_rl():
    assert get_algorithm("rl") is RL


def test_config_defaults():
    cfg = RLConfig()
    assert cfg.batch_size == 4
    assert cfg.num_samples == 1
    assert cfg.drop_constant_reward is True
    assert cfg.learning_rate == 1.0e-5  # lower default than SFT


def test_config_rejects_unknown_kwarg():
    with pytest.raises(Exception):
        RLConfig(bogus_field=True)


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------


def test_rejects_non_tinker_backend(ctx):
    class _M:
        name = "mock"
    ctx.backend = _M()
    with pytest.raises(RuntimeError, match="backend=tinker"):
        RL(max_steps=2, batch_size=4, verifier_name="exact_match").train(ctx)


def test_rejects_missing_train_rows_and_no_env_builders(ctx, patched_tinker_backend):
    ctx.extras["train_rows"] = []
    ctx.extras.pop("env_builders", None)
    with pytest.raises(RuntimeError, match="env_builders.*train_rows|train_rows.*env_builders"):
        RL(max_steps=2, batch_size=4, verifier_name="exact_match").train(ctx)


def test_rejects_unknown_verifier(ctx, patched_tinker_backend):
    # The verifier fn_name now rides on each HarborTask row.
    ctx.extras["train_rows"] = [{
        "task_id": "t0", "instruction": "P0",
        "verifier": {"kind": "in_process", "fn_name": "not_real_verifier",
                     "expected": "x"},
    }]
    with pytest.raises(RuntimeError, match="unknown verifier_name"):
        RL(max_steps=2, batch_size=1).train(ctx)


def test_rejects_missing_verifier_when_rows_path_chosen(ctx, patched_tinker_backend):
    # Empty fn_name on the row + no cfg.verifier_name fallback.
    ctx.extras["train_rows"] = [{
        "task_id": "t0", "instruction": "P0",
        "verifier": {"kind": "in_process", "fn_name": "", "expected": "x"},
    }]
    with pytest.raises(RuntimeError, match="no verifier fn_name"):
        RL(max_steps=2, batch_size=1).train(ctx)


def test_rejects_non_harbor_task_rows(ctx, patched_tinker_backend):
    # Wrong format entirely → parse_rows rejects strictly.
    ctx.extras["train_rows"] = [{"prompt": "P0", "expected": "x"}]
    with pytest.raises(ValueError, match="expected 'harbor_task'"):
        RL(max_steps=2, batch_size=1, verifier_name="exact_match").train(ctx)


# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------


def test_train_runs_end_to_end(patched_tinker_backend, ctx):
    """drop_constant_reward=False so the rewards (all 1.0 from the canned
    sampler) don't get filtered out — keeps the test's batch non-empty."""
    algo = RL(
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
    algo = RL(
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
    RL(
        max_steps=2, batch_size=4, num_samples=1,
        verifier_name="exact_match", drop_constant_reward=False,
    ).train(ctx)
    hp = ctx.log_store.hyperparams
    assert hp is not None
    assert hp["algorithm"] == "rl"
    assert hp["n_builders"] == 20
    assert hp["total_steps"] == 2

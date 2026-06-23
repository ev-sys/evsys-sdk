"""End-to-end test of ``evsys_sdk.algorithms.sft.SFT``.

Composer test — proves the full SFT pipeline (tokenize → step build → loop
→ checkpoint → artifact) wires together correctly. Uses a MockBackend so
no real tinker session is needed; the wiring is what matters.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("tinker")  # optional dep; not installed in base CI
pytest.importorskip("torch")

import evsys_sdk.algorithms.sft as sft_module
from evsys_sdk.algorithms.sft import SFT, SFTConfig
from evsys_sdk.protocols import RunContext, RunResult
from evsys_sdk.registry import get_algorithm
from evsys_sdk.training import MockBackend


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _StubLogStore:
    def __init__(self) -> None:
        self.hyperparams: dict[str, Any] | None = None
        self.metric_rows: list[dict[str, Any]] = []
        self.artifacts: list[tuple[str, str, str]] = []

    def log_hyperparams(self, hp: dict[str, Any]) -> None:
        self.hyperparams = dict(hp)

    def log_metrics(self, metrics: dict[str, float], *, step: int,
                    split: str = "train") -> None:
        self.metric_rows.append({"step": step, "split": split, "metrics": dict(metrics)})

    def log_artifact(self, key: str, value: str, *, kind: str) -> None:
        self.artifacts.append((key, value, kind))


class _StubTokenizer:
    """Reads message contents, returns char-code ints. Marks an assistant
    span by adding a "$" sentinel at the end of each turn."""

    def apply_chat_template(self, messages, *, tokenize=True,
                            add_generation_prompt=False, **extra):
        parts = []
        for m in messages:
            parts.append(f"<{m['role'][0]}>{m['content']}$")
        text = "".join(parts)
        if add_generation_prompt:
            text += "?"
        return text

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


@pytest.fixture
def patched_tinker_backend(monkeypatch):
    """Replace TinkerBackend.create with a factory that returns MockBackend.

    This lets us exercise SFT.train end-to-end (the composer plus the
    real TrainingLoop) without needing a tinker server.
    """
    mock = MockBackend(tokenizer=_StubTokenizer())

    async def _factory(**kwargs):
        # Echo the model_name into the mock so log_hyperparams checks make sense.
        mock._model_name = kwargs.get("model_name")  # type: ignore[attr-defined]
        return mock

    monkeypatch.setattr(sft_module.TinkerBackend, "create", _factory)
    return mock


@pytest.fixture
def ctx(tmp_path: Path):
    """Minimal RunContext satisfying SFT's reads from `ctx.extras`."""
    rows = [
        {"messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ]}
        for _ in range(20)   # 20 short rows → 5 batches of 4
    ]
    log = _StubLogStore()

    class _Backend:
        name = "tinker"

    class _Ctx:
        def __init__(self):
            self.run_id = "run-x"
            self.output_dir = tmp_path
            self.log_store = log
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


def test_registered_under_sft():
    assert get_algorithm("sft") is SFT


def test_config_defaults_match_documented_shape():
    cfg = SFTConfig()
    assert cfg.learning_rate == 1.0e-4
    assert cfg.batch_size == 4
    assert cfg.lora_rank == 8
    assert cfg.enable_thinking is None   # tokenizer default
    assert cfg.save_at_fractions == [1.0]


def test_config_rejects_unknown_kwarg():
    with pytest.raises(Exception):
        SFTConfig(boguous_field=True)


# ---------------------------------------------------------------------------
# Backend gate
# ---------------------------------------------------------------------------


def test_train_rejects_non_tinker_backend(ctx):
    class _MockBackend:
        name = "mock"
    ctx.backend = _MockBackend()
    algo = SFT(max_steps=2, batch_size=4)
    with pytest.raises(RuntimeError, match="backend=tinker"):
        algo.train(ctx)


def test_train_rejects_missing_train_rows(ctx):
    ctx.extras["train_rows"] = []
    algo = SFT(max_steps=2, batch_size=4)
    with pytest.raises(RuntimeError, match="train_rows"):
        algo.train(ctx)


def test_train_rejects_missing_model_name(ctx):
    ctx.extras["backend_handles"] = {}
    ctx.extras.pop("model_name", None)
    algo = SFT(max_steps=2, batch_size=4)
    with pytest.raises(RuntimeError, match="model_name"):
        algo.train(ctx)


# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------


def test_train_runs_end_to_end_and_returns_run_result(patched_tinker_backend, ctx):
    """The composer should tokenize, run the loop, save checkpoints, and
    surface a RunResult with run_dir + per-checkpoint URIs in artifacts."""
    algo = SFT(max_steps=3, batch_size=4, save_at_fractions=[1.0])
    result = algo.train(ctx)

    assert isinstance(result, RunResult)
    assert result.status == "completed"
    # run_dir + final-sampler URI both present.
    assert result.artifacts["run_dir"] == str(ctx.output_dir)
    assert any(k.startswith("checkpoint-") for k in result.artifacts)
    assert result.artifacts.get("checkpoint-final", "").startswith("mock://sampler/")

    # MockBackend should have seen 3 forward_backward + optim pairs.
    assert len(patched_tinker_backend.fb_calls) == 3
    assert len(patched_tinker_backend.optim_calls) == 3
    # save_every defaults to total_steps=3 (single fraction), so save fires at
    # step 3 + final save.
    assert "final" in patched_tinker_backend.save_sampler_calls


def test_train_logs_hyperparams_once(patched_tinker_backend, ctx):
    algo = SFT(max_steps=2, batch_size=4)
    algo.train(ctx)
    hp = ctx.log_store.hyperparams
    assert hp is not None
    assert hp["algorithm"] == "sft"
    assert hp["model_name"] == "Qwen/Qwen3-4B"
    assert hp["n_train_rows"] == 20
    assert hp["total_steps"] == 2


def test_train_writes_one_metric_row_per_step(patched_tinker_backend, ctx):
    # Per-step metrics now flow through callbacks (on_step_end), not the loop's
    # log_store (which BaseAlgorithm hands a no-op store so local_logger is the
    # single writer). Capture them with a recording callback.
    from evsys_sdk.training.callbacks import Callback

    rows: list[dict] = []

    class _Rec(Callback):
        def on_step_end(self, state, step_idx, batch, metrics):
            rows.append({"step": step_idx, "split": "train", "metrics": dict(metrics)})

    ctx.extras["callbacks"] = [_Rec()]
    algo = SFT(max_steps=4, batch_size=4)
    algo.train(ctx)
    train_rows = [r for r in rows if r["split"] == "train"]
    assert len(train_rows) == 4
    # No duplicate: the loop is handed a no-op store, so per-step metrics do NOT
    # also land in ctx.log_store — callbacks (local_logger) are the single writer.
    assert ctx.log_store.metric_rows == []
    # hyperparams + checkpoint artifacts still flow to the store.
    assert ctx.log_store.hyperparams is not None
    # Each row carries the always-on loop keys.
    for r in train_rows:
        m = r["metrics"]
        assert "progress/step" in m
        assert "progress/done_frac" in m
        assert "optim/lr" in m


# ---------------------------------------------------------------------------
# Save cadence resolution
# ---------------------------------------------------------------------------


def test_save_at_fractions_resolves_to_gcd_when_reasonable():
    """Marks {25, 50, 75, 100} → gcd 25; total_steps/20 = 5 → 25 >= 5 → 25."""
    algo = SFT(max_steps=100, save_at_fractions=[0.25, 0.5, 0.75, 1.0])
    assert algo._resolve_save_every(total_steps=100) == 25


def test_save_at_fractions_falls_back_when_gcd_too_small():
    """Marks {7, 100} → gcd 1; total_steps/20 = 5 → 1 < 5 → fallback total/10."""
    algo = SFT(max_steps=100, save_at_fractions=[0.07, 1.0])
    assert algo._resolve_save_every(total_steps=100) == 10


def test_explicit_save_every_overrides_fractions():
    algo = SFT(max_steps=100, save_every=42,
                     save_at_fractions=[0.5, 1.0])
    assert algo._resolve_save_every(total_steps=100) == 42


# ---------------------------------------------------------------------------
# Callbacks wired from config (BaseAlgorithm → TrainingLoop)
# ---------------------------------------------------------------------------


def test_callbacks_from_config_fire_during_train(patched_tinker_backend, ctx):
    """A callback declared in `algorithm.params.callbacks` is resolved through
    the registry and attached to the loop, so its hooks fire end-to-end."""
    from evsys_sdk.registry import _callbacks, register_callback
    from evsys_sdk.training import Callback
    from pydantic import BaseModel

    seen: dict[str, int] = {"start": 0, "steps": 0, "end": 0}

    class _RecorderConfig(BaseModel):
        pass

    @register_callback("sft_recorder_test")
    class _Recorder(Callback):
        name = "sft_recorder_test"
        Config = _RecorderConfig

        def on_train_start(self, state):
            seen["start"] += 1
        def on_step_end(self, state, step_idx, batch, metrics):
            seen["steps"] += 1
        def on_train_end(self, state, artifacts):
            seen["end"] += 1

    try:
        algo = SFT(max_steps=3, batch_size=4,
                   callbacks=[{"kind": "sft_recorder_test", "params": {}}])
        algo.train(ctx)
        assert seen == {"start": 1, "steps": 3, "end": 1}
    finally:
        _callbacks.unregister("sft_recorder_test")

"""Tests for ``trajectory_labs.algorithms.tinker_sdft.TinkerSDFT``.

The wrapper drives ``tinker_cookbook.distillation.sdft.main`` over an
``SDFTDataset`` built from ``ctx.extras["train_rows"]``. Tests monkeypatch
``sdft.main`` so we don't stand up a real Tinker session.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from trajectory_labs.algorithms.tinker_sdft import TinkerSDFT, TinkerSDFTConfig
from trajectory_labs.protocols import RunResult
from trajectory_labs.registry import get_algorithm


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _LogStoreStub:
    def __init__(self) -> None:
        self.hyperparams: dict | None = None
        self.artifacts: list[tuple[str, str, str]] = []

    def log_hyperparams(self, hp: dict) -> None:
        self.hyperparams = dict(hp)

    def log_artifact(self, key: str, value: str, *, kind: str) -> None:
        self.artifacts.append((key, value, kind))


class _Backend:
    name = "tinker"


class _Ctx:
    def __init__(self, tmp_path: Path, *, rows: list[dict] | None = None,
                 model_name: str = "Qwen/Qwen3.5-4B",
                 renderer_name: str | None = "qwen3_5") -> None:
        self.run_id = "test-run"
        self.output_dir = str(tmp_path)
        self.backend = _Backend()
        self.log_store = _LogStoreStub()
        self.extras = {
            "train_rows": rows or [],
            "backend_handles": {
                "model_name": model_name,
                "renderer_name": renderer_name,
            },
        }


def _good_rows(n: int = 8) -> list[dict]:
    return [
        {"question": f"Q{i}", "golden_answer": f"A{i}", "toolkit": "X"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Registration + config
# ---------------------------------------------------------------------------


def test_registered_under_tinker_sdft_kind():
    assert get_algorithm("tinker_sdft") is TinkerSDFT


def test_config_defaults_sensible():
    cfg = TinkerSDFTConfig()
    # Top-K mode by default; cookbook recommends K=20 to approximate full-vocab KL.
    assert cfg.topk == 20
    assert cfg.batch_size == 4
    assert cfg.group_size == 1
    assert cfg.lora_rank == 8
    assert cfg.save_at_fractions == [1.0]


def test_config_rejects_extra_fields():
    with pytest.raises(Exception):
        TinkerSDFTConfig(no_such_field=1)


# ---------------------------------------------------------------------------
# train() — happy path with monkeypatched sdft.main
# ---------------------------------------------------------------------------


def test_train_calls_sdft_main_with_right_config(tmp_path: Path, monkeypatch):
    """Drive .train() with a stubbed sdft.main; capture the Config it sees."""
    captured: dict[str, Any] = {}

    async def fake_main(cfg: Any, sdft_dataset: Any, test_dataset: Any = None) -> None:
        captured["cfg"] = cfg
        captured["dataset"] = sdft_dataset
        captured["test_dataset"] = test_dataset

    from trajectory_labs.algorithms import tinker_sdft as mod
    monkeypatch.setattr(mod.sdft, "main", fake_main)

    # Avoid touching the network: tokenizer + renderer are cheap-ish loads
    # but get_tokenizer hits HF Hub. Stub them out.
    monkeypatch.setattr(mod, "get_tokenizer", lambda name: object())
    monkeypatch.setattr(mod.renderers, "get_renderer",
                        lambda name, *, tokenizer: object())
    # SDFTDataset's renderer arg is duck-typed; nothing else introspects it
    # inside the wrapper.

    algo = TinkerSDFT(learning_rate=5e-4, topk=15, max_steps=50, batch_size=4)
    res = algo.train(_Ctx(tmp_path, rows=_good_rows(16)))

    assert isinstance(res, RunResult)
    assert res.status == "completed"
    assert res.artifacts["run_dir"] == str(tmp_path)

    cfg = captured["cfg"]
    assert cfg.model_name == "Qwen/Qwen3.5-4B"
    assert cfg.renderer_name == "qwen3_5"
    assert cfg.learning_rate == 5e-4
    assert cfg.topk == 15
    assert cfg.max_steps == 50
    assert cfg.log_path == str(tmp_path)


def test_train_uses_renderer_from_handles_when_cfg_omits_it(tmp_path: Path, monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_main(cfg, sdft_dataset, test_dataset=None):
        captured["renderer_name"] = cfg.renderer_name

    from trajectory_labs.algorithms import tinker_sdft as mod
    monkeypatch.setattr(mod.sdft, "main", fake_main)
    monkeypatch.setattr(mod, "get_tokenizer", lambda n: object())
    monkeypatch.setattr(mod.renderers, "get_renderer",
                        lambda n, *, tokenizer: object())

    algo = TinkerSDFT()  # renderer_name=None on cfg
    algo.train(_Ctx(tmp_path, rows=_good_rows(), renderer_name="from_handles"))
    assert captured["renderer_name"] == "from_handles"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_train_rejects_non_tinker_backend(tmp_path: Path):
    algo = TinkerSDFT()
    ctx = _Ctx(tmp_path, rows=_good_rows())
    ctx.backend = type("B", (), {"name": "mock"})()
    with pytest.raises(RuntimeError, match="requires backend=tinker"):
        algo.train(ctx)


def test_train_rejects_missing_train_rows(tmp_path: Path):
    algo = TinkerSDFT()
    ctx = _Ctx(tmp_path)  # rows defaults to []
    with pytest.raises(RuntimeError, match="train_rows.*missing/empty"):
        algo.train(ctx)


def test_train_rejects_rows_missing_question_or_answer(tmp_path: Path, monkeypatch):
    algo = TinkerSDFT()
    bad_rows = [
        {"golden_answer": "A"},                 # no question
        {"question": "Q", "golden_answer": ""}, # empty answer
    ]
    with pytest.raises(RuntimeError, match="question.*golden_answer"):
        algo.train(_Ctx(tmp_path, rows=bad_rows))


def test_train_rejects_missing_renderer_name(tmp_path: Path):
    algo = TinkerSDFT()  # cfg.renderer_name = None
    with pytest.raises(RuntimeError, match="renderer_name not set"):
        algo.train(_Ctx(tmp_path, rows=_good_rows(), renderer_name=None))


def test_train_failure_returns_failed_run_result(tmp_path: Path, monkeypatch):
    """If sdft.main raises, the wrapper catches and returns status=failed."""
    async def boom(cfg, sdft_dataset, test_dataset=None):
        raise RuntimeError("teacher exploded")

    from trajectory_labs.algorithms import tinker_sdft as mod
    monkeypatch.setattr(mod.sdft, "main", boom)
    monkeypatch.setattr(mod, "get_tokenizer", lambda n: object())
    monkeypatch.setattr(mod.renderers, "get_renderer",
                        lambda n, *, tokenizer: object())

    algo = TinkerSDFT()
    res = algo.train(_Ctx(tmp_path, rows=_good_rows()))
    assert res.status == "failed"
    assert "teacher exploded" in (res.error or "")


# ---------------------------------------------------------------------------
# Checkpoint harvest
# ---------------------------------------------------------------------------


def test_checkpoint_manifest_harvested_into_artifacts(tmp_path: Path, monkeypatch):
    """A `checkpoints.jsonl` left by the cookbook is parsed into artifacts."""
    async def fake_main(cfg, sdft_dataset, test_dataset=None):
        # Simulate the cookbook writing a checkpoint manifest into log_path.
        (Path(cfg.log_path) / "checkpoints.jsonl").write_text(
            '{"name": "step_50", "state_path": "tinker://abc/50"}\n'
            '{"name": "final", "state_path": "tinker://abc/final"}\n'
        )

    from trajectory_labs.algorithms import tinker_sdft as mod
    monkeypatch.setattr(mod.sdft, "main", fake_main)
    monkeypatch.setattr(mod, "get_tokenizer", lambda n: object())
    monkeypatch.setattr(mod.renderers, "get_renderer",
                        lambda n, *, tokenizer: object())

    algo = TinkerSDFT()
    res = algo.train(_Ctx(tmp_path, rows=_good_rows()))
    assert res.artifacts["run_dir"] == str(tmp_path)
    assert res.artifacts["checkpoint-step_50"] == "tinker://abc/50"
    assert res.artifacts["checkpoint-final"] == "tinker://abc/final"

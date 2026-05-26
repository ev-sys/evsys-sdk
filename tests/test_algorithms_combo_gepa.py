"""Regression tests for ComboAlgorithm + GEPAPromptAlgorithm.

Combo: composes registered sub-algorithms (we use mock_sft + mock_rl which
ship with the SDK), verifies that artifacts and metrics from each phase get
namespaced and threaded forward, and that fail_fast aborts a chain.

GEPA: built-in hill-climber path (no gepa lib dep), with a simple scoring
function that rewards prompts containing a particular token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from trajectory_labs.algorithms import ComboAlgorithm, GEPAPromptAlgorithm
from trajectory_labs.log_stores.jsonl import JSONLLogStore
from trajectory_labs.protocols import RunContext
from trajectory_labs.registry import (
    get_algorithm,
    register_algorithm,
)


@dataclass
class _MockBackend:
    name: str = "mock"

    def prepare(self, *, model, run_dir): return {}
    def teardown(self, handles): return None


@dataclass
class _MockDataStore:
    name: str = "in_memory"
    def read_jsonl(self, path): return []
    def write_jsonl(self, path, rows): pass
    def read_json(self, path): return None
    def write_json(self, path, value): pass
    def exists(self, path): return False
    def list(self, prefix): return []


def _make_ctx(tmp_path: Path, extras: dict | None = None) -> RunContext:
    return RunContext(
        run_id="test_run",
        output_dir=str(tmp_path),
        config=None,
        data_store=_MockDataStore(),                # type: ignore[arg-type]
        log_store=JSONLLogStore(log_dir=str(tmp_path)),
        backend=_MockBackend(),                     # type: ignore[arg-type]
        extras=extras or {},
    )


class TestComboAlgorithm:
    def test_runs_phases_in_order(self, tmp_path: Path):
        algo = ComboAlgorithm(phases=[
            {"kind": "mock_sft", "config": {"num_epochs": 1, "save_at_fractions": [1.0]}},
            {"kind": "mock_rl",  "config": {"num_steps": 5, "save_every": 5}},
        ])
        ctx = _make_ctx(tmp_path)
        result = algo.train(ctx)
        assert result.status == "completed"
        # Each phase's artifacts/metrics are namespaced.
        assert any(k.startswith("phase1_mock_sft/") for k in result.metrics), result.metrics
        assert any(k.startswith("phase2_mock_rl/")  for k in result.metrics), result.metrics
        # Phase output dirs exist.
        assert (tmp_path / "phase1_mock_sft").is_dir()
        assert (tmp_path / "phase2_mock_rl").is_dir()

    def test_phase_artifacts_threaded_forward(self, tmp_path: Path):
        """ComboAlgorithm should make phase 1's final_checkpoint available to
        phase 2 via ctx.extras['init_checkpoint']."""
        seen: dict[str, Any] = {}

        @register_algorithm("_capture_init_ckpt")
        class CaptureInit:
            name = "_capture_init_ckpt"
            Config = type("Cfg", (), {"model_validate": staticmethod(lambda d: None)})

            def __init__(self, **kwargs): pass
            def train(self, ctx):
                seen["init"] = ctx.extras.get("init_checkpoint")
                from trajectory_labs.protocols import RunResult
                return RunResult(run_id=ctx.run_id, status="completed",
                                 metrics={"ok": 1.0}, artifacts={"final_checkpoint": "fake"})

        try:
            algo = ComboAlgorithm(phases=[
                {"kind": "mock_sft", "config": {"num_epochs": 1, "save_at_fractions": [1.0]}},
                {"kind": "_capture_init_ckpt"},
            ])
            algo.train(_make_ctx(tmp_path))
            assert seen.get("init") is not None
            assert "phase1_mock_sft" in seen["init"]
        finally:
            # Clean up the test-only registration to avoid leaking into other tests.
            from trajectory_labs.registry import _algorithms
            _algorithms.unregister("_capture_init_ckpt")

    def test_fail_fast_aborts(self, tmp_path: Path):
        @register_algorithm("_always_fail")
        class AlwaysFail:
            name = "_always_fail"
            Config = type("Cfg", (), {"model_validate": staticmethod(lambda d: None)})

            def __init__(self, **kwargs): pass
            def train(self, ctx):
                from trajectory_labs.protocols import RunResult
                return RunResult(run_id=ctx.run_id, status="failed", error="boom")

        try:
            algo = ComboAlgorithm(phases=[
                {"kind": "_always_fail"},
                {"kind": "mock_sft", "config": {"num_epochs": 1, "save_at_fractions": [1.0]}},
            ])
            result = algo.train(_make_ctx(tmp_path))
            assert result.status == "failed"
            # Phase 2 should have been skipped.
            assert not (tmp_path / "phase2_mock_sft").exists()
        finally:
            from trajectory_labs.registry import _algorithms
            _algorithms.unregister("_always_fail")

    def test_unknown_phase_kind_raises(self, tmp_path: Path):
        # ComboAlgorithm itself validates phases; an unknown registry name
        # should fail at train() not construction.
        algo = ComboAlgorithm(phases=[{"kind": "does_not_exist"}])
        result = algo.train(_make_ctx(tmp_path))
        assert result.status == "failed"
        assert "does_not_exist" in (result.error or "")

    def test_requires_at_least_one_phase(self):
        with pytest.raises(Exception):
            ComboAlgorithm(phases=[])

    def test_registered_in_registry(self):
        assert get_algorithm("combo") is ComboAlgorithm


class TestGEPAPromptAlgorithm:
    def test_registered(self):
        assert get_algorithm("gepa_prompt") is GEPAPromptAlgorithm

    def test_requires_examples(self, tmp_path: Path):
        algo = GEPAPromptAlgorithm(seed_prompt="seed", task_lm="mock", max_iterations=2)
        result = algo.train(_make_ctx(tmp_path))
        assert result.status == "failed"
        assert "prompt_examples" in (result.error or "")

    def test_hill_climber_runs(self, tmp_path: Path):
        examples = [
            {"inputs": {"q": "what is 1+1?"}, "expected": "MOCK_ANSWER"},
            {"inputs": {"q": "what is 2+2?"}, "expected": "MOCK_ANSWER"},
        ]
        algo = GEPAPromptAlgorithm(
            seed_prompt="You are helpful.",
            task_lm="mock",
            max_iterations=3,
            examples_per_eval=2,
            use_gepa_lib=False,  # force the built-in hill-climber
        )
        result = algo.train(_make_ctx(tmp_path, extras={"prompt_examples": examples}))
        assert result.status == "completed"
        # Hit gepa/best_score in metrics + wrote a prompts.json artifact.
        assert "gepa/best_score" in result.metrics
        prompt_path = result.artifacts.get("final_prompt")
        assert prompt_path is not None
        assert Path(prompt_path).is_file()
        import json
        content = json.loads(Path(prompt_path).read_text())
        assert "system_prompt" in content

    def test_score_fn_pluggable(self, tmp_path: Path):
        """User-supplied scoring fn should be called in place of the default."""
        called = []

        def reward_for_FOO(completion: str, expected) -> float:
            called.append(completion)
            return 1.0 if "FOO" in completion else 0.0

        algo = GEPAPromptAlgorithm(
            seed_prompt="seed",
            task_lm="mock",
            max_iterations=2,
            examples_per_eval=1,
            use_gepa_lib=False,
        )
        result = algo.train(_make_ctx(
            tmp_path,
            extras={
                "prompt_examples":  [{"inputs": {"q": "anything"}, "expected": "?"}],
                "prompt_score_fn":  reward_for_FOO,
            },
        ))
        assert result.status == "completed"
        assert len(called) >= 1  # scoring fn was used

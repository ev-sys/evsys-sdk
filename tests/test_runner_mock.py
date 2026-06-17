"""End-to-end with the mock backend (no external deps).

Validates the full pipeline:
  in_memory data -> transform -> mock backend -> mock_sft -> jsonl logs -> result
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evsys_sdk import (
    AlgorithmConfig,
    BackendConfig,
    DataConfig,
    DataStoreSpec,
    ExperimentConfig,
    LogStoreSpec,
    ModelConfig,
    RunConfig,
    TransformSpec,
    run_experiment,
)
from evsys_sdk.runner import _execute_run as execute_run  # noqa: F401  (smoke import)


def _make_cfg(tmp_path: Path, sample_rows: list[dict]) -> ExperimentConfig:
    return ExperimentConfig(
        name="mock_e2e",
        output_dir=str(tmp_path / "out"),
        log_store=LogStoreSpec(kind="jsonl"),
        run=RunConfig(
            name="mock_run",
            data=DataConfig(
                source_kind="in_memory",
                rows=sample_rows,
                transforms=[TransformSpec(kind="identity")],
            ),
            model=ModelConfig(name="tiny/fake"),
            algorithm=AlgorithmConfig(
                kind="mock_sft",
                params={"num_epochs": 1, "batch_size": 1, "save_at_fractions": [0.5, 1.0]},
            ),
            backend=BackendConfig(kind="mock"),
        ),
    )


def test_runner_mock_sft_end_to_end(tmp_path: Path, sample_rows):
    cfg = _make_cfg(tmp_path, sample_rows)
    results = run_experiment(cfg)
    assert len(results) == 1
    r = results[0]
    assert r.status == "completed"
    # Mock writes 2 checkpoints (50%, 100%) + final.
    assert "final_checkpoint" in r.artifacts
    # Logs landed.
    metrics_path = tmp_path / "out" / "mock_run" / "logs" / "metrics.jsonl"
    assert metrics_path.exists()
    rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    assert any("train/loss" in row.get("metrics", {}) for row in rows)


def test_runner_mock_rl_end_to_end(tmp_path: Path, sample_rows):
    cfg = ExperimentConfig(
        name="mock_rl_e2e",
        output_dir=str(tmp_path / "out"),
        run=RunConfig(
            name="rl_run",
            data=DataConfig(
                source_kind="in_memory",
                rows=sample_rows,
                transforms=[TransformSpec(kind="identity")],
            ),
            model=ModelConfig(name="tiny/fake"),
            algorithm=AlgorithmConfig(
                kind="mock_rl",
                params={
                    "num_steps": 50,
                    "save_every": 25,
                    "verifier_kind": "format_only",
                },
            ),
            backend=BackendConfig(kind="mock"),
        ),
    )
    results = run_experiment(cfg)
    assert results[0].status == "completed"
    assert "final_checkpoint" in results[0].artifacts
    # Should have 2 checkpoints saved at step 25, 50.
    assert any(k.startswith("ckpt_step_") for k in results[0].artifacts)


def test_runner_merges_extra_context_into_ctx_extras(tmp_path: Path, sample_rows):
    """``extra_context`` (store + dashboard_run_id from Experiment) lands in
    ``RunContext.extras`` so in-loop validation can upload."""
    from evsys_sdk.registry import register_algorithm

    captured: dict = {}

    @register_algorithm("capture_extras")
    class _Capture:
        name = "capture_extras"
        Config = type("C", (), {})

        def __init__(self, **kwargs):
            pass

        def train(self, ctx):
            captured.update(ctx.extras)
            from evsys_sdk.protocols import RunResult
            return RunResult(run_id=ctx.run_id, status="completed", metrics={}, artifacts={})

    cfg = _make_cfg(tmp_path, sample_rows).model_copy()
    cfg.run.algorithm = AlgorithmConfig(kind="capture_extras", params={})

    sentinel_store = object()
    run_experiment(
        cfg,
        extra_context={"store": sentinel_store, "dashboard_run_id": "run_xyz"},
    )
    assert captured.get("store") is sentinel_store
    assert captured.get("dashboard_run_id") == "run_xyz"


def test_runner_yaml_path(tmp_path: Path, sample_rows):
    cfg = _make_cfg(tmp_path, sample_rows)
    yaml_path = tmp_path / "exp.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg.model_dump(exclude_none=True, mode="json")))
    results = run_experiment(yaml_path)
    assert results[0].status == "completed"


def test_runner_failure_path_produces_failed_result(tmp_path: Path, sample_rows):
    cfg = ExperimentConfig(
        name="fail_e2e",
        output_dir=str(tmp_path / "out"),
        run=RunConfig(
            name="bad",
            data=DataConfig(source_kind="in_memory", rows=sample_rows),
            model=ModelConfig(name="x"),
            algorithm=AlgorithmConfig(kind="mock_sft"),
            backend=BackendConfig(kind="mock", params={"fail_on_prepare": True}),
        ),
    )
    results = run_experiment(cfg)
    assert results[0].status == "failed"
    assert "MockBackend" in (results[0].error or "")

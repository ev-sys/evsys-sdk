"""Continual-learning experiment mode.

One base ``run`` + a list of datasets → sequential stages in one experiment,
each starting from the previous stage's weights (fresh optimizer), each scored
on all benchmarks. These tests use a custom ``train_fn`` so the chaining logic
is exercised without a real backend.
"""

from __future__ import annotations

from evsys_sdk.config import ExperimentConfig
from evsys_sdk.experiment import Experiment
from evsys_sdk.protocols import RunResult


def _ds(v: int) -> dict:
    return {"source_kind": "in_memory", "rows": [{"a": v}],
            "transforms": [{"kind": "identity"}]}


def _cfg(tmp_path, n: int) -> ExperimentConfig:
    return ExperimentConfig(
        name="cont", output_dir=str(tmp_path / "out"),
        run={
            "name": "base",
            "data": _ds(0),
            "model": {"name": "m"},
            "algorithm": {"kind": "mock_sft"},
            "backend": {"kind": "mock"},
        },
        continual={"datasets": [_ds(i + 1) for i in range(n)]},
    )


def test_continual_chains_weights_and_swaps_data(tmp_path):
    seen: list[dict] = []

    def fake_train(cfg: ExperimentConfig) -> list[RunResult]:
        run = cfg.run
        seen.append({
            "name": run.name,
            "rows": run.data.rows,
            "init": run.model.init_from_checkpoint,
            "tags": run.tags,
        })
        return [RunResult(
            run_id="r", status="completed", metrics={"loss": 0.1},
            artifacts={
                "state-final": f"state::{run.name}",
                "checkpoint-final": f"sampler::{run.name}",
            },
        )]

    exp = Experiment(_cfg(tmp_path, 3), train_fn=fake_train)
    res = exp.run()

    # Three stages, one per dataset, in order.
    assert len(res.arms) == 3
    assert [s["name"] for s in seen] == ["base_stage0", "base_stage1", "base_stage2"]
    # Data is swapped per stage.
    assert [s["rows"] for s in seen] == [[{"a": 1}], [{"a": 2}], [{"a": 3}]]
    # Weights chain: stage0 fresh; each later stage inits from the prior
    # stage's full training-state checkpoint (not the sampler path).
    assert seen[0]["init"] is None
    assert seen[1]["init"] == "state::base_stage0"
    assert seen[2]["init"] == "state::base_stage1"
    # Stages are tagged so the chain is identifiable.
    assert "continual" in seen[1]["tags"] and "stage:1" in seen[1]["tags"]


def test_continual_repeats_whole_chain_per_seed(tmp_path):
    """n_repeats > 1 replicates the entire chain once per seed; weights chain
    only within a chain, and each chain restarts from the base model."""
    seen: list[dict] = []

    def fake_train(cfg: ExperimentConfig) -> list[RunResult]:
        run = cfg.run
        seen.append({"name": run.name, "seed": run.seed,
                     "init": run.model.init_from_checkpoint})
        return [RunResult(
            run_id="r", status="completed", metrics={"loss": 0.1},
            artifacts={"state-final": f"state::{run.name}"},
        )]

    cfg = _cfg(tmp_path, 2)                       # 2 datasets
    cfg = cfg.model_copy(update={"n_repeats": 2})  # 2 seeds → 2 chains
    exp = Experiment(cfg, train_fn=fake_train)
    res = exp.run()

    assert len(res.arms) == 4  # 2 chains x 2 stages
    assert [s["name"] for s in seen] == [
        "base_stage0__s42", "base_stage1__s42",
        "base_stage0__s43", "base_stage1__s43",
    ]
    assert [s["seed"] for s in seen] == [42, 42, 43, 43]
    # Each chain starts fresh; stage1 inits from THIS chain's stage0.
    assert seen[0]["init"] is None
    assert seen[1]["init"] == "state::base_stage0__s42"
    assert seen[2]["init"] is None
    assert seen[3]["init"] == "state::base_stage0__s43"


def test_continual_stops_on_failure(tmp_path):
    calls: list[str] = []

    def fake_train(cfg: ExperimentConfig) -> list[RunResult]:
        run = cfg.run
        calls.append(run.name)
        status = "failed" if run.name == "base_stage1" else "completed"
        return [RunResult(
            run_id="r", status=status,
            metrics={}, artifacts={"state-final": f"state::{run.name}"},
            error=None if status == "completed" else "boom",
        )]

    exp = Experiment(_cfg(tmp_path, 3), train_fn=fake_train)
    res = exp.run()

    # Stage 2 never runs because stage 1 failed.
    assert calls == ["base_stage0", "base_stage1"]
    assert len(res.arms) == 2
    assert res.arms[-1].status == "failed"

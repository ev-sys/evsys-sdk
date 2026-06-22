"""Multi-stage recipes (e.g. SFT → RL) — the ``stages`` modifier.

One base ``run`` + a list of stages, each with its OWN algorithm + data → one arm
per stage in one experiment, each starting from the previous stage's weights
(fresh optimizer). The multi-algorithm sibling of ``continual``. Uses a custom
``train_fn`` so the chaining is exercised without a real backend (mirrors
``test_continual.py``).
"""

from __future__ import annotations

import pytest

from evsys_sdk.config import ExperimentConfig
from evsys_sdk.experiment import Experiment
from evsys_sdk.protocols import RunResult
from evsys_sdk.yaml_loader import apply_dry_run


def _data(v: int) -> dict:
    return {"source_kind": "in_memory", "rows": [{"a": v}],
            "transforms": [{"kind": "identity"}]}


def _cfg(tmp_path, stage_algos: list[str]) -> ExperimentConfig:
    return ExperimentConfig(
        name="stg", output_dir=str(tmp_path / "out"),
        run={
            "name": "base",
            "data": _data(0),
            "model": {"name": "m"},
            "algorithm": {"kind": "mock_sft"},   # placeholder; stages override
            "backend": {"kind": "mock"},
        },
        stages={"stages": [
            {"algorithm": {"kind": k}, "data": _data(i + 1)}
            for i, k in enumerate(stage_algos)
        ]},
    )


def test_stages_chains_weights_and_swaps_algorithm_and_data(tmp_path):
    seen: list[dict] = []

    def fake_train(cfg: ExperimentConfig) -> list[RunResult]:
        run = cfg.run
        seen.append({
            "name": run.name,
            "algo": run.algorithm.kind,
            "rows": run.data.rows,
            "init": run.model.init_from_checkpoint,
            "tags": run.tags,
        })
        return [RunResult(
            run_id="r", status="completed", metrics={"loss": 0.1},
            artifacts={"state-final": f"state::{run.name}"},
        )]

    exp = Experiment(_cfg(tmp_path, ["mock_sft", "mock_rl"]), train_fn=fake_train)
    res = exp.run()

    # Two stages → two arms (one run per stage), like continual for datasets.
    assert len(res.arms) == 2
    assert [s["name"] for s in seen] == ["base_stage0_mock_sft", "base_stage1_mock_rl"]
    # Each stage swaps BOTH algorithm and data.
    assert [s["algo"] for s in seen] == ["mock_sft", "mock_rl"]
    assert [s["rows"] for s in seen] == [[{"a": 1}], [{"a": 2}]]
    # Weights chain: stage 0 fresh; stage 1 inits from stage 0's training state.
    assert seen[0]["init"] is None
    assert seen[1]["init"] == "state::base_stage0_mock_sft"
    # Tagged so the chain + per-stage recipe are identifiable.
    assert "stages" in seen[1]["tags"] and "stage:1" in seen[1]["tags"]
    assert "algo:mock_rl" in seen[1]["tags"]


def test_stages_repeats_whole_chain_per_seed(tmp_path):
    seen: list[dict] = []

    def fake_train(cfg: ExperimentConfig) -> list[RunResult]:
        run = cfg.run
        seen.append({"name": run.name, "seed": run.seed,
                     "init": run.model.init_from_checkpoint})
        return [RunResult(run_id="r", status="completed", metrics={},
                          artifacts={"state-final": f"state::{run.name}"})]

    cfg = _cfg(tmp_path, ["mock_sft", "mock_rl"]).model_copy(update={"n_repeats": 2})
    exp = Experiment(cfg, train_fn=fake_train)
    res = exp.run()

    assert len(res.arms) == 4  # 2 chains x 2 stages
    assert [s["name"] for s in seen] == [
        "base_stage0_mock_sft__s42", "base_stage1_mock_rl__s42",
        "base_stage0_mock_sft__s43", "base_stage1_mock_rl__s43",
    ]
    # Each chain restarts from base; stage 1 inits from THIS chain's stage 0.
    assert [s["init"] for s in seen] == [
        None, "state::base_stage0_mock_sft__s42",
        None, "state::base_stage0_mock_sft__s43",
    ]


def test_stages_stops_on_failure(tmp_path):
    calls: list[str] = []

    def fake_train(cfg: ExperimentConfig) -> list[RunResult]:
        run = cfg.run
        calls.append(run.name)
        status = "failed" if "stage0" in run.name else "completed"
        return [RunResult(run_id="r", status=status, metrics={},
                          artifacts={"state-final": f"state::{run.name}"},
                          error=None if status == "completed" else "boom")]

    exp = Experiment(_cfg(tmp_path, ["mock_sft", "mock_rl"]), train_fn=fake_train)
    res = exp.run()
    # Stage 0 fails → stage 1 (RL) never runs.
    assert calls == ["base_stage0_mock_sft"]
    assert len(res.arms) == 1
    assert res.arms[-1].status == "failed"


def test_stages_requires_single_run(tmp_path):
    base = _cfg(tmp_path, ["mock_sft"]).run
    stages = _cfg(tmp_path, ["mock_sft", "mock_rl"]).stages
    with pytest.raises(Exception, match="stages requires a single"):
        ExperimentConfig(name="x", runs=[base], stages=stages)


def test_stages_and_continual_mutually_exclusive(tmp_path):
    base = _cfg(tmp_path, ["mock_sft"]).run
    stages = _cfg(tmp_path, ["mock_sft"]).stages
    with pytest.raises(Exception, match="either"):
        ExperimentConfig(name="x", run=base, stages=stages,
                         continual={"datasets": [_data(1)]})


def test_dry_run_caps_each_stage(tmp_path):
    cfg = _cfg(tmp_path, ["mock_sft", "mock_sft"])  # mock_sft has max_steps
    apply_dry_run(cfg, steps=3)
    assert cfg.log_rollouts is True
    assert all(s.algorithm.params["max_steps"] == 3 for s in cfg.stages.stages)

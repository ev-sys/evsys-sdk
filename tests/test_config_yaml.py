"""Config + YAML round-trip + matrix expansion."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evsys_sdk import (
    ExperimentConfig,
    RunConfig,
    dump_yaml,
    load_yaml,
    validate_yaml,
)


def _minimal_run_dict():
    return {
        "name": "r1",
        "data": {"source_kind": "in_memory", "rows": [{"x": 1}]},
        "model": {"name": "tiny/fake"},
        "algorithm": {"kind": "mock_sft", "params": {"num_epochs": 1}},
        "backend": {"kind": "mock"},
    }


def test_minimal_yaml(tmp_path: Path):
    data = {"name": "exp", "run": _minimal_run_dict()}
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(data))
    cfg = load_yaml(p)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.run is not None
    assert cfg.run.name == "r1"


def test_extra_field_is_rejected(tmp_path: Path):
    data = {"name": "exp", "run": _minimal_run_dict(), "extra_typo": True}
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(Exception):
        load_yaml(p)


def test_exactly_one_of_run_runs_matrix(tmp_path: Path):
    data = {
        "name": "exp",
        "run": _minimal_run_dict(),
        "runs": [_minimal_run_dict()],
    }
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(Exception):
        load_yaml(p)


def test_matrix_expansion(tmp_path: Path):
    base = _minimal_run_dict()
    data = {
        "name": "exp",
        "matrix": {
            "base_run": base,
            "axes": {
                "algorithm.params.num_epochs": [1, 2],
                "algorithm.params.lora_rank": [4, 8],
            },
            "name_template": "r__e{algorithm.params.num_epochs}__r{algorithm.params.lora_rank}",
        },
    }
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(data))
    cfg = load_yaml(p)
    assert cfg.matrix is None
    assert cfg.runs is not None
    assert len(cfg.runs) == 4
    names = sorted(r.name for r in cfg.runs)
    assert names == sorted(["r__e1__r4", "r__e1__r8", "r__e2__r4", "r__e2__r8"])
    # Confirm the values made it into the run
    by_name = {r.name: r for r in cfg.runs}
    assert by_name["r__e2__r8"].algorithm.params["num_epochs"] == 2
    assert by_name["r__e2__r8"].algorithm.params["lora_rank"] == 8


def test_dump_yaml_roundtrip(tmp_path: Path):
    base = _minimal_run_dict()
    cfg = ExperimentConfig(name="exp", run=RunConfig.model_validate(base))
    text = dump_yaml(cfg)
    cfg2 = load_yaml(yaml.safe_load(text))
    assert cfg2.name == cfg.name
    assert cfg2.run is not None
    assert cfg2.run.name == cfg.run.name


def test_validate_deep_catches_bad_algorithm_param(tmp_path: Path):
    base = _minimal_run_dict()
    base["algorithm"]["params"]["unknown_field"] = 9000
    data = {"name": "exp", "run": base}
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(data))
    errors = validate_yaml(p, deep=True)
    assert any("algorithm" in e for e in errors)


def test_unknown_algorithm_in_deep_validate(tmp_path: Path):
    base = _minimal_run_dict()
    base["algorithm"]["kind"] = "no_such_algorithm"
    data = {"name": "exp", "run": base}
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(data))
    errors = validate_yaml(p, deep=True)
    assert any("no_such_algorithm" in e or "algorithm" in e for e in errors)

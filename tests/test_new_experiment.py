"""Tests for `trajectory_labs.new_experiment.new_experiment` and the
`trajex new-experiment` CLI subcommand.
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pytest
import yaml

from trajectory_labs.cli import main as cli_main
from trajectory_labs.config import ExperimentConfig
from trajectory_labs.new_experiment import new_experiment


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_creates_dated_dir(tmp_path: Path):
    path = new_experiment(tmp_path, "lora_rank_sweep", today=date(2026, 5, 31))
    assert path == tmp_path / "experiments" / "20260531_lora_rank_sweep"
    assert path.is_dir()


def test_writes_config_yaml_and_run_py(tmp_path: Path):
    path = new_experiment(tmp_path, "first_check", today=date(2026, 1, 2))
    assert (path / "config.yaml").is_file()
    assert (path / "run.py").is_file()


def test_creates_experiments_root_if_missing(tmp_path: Path):
    new_experiment(tmp_path, "x", today=date(2026, 1, 1))
    assert (tmp_path / "experiments").is_dir()


def test_works_in_existing_experiments_dir(tmp_path: Path):
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "prior").mkdir()
    new_experiment(tmp_path, "another", today=date(2026, 1, 1))
    assert (tmp_path / "experiments" / "prior").is_dir()  # unchanged
    assert (tmp_path / "experiments" / "20260101_another").is_dir()


# ---------------------------------------------------------------------------
# Slug normalization
# ---------------------------------------------------------------------------


def test_slug_lowercased(tmp_path: Path):
    path = new_experiment(tmp_path, "LoraRankSweep", today=date(2026, 5, 31))
    assert path.name == "20260531_lorarank-sweep" or path.name.endswith("loraranksweep")


def test_slug_spaces_become_underscores(tmp_path: Path):
    path = new_experiment(tmp_path, "lora rank sweep", today=date(2026, 5, 31))
    assert path.name == "20260531_lora_rank_sweep"


def test_slug_strips_invalid_chars(tmp_path: Path):
    path = new_experiment(tmp_path, "foo@bar!baz", today=date(2026, 5, 31))
    assert path.name == "20260531_foobarbaz"


def test_slug_preserves_dash_and_underscore(tmp_path: Path):
    path = new_experiment(tmp_path, "rank-sweep_v2", today=date(2026, 5, 31))
    assert path.name == "20260531_rank-sweep_v2"


def test_empty_slug_after_normalization_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="empty"):
        new_experiment(tmp_path, "@@@@", today=date(2026, 5, 31))


def test_whitespace_only_slug_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="empty"):
        new_experiment(tmp_path, "   ", today=date(2026, 5, 31))


# ---------------------------------------------------------------------------
# Refuse-overwrite
# ---------------------------------------------------------------------------


def test_refuses_existing_dir(tmp_path: Path):
    new_experiment(tmp_path, "twice", today=date(2026, 1, 1))
    with pytest.raises(FileExistsError, match="already exists"):
        new_experiment(tmp_path, "twice", today=date(2026, 1, 1))


# ---------------------------------------------------------------------------
# Content validity
# ---------------------------------------------------------------------------


def test_config_yaml_parses_as_experiment_config(tmp_path: Path):
    path = new_experiment(tmp_path, "valid", today=date(2026, 1, 1))
    data = yaml.safe_load((path / "config.yaml").read_text())
    # Pydantic round-trip works (we don't actually run it; just validate shape).
    cfg = ExperimentConfig.model_validate(data)
    assert cfg.name == "valid"
    assert cfg.run is not None
    assert cfg.run.algorithm.kind == "local_sft"
    assert cfg.metadata.get("hypothesis", "").startswith("TODO")


def test_config_yaml_includes_metadata_block(tmp_path: Path):
    path = new_experiment(tmp_path, "valid", today=date(2026, 1, 1))
    text = (path / "config.yaml").read_text()
    # The commented-out keys are pedagogical — researcher discovers them.
    for tok in ("hypothesis:", "tags:", "success_metric:", "benchmark:", "breakdown_keys:"):
        assert tok in text


def test_run_py_is_syntactically_valid(tmp_path: Path):
    path = new_experiment(tmp_path, "valid", today=date(2026, 1, 1))
    src = (path / "run.py").read_text()
    ast.parse(src)  # raises SyntaxError if invalid


def test_run_py_imports_experiment_and_scripts(tmp_path: Path):
    path = new_experiment(tmp_path, "valid", today=date(2026, 1, 1))
    src = (path / "run.py").read_text()
    assert "from trajectory_labs import Experiment" in src
    assert "import scripts" in src
    assert "Experiment.from_yaml" in src
    assert ".run()" in src


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_new_experiment_happy(tmp_path: Path, capsys):
    rc = cli_main(["new-experiment", "rank_sweep",
                   "--project-root", str(tmp_path),
                   "--date", "20260601"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "scaffolded experiment" in out
    assert (tmp_path / "experiments" / "20260601_rank_sweep" / "run.py").is_file()


def test_cli_new_experiment_invalid_date(tmp_path: Path, capsys):
    rc = cli_main(["new-experiment", "x",
                   "--project-root", str(tmp_path),
                   "--date", "not-a-date"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "YYYYMMDD" in err


def test_cli_new_experiment_default_date_is_today(tmp_path: Path):
    """Without --date, we use today's date; just check the dir starts with 8 digits."""
    rc = cli_main(["new-experiment", "x", "--project-root", str(tmp_path)])
    assert rc == 0
    children = list((tmp_path / "experiments").iterdir())
    assert len(children) == 1
    assert children[0].name[:8].isdigit()


def test_cli_new_experiment_refuses_duplicate(tmp_path: Path, capsys):
    cli_main(["new-experiment", "dup", "--project-root", str(tmp_path),
              "--date", "20260101"])
    rc = cli_main(["new-experiment", "dup", "--project-root", str(tmp_path),
                   "--date", "20260101"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "already exists" in err


def test_cli_new_experiment_rejects_empty_slug(tmp_path: Path, capsys):
    rc = cli_main(["new-experiment", "@@@", "--project-root", str(tmp_path),
                   "--date", "20260101"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "empty" in err

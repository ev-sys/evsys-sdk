"""CLI smoke tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from evsys_sdk.cli import main as cli_main


def _make_cfg_dict(tmp_path: Path) -> dict:
    return {
        "name": "exp",
        "output_dir": str(tmp_path / "out"),
        "run": {
            "name": "r1",
            "data": {"source_kind": "in_memory", "rows": [{"x": 1}]},
            "model": {"name": "tiny/fake"},
            "algorithm": {"kind": "mock_sft", "params": {"num_epochs": 1}},
            "backend": {"kind": "mock"},
            "eval": {"enabled": False},
        },
    }


def test_cli_validate_ok(tmp_path: Path, capsys):
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(_make_cfg_dict(tmp_path)))
    rc = cli_main(["validate", str(p)])
    assert rc == 0


def test_cli_validate_deep_ok(tmp_path: Path, capsys):
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(_make_cfg_dict(tmp_path)))
    rc = cli_main(["validate", str(p), "--deep"])
    assert rc == 0


def test_cli_run(tmp_path: Path, capsys):
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(_make_cfg_dict(tmp_path)))
    rc = cli_main(["run", str(p), "-o", str(tmp_path / "summary.json")])
    assert rc == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary[0]["status"] == "completed"


def test_cli_list(capsys):
    rc = cli_main(["list", "--kind", "algorithms"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mock_sft" in out


def test_cli_schema(capsys):
    rc = cli_main(["schema", "algorithm", "mock_sft"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "properties" in parsed

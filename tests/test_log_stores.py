"""Log store implementations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trajectory_labs.log_stores.jsonl import JSONLLogStore
from trajectory_labs.log_stores.multiplex import MultiplexLogStore


def test_jsonl_log_store_writes_metrics(tmp_path: Path):
    s = JSONLLogStore(log_dir=str(tmp_path / "logs"))
    s.log_metrics({"a": 1.0, "b": 2.0}, step=1)
    s.log_scalar("a", 3.0, step=2)
    s.close()

    rows = []
    with (tmp_path / "logs" / "metrics.jsonl").open() as f:
        for line in f:
            rows.append(json.loads(line))
    assert rows[0]["metrics"]["a"] == 1.0
    assert rows[0]["metrics"]["b"] == 2.0
    assert rows[1]["metrics"]["a"] == 3.0


def test_jsonl_log_store_hyperparams_merge(tmp_path: Path):
    s = JSONLLogStore(log_dir=str(tmp_path / "logs"))
    s.log_hyperparams({"lr": 1e-4})
    s.log_hyperparams({"epochs": 3})
    data = json.loads((tmp_path / "logs" / "hyperparams.json").read_text())
    assert data == {"lr": 1e-4, "epochs": 3}


def test_jsonl_log_store_artifacts(tmp_path: Path):
    s = JSONLLogStore(log_dir=str(tmp_path / "logs"))
    s.log_artifact("ckpt", "/tmp/x", kind="checkpoint")
    s.log_artifact("plot", "/tmp/p.png", kind="image")
    arts = json.loads((tmp_path / "logs" / "artifacts.json").read_text())
    assert len(arts) == 2
    assert arts[0]["name"] == "ckpt"
    assert arts[1]["kind"] == "image"


def test_jsonl_log_store_close_blocks_further_logs(tmp_path: Path):
    s = JSONLLogStore(log_dir=str(tmp_path / "logs"))
    s.close()
    with pytest.raises(RuntimeError):
        s.log_metrics({"x": 1}, step=1)


def test_multiplex_fans_out(tmp_path: Path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    s = MultiplexLogStore(
        children=[
            {"kind": "jsonl", "params": {"log_dir": str(a_dir)}},
            {"kind": "jsonl", "params": {"log_dir": str(b_dir)}},
        ]
    )
    s.log_metrics({"x": 1.0}, step=1)
    s.close()
    assert (a_dir / "metrics.jsonl").exists()
    assert (b_dir / "metrics.jsonl").exists()

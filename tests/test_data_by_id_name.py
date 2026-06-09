"""Reference training data + benchmarks by dashboard id/name (fetched into .evsys/).

The preferred way to reference data in a stored experiment script is by
``dataset_id`` / ``dataset_name`` (training) or ``metadata.benchmark.id|name``
(benchmark): the SDK pulls it into the local ``.evsys/`` workspace and
works from that cache, instead of relying on a local ``path``. Name resolves to
the latest version's id.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evsys_sdk.config import DataConfig
from evsys_sdk.workspace import MaterializedDataset, Workspace, read_jsonl_rows


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_dataconfig_accepts_dataset_id():
    d = DataConfig(dataset_id="ds-1")
    assert d.dataset_id == "ds-1"
    assert d.source_kind == "jsonl"  # ignored when id set


def test_dataconfig_accepts_dataset_name():
    assert DataConfig(dataset_name="sft_overdose").dataset_name == "sft_overdose"


def test_dataconfig_rejects_unknown_key():
    with pytest.raises(Exception):
        DataConfig(datasetid="typo")


# ---------------------------------------------------------------------------
# Name → id resolution (latest version wins)
# ---------------------------------------------------------------------------


class _FakeStore:
    def __init__(self, datasets=None, benchmarks=None) -> None:
        self._datasets = datasets or []
        self._benchmarks = benchmarks or []

    def list_datasets(self, project_id=None):
        return list(self._datasets)

    def list_benchmarks(self, project_id=None):
        return list(self._benchmarks)


def test_dataset_id_for_name_picks_latest_version(tmp_path):
    store = _FakeStore(datasets=[
        {"id": "a", "name": "ds", "version": 1},
        {"id": "b", "name": "ds", "version": 3},
        {"id": "c", "name": "other", "version": 9},
    ])
    ws = Workspace(store, root=str(tmp_path))
    assert ws.dataset_id_for_name("ds") == "b"


def test_benchmark_id_for_name_picks_latest_version(tmp_path):
    store = _FakeStore(benchmarks=[
        {"id": "x", "name": "bench", "version": 2},
        {"id": "y", "name": "bench", "version": 5},
    ])
    ws = Workspace(store, root=str(tmp_path))
    assert ws.benchmark_id_for_name("bench") == "y"


def test_id_for_name_missing_raises(tmp_path):
    ws = Workspace(_FakeStore(), root=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no dataset named"):
        ws.dataset_id_for_name("nope")


def test_read_jsonl_rows(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n\n" + json.dumps({"a": 2}) + "\n")
    assert read_jsonl_rows(str(p)) == [{"a": 1}, {"a": 2}]


# ---------------------------------------------------------------------------
# runner._load_rows — training data by id / name
# ---------------------------------------------------------------------------


class _FakeWorkspace:
    """Stand-in for Workspace: serves a fixed dataset id and a local jsonl."""

    last_pulled: str | None = None

    def __init__(self, *a, **kw) -> None:
        pass

    def dataset_id_for_name(self, name: str) -> str:
        return {"sft_overdose": "ds-resolved"}[name]

    def pull_dataset(self, dataset_id: str, *, force: bool = False) -> MaterializedDataset:
        _FakeWorkspace.last_pulled = dataset_id
        return MaterializedDataset(_FakeWorkspace.jsonl_path, "chat_messages", None, 2, cached=False)


def test_load_rows_by_dataset_id(tmp_path, monkeypatch):
    from evsys_sdk import runner

    jsonl = tmp_path / "ds.jsonl"
    jsonl.write_text(json.dumps({"messages": [], "target_assistant": "y1"}) + "\n"
                     + json.dumps({"messages": [], "target_assistant": "y2"}) + "\n")
    _FakeWorkspace.jsonl_path = str(jsonl)
    _FakeWorkspace.last_pulled = None
    monkeypatch.setattr("evsys_sdk.workspace.Workspace", _FakeWorkspace)

    rows = runner._load_rows(DataConfig(dataset_id="ds-42"), data_store=None)
    assert _FakeWorkspace.last_pulled == "ds-42"
    assert [r["target_assistant"] for r in rows] == ["y1", "y2"]


def test_load_rows_by_dataset_name(tmp_path, monkeypatch):
    from evsys_sdk import runner

    jsonl = tmp_path / "ds.jsonl"
    jsonl.write_text(json.dumps({"messages": [], "target_assistant": "z"}) + "\n")
    _FakeWorkspace.jsonl_path = str(jsonl)
    _FakeWorkspace.last_pulled = None
    monkeypatch.setattr("evsys_sdk.workspace.Workspace", _FakeWorkspace)

    rows = runner._load_rows(DataConfig(dataset_name="sft_overdose"), data_store=None)
    assert _FakeWorkspace.last_pulled == "ds-resolved"  # name → id → pull
    assert rows == [{"messages": [], "target_assistant": "z"}]


# ---------------------------------------------------------------------------
# Experiment._resolve_benchmark — benchmark by id / name
# ---------------------------------------------------------------------------


class _FakeBenchWorkspace:
    last_pulled: str | None = None

    def __init__(self, *a, **kw) -> None:
        pass

    def benchmark_id_for_name(self, name: str) -> str:
        return {"composio_eval": "bench-resolved"}[name]

    def pull_benchmark(self, benchmark_id: str, *, force: bool = False) -> MaterializedDataset:
        _FakeBenchWorkspace.last_pulled = benchmark_id
        return MaterializedDataset(_FakeBenchWorkspace.jsonl_path, "harbor_task", None, 1, cached=False)


def _harbor_row(task_id: str, expected: str) -> dict:
    return {
        "task_id": task_id,
        "instruction": f"Q-{task_id}",
        "verifier": {"kind": "in_process", "fn_name": "exact_match", "expected": expected, "params": {}},
        "metadata": {},
    }


def _experiment(monkeypatch, tmp_path, spec):
    from evsys_sdk.config import ExperimentConfig
    from evsys_sdk.experiment import Experiment

    jsonl = tmp_path / "bench.jsonl"
    jsonl.write_text("\n".join(json.dumps(_harbor_row(t, "A")) for t in ["t1", "t2"]) + "\n")
    _FakeBenchWorkspace.jsonl_path = str(jsonl)
    _FakeBenchWorkspace.last_pulled = None
    monkeypatch.setattr("evsys_sdk.workspace.Workspace", _FakeBenchWorkspace)

    cfg = ExperimentConfig(name="exp", run={
        "name": "r1",
        "data": {"source_kind": "in_memory", "rows": [{"x": 1}]},
        "model": {"name": "tiny/fake"},
        "algorithm": {"kind": "mock_sft"},
        "backend": {"kind": "mock"},
    })
    return Experiment(cfg, store=object())._resolve_benchmark(spec)


def test_resolve_benchmark_by_id(tmp_path, monkeypatch):
    bench = _experiment(monkeypatch, tmp_path, {"id": "bench-9"})
    assert _FakeBenchWorkspace.last_pulled == "bench-9"
    assert bench is not None and len(bench.tasks) == 2
    assert bench.tasks[0].task_id == "t1"


def test_resolve_benchmark_by_name(tmp_path, monkeypatch):
    bench = _experiment(monkeypatch, tmp_path, {"name": "composio_eval"})
    assert _FakeBenchWorkspace.last_pulled == "bench-resolved"  # name → id → pull
    assert bench is not None and len(bench.tasks) == 2


def test_resolve_benchmark_path_still_wins(tmp_path, monkeypatch):
    # path is the offline fallback; when present it's used directly (no pull).
    root = tmp_path / "b"
    root.mkdir()
    (root / "tasks.jsonl").write_text(json.dumps(_harbor_row("p1", "A")) + "\n")
    monkeypatch.setattr("evsys_sdk.workspace.Workspace", _FakeBenchWorkspace)
    _FakeBenchWorkspace.last_pulled = None

    from evsys_sdk.config import ExperimentConfig
    from evsys_sdk.experiment import Experiment
    cfg = ExperimentConfig(name="exp", run={
        "name": "r1", "data": {"source_kind": "in_memory", "rows": [{"x": 1}]},
        "model": {"name": "t/f"}, "algorithm": {"kind": "mock_sft"}, "backend": {"kind": "mock"},
    })
    bench = Experiment(cfg, store=object())._resolve_benchmark({"path": str(root), "id": "ignored"})
    assert _FakeBenchWorkspace.last_pulled is None  # path used, no pull
    assert bench is not None and bench.tasks[0].task_id == "p1"

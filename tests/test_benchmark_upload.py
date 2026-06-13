"""Tests for `evsys_sdk.benchmark_upload.upload_benchmark` and the
`evsys benchmark upload` CLI subcommand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from evsys_sdk.benchmark_upload import (
    BENCHMARK_FORMAT,
    UploadResult,
    upload_benchmark,
)
from evsys_sdk.cli import main as cli_main


# ---------------------------------------------------------------------------
# Fake store + benchmark fixtures
# ---------------------------------------------------------------------------


class _FakeStore:
    """Tracks create_benchmark + add_benchmark_rows + list_benchmarks."""

    def __init__(self) -> None:
        self.benchmarks: list[dict] = []
        self.rows_by_id: dict[str, list[dict]] = {}
        self._next_id = 0
        self.list_calls = 0

    def _id(self) -> str:
        self._next_id += 1
        return f"bench-{self._next_id}"

    def list_benchmarks(self, project_id: str | None = None) -> list[dict]:
        self.list_calls += 1
        return list(self.benchmarks)

    def create_benchmark(self, **kw: Any) -> dict:
        record = {"id": self._id(), **kw}
        self.benchmarks.append(record)
        return record

    def add_benchmark_rows(self, benchmark_id: str, rows: list[dict],
                           *, start_idx: int = 0) -> list[dict]:
        self.rows_by_id.setdefault(benchmark_id, []).extend(rows)
        return rows


def _row(task_id: str, fn: str, expected: str) -> dict:
    return {
        "task_id": task_id, "instruction": f"Q-{task_id}",
        "verifier": {"kind": "in_process", "fn_name": fn, "expected": expected, "params": {}},
        "metadata": {},
    }


@pytest.fixture()
def bench_dir(tmp_path: Path) -> Path:
    root = tmp_path / "toy"
    root.mkdir()
    (root / "tasks.jsonl").write_text(
        "\n".join(json.dumps(_row(t, "exact_match", "A")) for t in ["t1", "t2", "t3"]) + "\n"
    )
    (root / "metadata.yaml").write_text(
        yaml.safe_dump({"name": "toy", "description": "fixture"})
    )
    return root


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_first_upload_creates_v1(bench_dir: Path):
    store = _FakeStore()
    result = upload_benchmark(store, bench_dir)

    assert isinstance(result, UploadResult)
    assert result.status == "created"
    assert result.version == 1
    assert result.n_tasks == 3
    assert result.name == "toy"
    assert result.content_hash and len(result.content_hash) == 64

    # Store calls
    assert len(store.benchmarks) == 1
    rec = store.benchmarks[0]
    assert rec["name"] == "toy"
    assert rec["format"] == BENCHMARK_FORMAT
    assert rec["version"] == 1
    assert rec["metadata"]["description"] == "fixture"
    assert rec["metadata"]["content_hash"] == result.content_hash

    # All 3 tasks pushed
    pushed = store.rows_by_id[result.benchmark_id]
    assert len(pushed) == 3
    assert pushed[0]["task_id"] == "t1"
    assert pushed[0]["verifier"]["kind"] == "in_process"


def test_idempotent_reupload_returns_unchanged(bench_dir: Path):
    store = _FakeStore()
    first = upload_benchmark(store, bench_dir)
    second = upload_benchmark(store, bench_dir)

    assert second.status == "unchanged"
    assert second.benchmark_id == first.benchmark_id
    assert second.version == 1
    # Store was NOT called to create a new record or push new rows.
    assert len(store.benchmarks) == 1
    assert len(store.rows_by_id[first.benchmark_id]) == 3  # unchanged


def test_changed_content_creates_new_version(bench_dir: Path):
    store = _FakeStore()
    first = upload_benchmark(store, bench_dir)
    # Modify tasks.jsonl — bump from 3 to 4 tasks.
    (bench_dir / "tasks.jsonl").write_text(
        "\n".join(json.dumps(_row(t, "exact_match", "X")) for t in
                  ["t1", "t2", "t3", "t4"]) + "\n"
    )
    second = upload_benchmark(store, bench_dir)

    assert second.status == "updated"
    assert second.version == 2
    assert second.benchmark_id != first.benchmark_id
    assert second.content_hash != first.content_hash
    assert second.n_tasks == 4

    # Two benchmark records exist (different versions).
    assert len(store.benchmarks) == 2
    assert {b["version"] for b in store.benchmarks} == {1, 2}


def test_third_upload_increments_version(bench_dir: Path):
    store = _FakeStore()
    upload_benchmark(store, bench_dir)
    # Change once
    (bench_dir / "tasks.jsonl").write_text(json.dumps(_row("t1", "exact_match", "X")) + "\n")
    upload_benchmark(store, bench_dir)
    # Change twice
    (bench_dir / "tasks.jsonl").write_text(json.dumps(_row("t1", "exact_match", "Y")) + "\n")
    third = upload_benchmark(store, bench_dir)

    assert third.version == 3


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------


def test_hash_is_stable_across_runs(bench_dir: Path):
    store_a, store_b = _FakeStore(), _FakeStore()
    r1 = upload_benchmark(store_a, bench_dir)
    r2 = upload_benchmark(store_b, bench_dir)
    assert r1.content_hash == r2.content_hash


def test_hash_changes_when_content_changes(bench_dir: Path):
    store = _FakeStore()
    r1 = upload_benchmark(store, bench_dir)
    (bench_dir / "tasks.jsonl").write_text(json.dumps(_row("x", "exact_match", "z")) + "\n")
    r2 = upload_benchmark(store, bench_dir)
    assert r1.content_hash != r2.content_hash


# ---------------------------------------------------------------------------
# Validation surface — leans on Benchmark.from_dir
# ---------------------------------------------------------------------------


def test_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        upload_benchmark(_FakeStore(), tmp_path / "nope")


def test_missing_tasks_jsonl(tmp_path: Path):
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(FileNotFoundError, match="missing tasks.jsonl"):
        upload_benchmark(_FakeStore(), bad)


def test_malformed_jsonl_surfaces_via_value_error(tmp_path: Path):
    bad = tmp_path / "bad_jsonl"
    bad.mkdir()
    (bad / "tasks.jsonl").write_text("not json\n")
    with pytest.raises(ValueError):
        upload_benchmark(_FakeStore(), bad)


def test_empty_benchmark_creates_record_with_no_rows(tmp_path: Path):
    """tasks.jsonl with zero rows is allowed (empty suite). No rows pushed."""
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "tasks.jsonl").write_text("")
    (empty / "metadata.yaml").write_text(yaml.safe_dump({"name": "empty"}))

    store = _FakeStore()
    result = upload_benchmark(store, empty)
    assert result.status == "created"
    assert result.n_tasks == 0
    assert store.rows_by_id.get(result.benchmark_id, []) == []


# ---------------------------------------------------------------------------
# Store flakiness on list_benchmarks
# ---------------------------------------------------------------------------


class _ListThrowingStore(_FakeStore):
    """Simulates a backend that's down on list — we should still create."""

    def list_benchmarks(self, project_id: str | None = None) -> list[dict]:
        raise RuntimeError("list failed")


def test_list_failure_falls_through_to_create(bench_dir: Path):
    store = _ListThrowingStore()
    result = upload_benchmark(store, bench_dir)
    assert result.status == "created"
    assert result.version == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_benchmark_upload_happy(bench_dir: Path, monkeypatch, capsys):
    fake_store = _FakeStore()

    def _fake_ctor(*args, **kwargs):
        return fake_store

    monkeypatch.setattr("evsys_sdk.store.EvsysStore", _fake_ctor)
    rc = cli_main(["benchmark", "upload", str(bench_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    payload_line = out.split("\n\n")[0]
    payload = json.loads(payload_line)
    assert payload["status"] == "created"
    assert payload["n_tasks"] == 3
    assert "bench-1" in out  # benchmark_id format from fake store
    assert "metadata.benchmark.id" in out


def test_cli_benchmark_upload_missing_dir(tmp_path: Path, monkeypatch, capsys):
    fake_store = _FakeStore()
    monkeypatch.setattr("evsys_sdk.store.EvsysStore",
                        lambda *a, **kw: fake_store)
    rc = cli_main(["benchmark", "upload", str(tmp_path / "nope")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR" in err


def test_cli_benchmark_upload_malformed(tmp_path: Path, monkeypatch, capsys):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "tasks.jsonl").write_text("nope\n")
    fake_store = _FakeStore()
    monkeypatch.setattr("evsys_sdk.store.EvsysStore",
                        lambda *a, **kw: fake_store)
    rc = cli_main(["benchmark", "upload", str(bad)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR" in err

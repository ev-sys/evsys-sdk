"""Tests for `trajectory_labs.step_metrics.forward_step_metrics`.

The forwarder replaces hand-rolled `backfill_step_metrics` loops in
researcher scripts: it reads a local metrics.jsonl and pushes each row to
a TrajectoryStore.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trajectory_labs.step_metrics import forward_step_metrics


class _RecordingStore:
    def __init__(self, *, raise_on: set[int] | None = None) -> None:
        self.calls: list[dict] = []
        self._raise_on = raise_on or set()

    def log_metrics(self, *, run_id: str, step: int, metrics: dict[str, float]) -> dict:
        if step in self._raise_on:
            raise RuntimeError(f"flaky upload at step {step}")
        self.calls.append({"run_id": run_id, "step": step, "metrics": dict(metrics)})
        return {"ok": True}


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_single_row_one_call(tmp_path: Path):
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl",
                 [{"ts": 0, "step": 1, "metrics": {"loss": 0.5}}])
    store = _RecordingStore()
    sent = forward_step_metrics(store, "run-A", tmp_path)
    assert sent == 1
    assert store.calls == [{"run_id": "run-A", "step": 1, "metrics": {"loss": 0.5}}]


def test_multiple_rows_preserve_order(tmp_path: Path):
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl", [
        {"ts": 0, "step": 1, "metrics": {"loss": 0.9}},
        {"ts": 1, "step": 2, "metrics": {"loss": 0.5}},
        {"ts": 2, "step": 3, "metrics": {"loss": 0.1}},
    ])
    store = _RecordingStore()
    sent = forward_step_metrics(store, "run-A", tmp_path)
    assert sent == 3
    assert [c["step"] for c in store.calls] == [1, 2, 3]


def test_step_coerced_to_int(tmp_path: Path):
    """metrics.jsonl might carry step as a float; the store wants an int."""
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl",
                 [{"ts": 0, "step": 10.0, "metrics": {"loss": 0.5}}])
    store = _RecordingStore()
    forward_step_metrics(store, "run-A", tmp_path)
    assert store.calls[0]["step"] == 10
    assert isinstance(store.calls[0]["step"], int)


# ---------------------------------------------------------------------------
# Location resolution
# ---------------------------------------------------------------------------


def test_finds_metrics_at_run_dir_root(tmp_path: Path):
    """Algorithms that don't wrap in logs/ should still be discovered."""
    _write_jsonl(tmp_path / "metrics.jsonl",
                 [{"ts": 0, "step": 1, "metrics": {"x": 1.0}}])
    store = _RecordingStore()
    sent = forward_step_metrics(store, "run-A", tmp_path)
    assert sent == 1


def test_finds_nested_metrics_recursively(tmp_path: Path):
    """Multiplex log stores nest under <run>/logs/jsonl/metrics.jsonl."""
    _write_jsonl(tmp_path / "logs" / "jsonl" / "metrics.jsonl",
                 [{"ts": 0, "step": 1, "metrics": {"x": 1.0}}])
    store = _RecordingStore()
    sent = forward_step_metrics(store, "run-A", tmp_path)
    assert sent == 1


def test_prefers_shallowest_path(tmp_path: Path):
    """When the metrics file exists at the canonical <run>/logs/ path AND in
    a deeper nested location, the canonical one is taken."""
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl",
                 [{"ts": 0, "step": 1, "metrics": {"chosen": 1.0}}])
    _write_jsonl(tmp_path / "logs" / "jsonl" / "metrics.jsonl",
                 [{"ts": 0, "step": 1, "metrics": {"NOT_chosen": 1.0}}])
    store = _RecordingStore()
    forward_step_metrics(store, "run-A", tmp_path)
    assert "chosen" in store.calls[0]["metrics"]


# ---------------------------------------------------------------------------
# Skip / no-op semantics
# ---------------------------------------------------------------------------


def test_none_store_returns_zero(tmp_path: Path):
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl",
                 [{"ts": 0, "step": 1, "metrics": {"x": 1}}])
    assert forward_step_metrics(None, "run-A", tmp_path) == 0


def test_none_run_id_returns_zero(tmp_path: Path):
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl",
                 [{"ts": 0, "step": 1, "metrics": {"x": 1}}])
    assert forward_step_metrics(_RecordingStore(), None, tmp_path) == 0


def test_none_run_dir_returns_zero():
    assert forward_step_metrics(_RecordingStore(), "run-A", None) == 0


def test_missing_metrics_file_returns_zero(tmp_path: Path):
    # run_dir exists, but no metrics.jsonl anywhere
    (tmp_path / "logs").mkdir()
    assert forward_step_metrics(_RecordingStore(), "run-A", tmp_path) == 0


def test_run_dir_not_a_directory_returns_zero(tmp_path: Path):
    p = tmp_path / "not_a_dir"
    p.write_text("file")
    assert forward_step_metrics(_RecordingStore(), "run-A", p) == 0


# ---------------------------------------------------------------------------
# Skip-bad-row semantics
# ---------------------------------------------------------------------------


def test_malformed_jsonl_lines_skipped(tmp_path: Path):
    p = tmp_path / "logs" / "metrics.jsonl"
    p.parent.mkdir()
    p.write_text(
        json.dumps({"ts": 0, "step": 1, "metrics": {"loss": 0.5}}) + "\n"
        + "not json\n"
        + json.dumps({"ts": 1, "step": 2, "metrics": {"loss": 0.4}}) + "\n"
    )
    store = _RecordingStore()
    sent = forward_step_metrics(store, "run-A", tmp_path)
    assert sent == 2
    assert [c["step"] for c in store.calls] == [1, 2]


def test_blank_lines_skipped(tmp_path: Path):
    p = tmp_path / "logs" / "metrics.jsonl"
    p.parent.mkdir()
    p.write_text(
        "\n"
        + json.dumps({"ts": 0, "step": 1, "metrics": {"loss": 0.5}}) + "\n"
        + "\n\n"
    )
    assert forward_step_metrics(_RecordingStore(), "run-A", tmp_path) == 1


def test_rows_missing_step_skipped(tmp_path: Path):
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl", [
        {"ts": 0, "metrics": {"loss": 0.5}},               # no step
        {"ts": 1, "step": 1, "metrics": {"loss": 0.4}},
    ])
    store = _RecordingStore()
    sent = forward_step_metrics(store, "run-A", tmp_path)
    assert sent == 1
    assert store.calls[0]["step"] == 1


def test_rows_missing_metrics_skipped(tmp_path: Path):
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl", [
        {"ts": 0, "step": 1},
        {"ts": 1, "step": 2, "metrics": {}},  # empty dict
        {"ts": 2, "step": 3, "metrics": {"loss": 0.4}},
    ])
    store = _RecordingStore()
    sent = forward_step_metrics(store, "run-A", tmp_path)
    assert sent == 1


def test_rows_with_non_dict_metrics_skipped(tmp_path: Path):
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl", [
        {"ts": 0, "step": 1, "metrics": "not a dict"},
        {"ts": 1, "step": 2, "metrics": {"loss": 0.4}},
    ])
    store = _RecordingStore()
    sent = forward_step_metrics(store, "run-A", tmp_path)
    assert sent == 1


# ---------------------------------------------------------------------------
# Flaky store
# ---------------------------------------------------------------------------


def test_per_row_store_failure_swallowed_other_rows_continue(tmp_path: Path):
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl", [
        {"ts": 0, "step": 1, "metrics": {"loss": 0.9}},
        {"ts": 1, "step": 2, "metrics": {"loss": 0.5}},
        {"ts": 2, "step": 3, "metrics": {"loss": 0.1}},
    ])
    store = _RecordingStore(raise_on={2})
    sent = forward_step_metrics(store, "run-A", tmp_path)
    # step 1 and 3 land; step 2 raises and is skipped.
    assert sent == 2
    assert [c["step"] for c in store.calls] == [1, 3]


# ---------------------------------------------------------------------------
# Custom metrics_file argument
# ---------------------------------------------------------------------------


def test_custom_metrics_file_name(tmp_path: Path):
    _write_jsonl(tmp_path / "logs" / "train_metrics.jsonl",
                 [{"ts": 0, "step": 1, "metrics": {"loss": 0.5}}])
    store = _RecordingStore()
    sent = forward_step_metrics(store, "run-A", tmp_path,
                                metrics_file="train_metrics.jsonl")
    assert sent == 1

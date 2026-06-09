"""Tests for `evsys_sdk.step_metrics.forward_step_metrics`.

The forwarder replaces hand-rolled `backfill_step_metrics` loops in
researcher scripts: it reads a local metrics.jsonl and pushes each row to
a EvsysStore.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evsys_sdk.step_metrics import forward_step_metrics


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


# ---------------------------------------------------------------------------
# Flat tinker_cookbook shape (no nested ``metrics`` key)
# ---------------------------------------------------------------------------


class _SplitRecordingStore:
    """Like _RecordingStore but accepts the optional `split` kwarg used for
    val-prefixed metrics, so val-routing tests can introspect both buckets."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def log_metrics(self, *, run_id: str, step: int, metrics: dict[str, float],
                    split: str = "train") -> dict:
        self.calls.append({"run_id": run_id, "step": step,
                           "metrics": dict(metrics), "split": split})
        return {"ok": True}


def test_forwards_flat_tinker_cookbook_format(tmp_path: Path):
    """tinker_cookbook writes flat rows where metric keys sit at the top
    level alongside `step`. They must forward as a normalized metrics dict."""
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl", [
        {"step": 0, "epoch": 0, "progress": 0.05,
         "learning_rate": 1e-4, "train_mean_nll": 2.09,
         "num_sequences": 16, "num_tokens": 1900,
         "time/get_batch": 3e-06, "time/step": 18.17},
    ])
    store = _RecordingStore()
    sent = forward_step_metrics(store, "run-A", tmp_path)
    assert sent == 1
    call = store.calls[0]
    assert call["step"] == 0
    assert call["metrics"]["learning_rate"] == 1e-4
    assert call["metrics"]["train_mean_nll"] == 2.09
    assert call["metrics"]["time/step"] == 18.17


def test_flat_row_excludes_meta_keys(tmp_path: Path):
    """`step`, `epoch`, `progress`, `ts`, `timestamp` are positional meta
    fields, not metric values; they must not leak into the metrics dict."""
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl", [
        {"step": 5, "epoch": 0, "progress": 0.5, "ts": 1700000000.0,
         "timestamp": 42, "loss": 0.42},
    ])
    store = _RecordingStore()
    forward_step_metrics(store, "run-A", tmp_path)
    assert store.calls[0]["metrics"] == {"loss": 0.42}


def test_flat_row_skips_non_numeric_fields(tmp_path: Path):
    """String- and dict-valued top-level fields must not appear in the
    forwarded metrics — only numerics make it through."""
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl", [
        {"step": 1, "loss": 0.5, "tag": "smoke", "nested": {"x": 1}, "flag": True},
    ])
    store = _RecordingStore()
    forward_step_metrics(store, "run-A", tmp_path)
    # bool is a subclass of int in Python — flag does technically pass the
    # isinstance check, so we accept it. The point of this test is that
    # `tag` (str) and `nested` (dict) are excluded.
    m = store.calls[0]["metrics"]
    assert "tag" not in m
    assert "nested" not in m
    assert m["loss"] == 0.5


def test_flat_row_with_only_meta_keys_skipped(tmp_path: Path):
    """If a row has only positional fields (no numeric metric values), it
    must be a silent no-op — no log_metrics call."""
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl", [
        {"step": 1, "epoch": 0, "progress": 0.5},
        {"step": 2, "loss": 0.5},
    ])
    store = _RecordingStore()
    sent = forward_step_metrics(store, "run-A", tmp_path)
    assert sent == 1
    assert store.calls[0]["step"] == 2


def test_val_prefix_split_works_on_flat_format(tmp_path: Path):
    """A flat row with `val/`-prefixed keys must still route to split='val'."""
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl", [
        {"step": 10, "train_mean_nll": 0.42, "val/exact_match": 0.7},
    ])
    store = _SplitRecordingStore()
    sent = forward_step_metrics(store, "run-A", tmp_path)
    # one train call (train_mean_nll) + one val call (val/exact_match)
    assert sent == 2
    by_split = {c["split"]: c["metrics"] for c in store.calls}
    assert by_split["train"] == {"train_mean_nll": 0.42}
    assert by_split["val"] == {"val/exact_match": 0.7}


def test_forwards_nested_format_unchanged(tmp_path: Path):
    """Regression: the original nested shape must continue to forward
    untouched, even though _extract_metrics now handles both."""
    _write_jsonl(tmp_path / "logs" / "metrics.jsonl",
                 [{"ts": 0, "step": 1, "metrics": {"loss": 0.5, "lr": 1e-4}}])
    store = _RecordingStore()
    sent = forward_step_metrics(store, "run-A", tmp_path)
    assert sent == 1
    assert store.calls[0]["metrics"] == {"loss": 0.5, "lr": 1e-4}

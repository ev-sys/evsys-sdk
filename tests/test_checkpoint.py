"""Tests for `evsys_sdk.checkpoint` — parse algorithm-emitted
`checkpoints.jsonl` manifests so researcher scripts don't hand-roll it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evsys_sdk.checkpoint import (
    MANIFEST_NAME,
    Checkpoint,
    find_manifest,
    read_manifest,
)

# ---------------------------------------------------------------------------
# Checkpoint.from_manifest_row
# ---------------------------------------------------------------------------


def test_from_manifest_row_full():
    row = {"name": "final", "batch": 1520, "epoch": 10,
           "state_path": "tinker://state", "sampler_path": "tinker://sampler"}
    c = Checkpoint.from_manifest_row(row)
    assert c.label == "final"
    assert c.step == 1520
    assert c.epoch == 10
    assert c.weights_path == "tinker://state"
    assert c.sampler_path == "tinker://sampler"
    assert c.raw == row
    assert c.has_path


def test_from_manifest_row_minimal():
    c = Checkpoint.from_manifest_row({"name": "epoch-1"})
    assert c.label == "epoch-1"
    assert c.step is None
    assert c.epoch is None
    assert c.weights_path is None
    assert c.sampler_path is None
    assert not c.has_path


def test_from_manifest_row_missing_name_defaults():
    c = Checkpoint.from_manifest_row({"batch": 50})
    assert c.label == "?"
    assert c.step == 50


def test_from_manifest_row_int_coercion_handles_strings():
    c = Checkpoint.from_manifest_row({"name": "x", "batch": "100", "epoch": "3"})
    assert c.step == 100
    assert c.epoch == 3


def test_from_manifest_row_unparseable_ints_are_none():
    c = Checkpoint.from_manifest_row({"name": "x", "batch": "notanumber"})
    assert c.step is None


def test_from_manifest_row_empty_paths_are_none():
    c = Checkpoint.from_manifest_row({"name": "x", "state_path": ""})
    assert c.weights_path is None


# ---------------------------------------------------------------------------
# read_manifest
# ---------------------------------------------------------------------------


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_read_manifest_single_row(tmp_path: Path):
    p = _write_manifest(tmp_path / MANIFEST_NAME, [
        {"name": "final", "batch": 10, "state_path": "x"},
    ])
    checkpoints = read_manifest(p)
    assert len(checkpoints) == 1
    assert checkpoints[0].label == "final"
    assert checkpoints[0].step == 10


def test_read_manifest_preserves_order(tmp_path: Path):
    p = _write_manifest(tmp_path / MANIFEST_NAME, [
        {"name": "epoch-1", "batch": 50, "state_path": "a"},
        {"name": "epoch-2", "batch": 100, "state_path": "b"},
        {"name": "final",   "batch": 150, "state_path": "c"},
    ])
    cs = read_manifest(p)
    assert [c.label for c in cs] == ["epoch-1", "epoch-2", "final"]
    assert [c.step for c in cs] == [50, 100, 150]


def test_read_manifest_blank_lines_skipped(tmp_path: Path):
    p = tmp_path / MANIFEST_NAME
    p.write_text(
        "\n"
        + json.dumps({"name": "a", "batch": 1, "state_path": "x"}) + "\n"
        + "\n\n"
        + json.dumps({"name": "b", "batch": 2, "state_path": "y"}) + "\n"
        + "\n"
    )
    cs = read_manifest(p)
    assert [c.label for c in cs] == ["a", "b"]


def test_read_manifest_empty_file(tmp_path: Path):
    p = tmp_path / MANIFEST_NAME
    p.write_text("")
    assert read_manifest(p) == []


def test_read_manifest_malformed_jsonl_raises(tmp_path: Path):
    p = tmp_path / MANIFEST_NAME
    p.write_text(json.dumps({"name": "a"}) + "\nnot json\n")
    with pytest.raises(ValueError, match="malformed jsonl"):
        read_manifest(p)


def test_read_manifest_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# find_manifest
# ---------------------------------------------------------------------------


def test_find_manifest_at_root(tmp_path: Path):
    p = _write_manifest(tmp_path / MANIFEST_NAME, [{"name": "final"}])
    assert find_manifest(tmp_path) == p


def test_find_manifest_nested(tmp_path: Path):
    (tmp_path / "sub" / "run-abc").mkdir(parents=True)
    p = _write_manifest(tmp_path / "sub" / "run-abc" / MANIFEST_NAME,
                        [{"name": "final"}])
    assert find_manifest(tmp_path) == p


def test_find_manifest_prefers_shallowest(tmp_path: Path):
    """When multiple manifests exist, the shallowest one wins — that's
    where the orchestrator's final manifest lives."""
    shallow = _write_manifest(tmp_path / MANIFEST_NAME, [{"name": "shallow"}])
    (tmp_path / "deep").mkdir()
    _write_manifest(tmp_path / "deep" / MANIFEST_NAME, [{"name": "deep"}])
    assert find_manifest(tmp_path) == shallow


def test_find_manifest_returns_none_when_absent(tmp_path: Path):
    assert find_manifest(tmp_path) is None


def test_find_manifest_returns_none_when_dir_missing(tmp_path: Path):
    assert find_manifest(tmp_path / "does_not_exist") is None


# ---------------------------------------------------------------------------
# Checkpoint.pick_final
# ---------------------------------------------------------------------------


def test_pick_final_with_named_final():
    cs = [
        Checkpoint(label="epoch-1", weights_path="a"),
        Checkpoint(label="final", weights_path="b"),
        Checkpoint(label="epoch-3", weights_path="c"),
    ]
    assert Checkpoint.pick_final(cs).label == "final"


def test_pick_final_falls_back_to_last_with_path():
    cs = [
        Checkpoint(label="epoch-1", weights_path="a"),
        Checkpoint(label="epoch-2", weights_path="b"),
        Checkpoint(label="epoch-3"),  # no path — skipped
    ]
    assert Checkpoint.pick_final(cs).label == "epoch-2"


def test_pick_final_returns_none_when_no_paths():
    cs = [
        Checkpoint(label="epoch-1"),
        Checkpoint(label="epoch-2"),
    ]
    assert Checkpoint.pick_final(cs) is None


def test_pick_final_empty_list_returns_none():
    assert Checkpoint.pick_final([]) is None


def test_pick_final_uses_sampler_path_too():
    """Tinker SFT exposes sampler_path; that should count as 'has path' too."""
    cs = [Checkpoint(label="final", sampler_path="tinker://sampler")]
    assert Checkpoint.pick_final(cs).label == "final"

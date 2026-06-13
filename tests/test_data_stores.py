"""Data store implementations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evsys_sdk.data_stores.in_memory import InMemoryDataStore
from evsys_sdk.data_stores.local import LocalDataStore


def test_in_memory_jsonl_roundtrip():
    s = InMemoryDataStore()
    s.write_jsonl("a.jsonl", [{"x": 1}, {"x": 2}])
    assert s.read_jsonl("a.jsonl") == [{"x": 1}, {"x": 2}]
    assert s.exists("a.jsonl")
    assert not s.exists("b.jsonl")


def test_in_memory_json_roundtrip():
    s = InMemoryDataStore()
    s.write_json("c.json", {"k": "v"})
    assert s.read_json("c.json") == {"k": "v"}


def test_in_memory_missing_raises():
    s = InMemoryDataStore()
    with pytest.raises(FileNotFoundError):
        s.read_jsonl("missing")


def test_local_jsonl_roundtrip(tmp_path: Path):
    s = LocalDataStore(root=tmp_path)
    s.write_jsonl("data/a.jsonl", [{"x": 1}, {"x": 2}])
    assert (tmp_path / "data" / "a.jsonl").exists()
    assert s.read_jsonl("data/a.jsonl") == [{"x": 1}, {"x": 2}]


def test_local_json_roundtrip(tmp_path: Path):
    s = LocalDataStore(root=tmp_path)
    s.write_json("d/c.json", {"k": "v"})
    assert s.read_json("d/c.json") == {"k": "v"}


def test_local_absolute_path_works(tmp_path: Path):
    s = LocalDataStore(root=tmp_path)
    p = tmp_path / "abs.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n")
    assert s.read_jsonl(str(p)) == [{"a": 1}]


def test_local_list(tmp_path: Path):
    s = LocalDataStore(root=tmp_path)
    s.write_jsonl("a/x.jsonl", [{}])
    s.write_jsonl("a/y.jsonl", [{}])
    s.write_jsonl("b/z.jsonl", [{}])
    assert sorted(p for p in s.list("a") if p.endswith(".jsonl")) == ["a/x.jsonl", "a/y.jsonl"]

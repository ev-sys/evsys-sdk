"""Context ingestion — model, directory adapter, store/cursor dedupe, and the
unified daemon that pulls traces AND context together."""

from __future__ import annotations

import os

from evsys_sdk import ContextItem
from evsys_sdk.config import SystemConfig
from evsys_sdk.context_sources import LocalContextStore, build_context_sources, run_context_pull


def _corpus(root):
    (root / "user-a").mkdir(parents=True)
    (root / "user-a" / "m1.txt").write_text("The export button is broken on billing.")
    (root / "user-b").mkdir(parents=True)
    (root / "user-b" / "note.txt").write_text("Prefers concise answers.")


def _spec(path, state_dir, **extra):
    return SystemConfig(context={"context_sources": [
        {"kind": "directory", "params": {"path": str(path), **extra},
         "state_dir": str(state_dir)}]}).context.context_sources


def test_context_item_roundtrip():
    it = ContextItem("i1", "directory", "hi", entity="user-a", metadata={"k": "v"})
    d = it.to_dict()
    assert d == {"item_id": "i1", "source": "directory",
                 "content": "hi", "entity": "user-a", "metadata": {"k": "v"}}


def test_directory_adapter_maps_files(tmp_path):
    _corpus(tmp_path / "ctx")
    (_, src), = build_context_sources(_spec(tmp_path / "ctx", tmp_path / ".evsys"))
    src.run_once()
    items = [__import__("json").loads(x) for x in
             (tmp_path / ".evsys" / "directory" / "items.jsonl").read_text().splitlines()]
    by_entity = {i["entity"]: i for i in items}
    assert set(by_entity) == {"user-a", "user-b"}                # folder → entity
    assert "export button" in by_entity["user-a"]["content"]     # body → content


def test_store_and_cursor_dedupe(tmp_path):
    _corpus(tmp_path / "ctx")
    specs = _spec(tmp_path / "ctx", tmp_path / ".evsys")
    assert run_context_pull(specs) == 2           # first pull lands both
    assert run_context_pull(specs) == 0           # nothing new → cursor dedupes
    (tmp_path / "ctx" / "user-c").mkdir()
    (tmp_path / "ctx" / "user-c" / "x.txt").write_text("a new note")
    assert run_context_pull(specs) == 1           # only the new item


def test_incremental_since_skips_old(tmp_path):
    _corpus(tmp_path / "ctx")
    store = LocalContextStore(tmp_path / ".evsys")
    (_, src), = build_context_sources(_spec(tmp_path / "ctx", tmp_path / ".evsys"), store=store)
    from datetime import UTC, datetime
    future = datetime.now(UTC).replace(year=2999)
    assert src.run_once(since_override=future) == 0  # everything older than `since` skipped


def test_unified_pull_gets_traces_and_context(tmp_path):
    """`run_all_once` pulls every trace source AND every context source."""
    from evsys_sdk.ingest import run_all_once
    from evsys_sdk.registry import _trace_sources, register_trace_source
    from evsys_sdk.trace_sources.base import BaseTraceSource

    @register_trace_source("fake_trace")
    class FakeTrace(BaseTraceSource):
        name = "fake_trace"
        Config = None

        def pull_raw(self, since):
            return []

        def to_trace(self, raw):  # never called
            raise AssertionError

    try:
        _corpus(tmp_path / "ctx")
        cfg = SystemConfig(
            traces={"trace_sources": [{"kind": "fake_trace", "state_dir": str(tmp_path / ".t")}]},
            context={"context_sources": [{"kind": "directory",
                                          "params": {"path": str(tmp_path / "ctx")},
                                          "state_dir": str(tmp_path / ".c")}]},
        )
        res = run_all_once(cfg)
        assert res == {"trace:fake_trace": 0, "context:directory": 2}
    finally:
        _trace_sources.unregister("fake_trace")


def test_directory_source_registered():
    from evsys_sdk.registry import list_context_sources

    assert "directory" in list_context_sources()


def test_live_directory_env_gated(tmp_path):
    if not os.environ.get("EVSYS_CONTEXT_LIVE"):
        import pytest
        pytest.skip("set EVSYS_CONTEXT_LIVE to run the live directory pull")
    _corpus(tmp_path / "ctx")
    assert run_context_pull(_spec(tmp_path / "ctx", tmp_path / ".evsys")) == 2

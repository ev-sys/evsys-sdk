"""Workspace local-cache tests (fake store — no network)."""

from __future__ import annotations

import json

from evsys_sdk import Workspace


class FakeStore:
    """Minimal store: one dataset of `n` raw rows, with call counters."""

    def __init__(self, n=1200, version=1, fmt="chat_messages",
                 transform=[{"kind": "jsonl_to_chat", "params": {}}]):
        self._rows = [{"idx": i, "payload": {"q": f"row{i}"}} for i in range(n)]
        self._meta = {"id": "ds1", "version": version, "n_rows": n,
                      "format": fmt, "transform": transform}
        self.meta_calls = 0
        self.row_calls = 0

    def get_dataset(self, dataset_id):
        self.meta_calls += 1
        return dict(self._meta) if dataset_id == "ds1" else None

    def get_dataset_rows(self, dataset_id, *, limit=100, offset=0):
        self.row_calls += 1
        return self._rows[offset:offset + limit]


def test_pull_writes_jsonl_and_manifest(tmp_path):
    store = FakeStore(n=1200)
    ws = Workspace(store, root=str(tmp_path / "wsp"))
    mat = ws.pull_dataset("ds1")

    assert mat.cached is False
    assert mat.n_rows == 1200
    assert mat.format == "chat_messages"
    assert mat.transform == [{"kind": "jsonl_to_chat", "params": {}}]

    lines = open(mat.path).read().splitlines()
    assert len(lines) == 1200                                   # paged across 3×500
    assert json.loads(lines[0]) == {"q": "row0"}                # raw payload only
    assert store.row_calls >= 3                                  # 1200/500 → 3 pages

    man = json.loads(open(mat.path.replace(".jsonl", ".meta.json")).read())
    assert man["complete"] is True and man["n_rows"] == 1200


def test_self_ignoring_gitignore_written(tmp_path):
    root = tmp_path / "wsp"
    Workspace(FakeStore(), root=str(root))
    assert (root / ".gitignore").read_text().strip() == "*"
    for sub in ("datasets", "benchmarks", "scripts", "outputs"):
        assert (root / sub).is_dir()


def test_second_pull_is_cache_hit(tmp_path):
    store = FakeStore(n=10)
    ws = Workspace(store, root=str(tmp_path / "wsp"))
    ws.pull_dataset("ds1")
    rows_after_first = store.row_calls

    mat2 = ws.pull_dataset("ds1")
    assert mat2.cached is True
    assert store.row_calls == rows_after_first                  # no re-fetch


def test_force_repulls(tmp_path):
    store = FakeStore(n=10)
    ws = Workspace(store, root=str(tmp_path / "wsp"))
    ws.pull_dataset("ds1")
    n = store.row_calls
    mat = ws.pull_dataset("ds1", force=True)
    assert mat.cached is False
    assert store.row_calls > n                                  # fetched again


def test_nrows_mismatch_invalidates_cache(tmp_path):
    store = FakeStore(n=10)
    ws = Workspace(store, root=str(tmp_path / "wsp"))
    mat = ws.pull_dataset("ds1")
    # Corrupt the manifest's n_rows → next pull must re-fetch.
    man_path = mat.path.replace(".jsonl", ".meta.json")
    man = json.loads(open(man_path).read()); man["n_rows"] = 999
    open(man_path, "w").write(json.dumps(man))
    n = store.row_calls
    mat2 = ws.pull_dataset("ds1")
    assert mat2.cached is False and store.row_calls > n


def test_missing_dataset_raises(tmp_path):
    ws = Workspace(FakeStore(), root=str(tmp_path / "wsp"))
    try:
        ws.pull_dataset("nope")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_standard_locations(tmp_path):
    ws = Workspace(FakeStore(), root=str(tmp_path / "wsp"))
    assert ws.script_path("e1").endswith("scripts/exp_e1.py")
    out = ws.outputs_dir("r1")
    assert out.endswith("outputs/r1")
    import os
    assert os.path.isdir(out)

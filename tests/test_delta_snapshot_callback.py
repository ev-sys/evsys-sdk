"""Tests for the delta_snapshot callback -- the node half of the router."""
import json
from pathlib import Path

import pytest

from evsys_sdk.checkpoint_delta import load_delta
from evsys_sdk.training.callbacks import DeltaSnapshotCallback, build_callbacks
from evsys_sdk.training.checkpoints import ManifestRow


class _State:
    def __init__(self, tmp):
        self.step = 0
        self.output_dir = Path(tmp)


def _write_ckpt(tmp, name, payload: bytes):
    p = Path(tmp) / name
    p.write_bytes(payload)
    return str(p)


def test_ships_base_and_delta_and_reports(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr("evsys_sdk.compute.events_topic.post_event",
                        lambda url, ev, transport=None: events.append(ev) or True)
    cb = DeltaSnapshotCallback(store_dir=str(tmp_path / "store"),
                               events_url="http://t/x", job_id="job1",
                               volume="vol-9")
    st = _State(tmp_path)
    p1 = _write_ckpt(tmp_path, "ck1.bin", b"A" * 1000)
    cb.on_checkpoint(st, ManifestRow(name="ck1", batch=100, state_path=p1))
    p2 = _write_ckpt(tmp_path, "ck2.bin", b"A" * 990 + b"B" * 10)
    # same filename convention: row 2 is a new file but same size
    p2 = _write_ckpt(tmp_path, "ck1.bin", b"A" * 990 + b"B" * 10)
    cb.on_checkpoint(st, ManifestRow(name="ck2", batch=200, state_path=p2))
    store = tmp_path / "store"
    assert (store / "base.evd").exists()
    assert (store / "step-100.evd").exists()
    assert (store / "step-200.evd").exists()
    assert len(events) == 2
    ev = events[-1]
    assert ev["kind"] == "checkpoint" and ev["step"] == 200
    assert ev["store"]["volume"] == "vol-9"
    assert set(ev["sha256"]) == {"base.evd", "step-200.evd"}
    # the delta reconstructs the exact checkpoint bytes
    recon = load_delta(str(store / "base.evd"), str(store / "step-200.evd"))
    assert bytes(recon["ck1.bin"].tobytes()) == b"A" * 990 + b"B" * 10


def test_rebases_when_file_size_changes(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr("evsys_sdk.compute.events_topic.post_event",
                        lambda url, ev, transport=None: events.append(ev) or True)
    cb = DeltaSnapshotCallback(store_dir=str(tmp_path / "store"),
                               events_url="http://t/x", job_id="j")
    st = _State(tmp_path)
    p = _write_ckpt(tmp_path, "w.bin", b"x" * 100)
    cb.on_checkpoint(st, ManifestRow(name="c1", batch=1, state_path=p))
    p = _write_ckpt(tmp_path, "w.bin", b"y" * 500)      # grew: shapes mismatch
    cb.on_checkpoint(st, ManifestRow(name="c2", batch=2, state_path=p))
    assert events[-1]["base_key"] == "base-2.evd"       # re-based


def test_done_event_on_train_end(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr("evsys_sdk.compute.events_topic.post_event",
                        lambda url, ev, transport=None: events.append(ev) or True)
    cb = DeltaSnapshotCallback(store_dir=str(tmp_path / "s"),
                               events_url="http://t/x", job_id="j")
    cb.on_train_end(_State(tmp_path), None)
    assert events == [{"kind": "done", "job_id": "j"}]


def test_no_events_without_url(tmp_path):
    cb = DeltaSnapshotCallback(store_dir=str(tmp_path / "s"))
    st = _State(tmp_path)
    p = _write_ckpt(tmp_path, "w.bin", b"z" * 64)
    cb.on_checkpoint(st, ManifestRow(name="c", batch=1, state_path=p))
    assert (tmp_path / "s" / "step-1.evd").exists()     # still checkpoints


def test_registered_in_registry():
    cbs = build_callbacks([{"kind": "delta_snapshot",
                            "params": {"store_dir": "/tmp/x"}}])
    assert isinstance(cbs[0], DeltaSnapshotCallback)


def test_remote_scheme_checkpoint_reports_without_encoding(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr("evsys_sdk.compute.events_topic.post_event",
                        lambda url, ev, transport=None: events.append(ev) or True)
    cb = DeltaSnapshotCallback(store_dir="/data/store", events_url="http://t/x",
                               job_id="j", volume="vol-1")
    st = _State(tmp_path)
    row = ManifestRow(name="c", batch=500,
                      state_path="tinker://run-1/weights/step_500")
    cb.on_checkpoint(st, row)
    assert len(events) == 1
    ev = events[0]
    assert ev["step"] == 500 and ev["delta_key"] == ""
    assert ev["meta"]["state_path"].startswith("tinker://")

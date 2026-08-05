"""Tests for evsys_sdk.compute.checkpoint_map."""
from evsys_sdk.compute.checkpoint_map import CheckpointMap, StoreRef


def _ref(vol="vol-1"):
    return StoreRef(kind="local_dir", provider="verda", volume=vol, path="/data")


def test_record_and_latest(tmp_path):
    m = CheckpointMap(path=str(tmp_path / "map.jsonl"))
    m.record("job1", 100, _ref(), base_key="base", delta_key="d100")
    ck = m.record("job1", 200, _ref(), base_key="base", delta_key="d200",
                  sha256={"base": "a" * 64}, meta={"cursor": 42})
    latest = m.latest("job1")
    assert latest is not None and latest.id == ck.id
    assert latest.step == 200
    assert latest.keys == ["base", "d200"]
    assert latest.meta["cursor"] == 42
    assert latest.store.provider == "verda"


def test_latest_prefers_highest_step_then_newest(tmp_path):
    m = CheckpointMap(path=str(tmp_path / "map.jsonl"))
    m.record("j", 300, _ref(), base_key="b", delta_key="old300")
    rewritten = m.record("j", 300, _ref(), base_key="b", delta_key="new300")
    m.record("j", 200, _ref(), base_key="b", delta_key="late200")
    assert m.latest("j").delta_key == "new300"
    assert m.latest("j").id == rewritten.id


def test_jobs_are_isolated(tmp_path):
    m = CheckpointMap(path=str(tmp_path / "map.jsonl"))
    m.record("a", 1, _ref(), base_key="b", delta_key="da")
    m.record("b", 9, _ref("vol-2"), base_key="b", delta_key="db")
    assert m.latest("a").delta_key == "da"
    assert len(m.for_job("b")) == 1
    assert m.latest("missing") is None


def test_survives_torn_final_line(tmp_path):
    p = tmp_path / "map.jsonl"
    m = CheckpointMap(path=str(p))
    m.record("j", 1, _ref(), base_key="b", delta_key="d1")
    with p.open("a") as f:
        f.write('{"id": "torn')          # process died mid-write
    assert m.latest("j").step == 1       # replay skips the torn record


def test_forget_compacts(tmp_path):
    m = CheckpointMap(path=str(tmp_path / "map.jsonl"))
    m.record("keep", 1, _ref(), base_key="b", delta_key="d")
    m.record("done", 1, _ref(), base_key="b", delta_key="d")
    m.record("done", 2, _ref(), base_key="b", delta_key="d2")
    assert m.forget("done") == 2
    assert m.latest("done") is None
    assert m.latest("keep").step == 1

"""Tests for evsys_sdk.compute.portability — including the full lifecycle:
config run -> queue -> scheduler poll -> placement on a storage-capable
provider -> delta checkpoints on node storage -> preemption -> requeue ->
restart on a DIFFERENT provider with a streamed transfer and an exact resume.
"""
import numpy as np
import pytest

from evsys_sdk.checkpoint_delta import DeltaCheckpointer, load_delta
from evsys_sdk.compute import availability as av
from evsys_sdk.compute.checkpoint_map import CheckpointMap, StoreRef
from evsys_sdk.compute.checkpoint_store import (LocalDirStore, put_file,
                                                sha256_of)
from evsys_sdk.compute.portability import (CheckpointingScheduler,
                                           has_persistent_storage,
                                           on_preempted,
                                           register_storage_caps, restore)
from evsys_sdk.compute.queue import QUEUED, Queue


# -- storage capability registry ------------------------------------------

def test_known_caps_and_default_deny():
    assert has_persistent_storage("verda") is True
    assert has_persistent_storage("vast") is False
    assert has_persistent_storage("brand-new-cloud") is False


def test_register_extends():
    register_storage_caps("newcloud", True)
    try:
        assert has_persistent_storage("newcloud") is True
    finally:
        register_storage_caps("newcloud", False)


# -- storage-aware scheduling ---------------------------------------------

def _cap(provider, gpu="H100", usd=2.0, ok=True):
    return av.Capacity(provider=provider, gpu=gpu, count=1,
                       state="available" if ok else "none",
                       region="r1", spot=True, usd_hr=usd)


def test_scheduler_skips_storageless_provider(tmp_path, monkeypatch):
    """The cheaper vendor has no persistent storage; the placement must go to
    the storage-capable one even though it costs more."""
    q = Queue(path=str(tmp_path / "q.jsonl"))
    job = q.submit("configs/run.yaml", model="Qwen/Qwen3.5-9B", gpus=["H100"])

    def fake_scan(gpu, count, spot=None, clouds_=None):
        return [_cap("vast", usd=1.0), _cap("verda", usd=2.0)]

    monkeypatch.setattr(av, "scan", fake_scan)
    sched = CheckpointingScheduler(q)
    plans = sched.plan()
    assert len(plans) == 1
    assert plans[0].action == "rent"
    assert plans[0].capacity.provider == "verda"
    assert job.state == QUEUED


def test_scheduler_waits_when_only_storageless(tmp_path, monkeypatch):
    q = Queue(path=str(tmp_path / "q.jsonl"))
    q.submit("configs/run.yaml", model="m", gpus=["H100"])
    monkeypatch.setattr(av, "scan",
                        lambda *a, **k: [_cap("vast", usd=0.5)])
    plans = CheckpointingScheduler(q).plan()
    assert plans[0].action == "wait"


# -- preemption + restore --------------------------------------------------

def _base_state():
    r = np.random.default_rng(7)
    return {"w": r.standard_normal((128, 128), dtype=np.float32)}


def test_on_preempted_stamps_resume_point(tmp_path):
    q = Queue(path=str(tmp_path / "q.jsonl"))
    m = CheckpointMap(path=str(tmp_path / "map.jsonl"))
    job = q.submit("c.yaml", model="m")
    ck = m.record(job.id, 500, StoreRef(kind="local_dir", provider="verda",
                                        volume="vol-9"),
                  base_key="base", delta_key="d500")
    back = on_preempted(q, m, job)
    assert back.state == QUEUED
    assert back.attempts == 1
    assert f"resume:{ck.id}" in back.error


def test_on_preempted_without_checkpoint_is_explicit(tmp_path):
    q = Queue(path=str(tmp_path / "q.jsonl"))
    m = CheckpointMap(path=str(tmp_path / "map.jsonl"))
    job = q.submit("c.yaml", model="m")
    back = on_preempted(q, m, job)
    assert "no checkpoint" in back.error


def test_restore_requires_some_checkpoint(tmp_path):
    m = CheckpointMap(path=str(tmp_path / "map.jsonl"))
    dst = LocalDirStore(tmp_path / "dst")
    with pytest.raises(LookupError):
        restore(m, "ghost", dst)


def test_restore_same_store_verifies_presence(tmp_path):
    m = CheckpointMap(path=str(tmp_path / "map.jsonl"))
    store = LocalDirStore(tmp_path / "vol")
    m.record("j", 10, StoreRef(kind="local_dir", volume="vol"),
             base_key="base", delta_key="d10")
    with pytest.raises(FileNotFoundError):
        restore(m, "j", store)          # blobs missing, src not given


def test_full_lifecycle_cross_provider_resume(tmp_path):
    """The test the module exists for."""
    # -- a config run goes into the queue ---------------------------------
    q = Queue(path=str(tmp_path / "q.jsonl"))
    cmap = CheckpointMap(path=str(tmp_path / "map.jsonl"))
    job = q.submit("configs/sft_9b.yaml", model="Qwen/Qwen3.5-9B",
                   gpus=["H100"])

    # -- node A (provider verda, persistent volume) runs it ----------------
    vol_a = LocalDirStore(tmp_path / "verda-vol")
    ref_a = StoreRef(kind="local_dir", provider="verda", volume="verda-vol",
                     path=str(tmp_path / "verda-vol"))
    base = _base_state()
    work = tmp_path / "workA"
    ck = DeltaCheckpointer(base, str(work), keep_last=2)

    trained = {k: v.copy() for k, v in base.items()}
    for step in (100, 200):
        trained["w"][0, :10] += 0.01 * step          # "training"
        ck.save(step, trained)
        put_file(vol_a, "base.evd", work / "base.evd")
        put_file(vol_a, f"step-{step}.evd", work / f"step-{step}.evd")
        cmap.record(job.id, step, ref_a,
                    base_key="base.evd", delta_key=f"step-{step}.evd",
                    sha256={"base.evd": sha256_of(vol_a, "base.evd"),
                            f"step-{step}.evd": sha256_of(vol_a, f"step-{step}.evd")},
                    meta={"config": "configs/sft_9b.yaml"})

    # -- node A is preempted ----------------------------------------------
    job = on_preempted(q, cmap, job)
    assert job.state == QUEUED and "resume:" in job.error

    # -- placement lands on a DIFFERENT provider; volume A survived --------
    vol_b = LocalDirStore(tmp_path / "othercloud-vol")
    plan = restore(cmap, job.id, vol_b, src=vol_a)
    assert plan.checkpoint.step == 200
    assert plan.cross_store
    assert set(plan.copied) == {"base.evd", "step-200.evd"}

    # -- node B reconstructs the exact weights and resumes -----------------
    from evsys_sdk.compute.checkpoint_store import get_to_file
    get_to_file(vol_b, "base.evd", tmp_path / "b_base.evd")
    get_to_file(vol_b, "step-200.evd", tmp_path / "b_delta.evd")
    recon = load_delta(str(tmp_path / "b_base.evd"), str(tmp_path / "b_delta.evd"))
    assert np.array_equal(recon["w"], trained["w"])   # byte-exact resume
    resume_step = plan.checkpoint.step
    assert resume_step == 200
    assert plan.checkpoint.meta["config"] == "configs/sft_9b.yaml"

    # -- a second restore is nearly free (blobs already there) -------------
    plan2 = restore(cmap, job.id, vol_b, src=vol_a)
    assert plan2.copied == []
    assert set(plan2.already_there) == {"base.evd", "step-200.evd"}

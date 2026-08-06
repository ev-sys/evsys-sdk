"""Tests for evsys_sdk.checkpoint_delta — exact XOR-delta reconstruction + size win."""
import os

import numpy as np
import pytest

from evsys_sdk.checkpoint_delta import DeltaCheckpointer, load_delta, save_base, save_delta


def _rng(seed):
    return np.random.default_rng(seed)


def _base_state():
    r = _rng(0)
    return {
        "layer0.weight": r.standard_normal((256, 256), dtype=np.float32),
        "layer1.weight": r.standard_normal((512, 128), dtype=np.float32),
        "norm.weight": r.standard_normal((256,), dtype=np.float32),
    }


def test_roundtrip_is_byte_exact(tmp_path):
    base = _base_state()
    ckpt = {k: v.copy() for k, v in base.items()}
    # nudge a handful of weights, like a light fine-tune step
    ckpt["layer0.weight"][0, :5] += 0.01
    save_base(base, str(tmp_path / "base.evd"))
    save_delta(ckpt, str(tmp_path / "base.evd"), str(tmp_path / "d.evd"))
    recon = load_delta(str(tmp_path / "base.evd"), str(tmp_path / "d.evd"))
    for k in ckpt:
        assert np.array_equal(recon[k], ckpt[k])           # exact, not approximate
        assert recon[k].dtype == ckpt[k].dtype
        assert recon[k].shape == ckpt[k].shape


def test_unchanged_checkpoint_delta_is_tiny(tmp_path):
    base = _base_state()
    full = save_base(base, str(tmp_path / "base.evd"))
    # identical checkpoint -> XOR all zero -> compresses to near nothing
    delta = save_delta(base, str(tmp_path / "base.evd"), str(tmp_path / "d.evd"))
    assert delta < full * 0.05


def test_sparse_change_compresses_well(tmp_path):
    base = _base_state()
    ckpt = {k: v.copy() for k, v in base.items()}
    ckpt["layer1.weight"][:2, :] += 0.5          # ~0.4% of all weights touched
    full = save_base(base, str(tmp_path / "base.evd"))
    delta = save_delta(ckpt, str(tmp_path / "base.evd"), str(tmp_path / "d.evd"))
    # a few checkpoints should fit in far less than storing each in full
    assert delta < full * 0.25
    recon = load_delta(str(tmp_path / "base.evd"), str(tmp_path / "d.evd"))
    assert np.array_equal(recon["layer1.weight"], ckpt["layer1.weight"])


def test_shape_mismatch_raises(tmp_path):
    base = _base_state()
    save_base(base, str(tmp_path / "base.evd"))
    bad = {k: v.copy() for k, v in base.items()}
    bad["norm.weight"] = np.zeros((128,), dtype=np.float32)
    with pytest.raises(ValueError):
        save_delta(bad, str(tmp_path / "base.evd"), str(tmp_path / "d.evd"))


def test_missing_key_raises(tmp_path):
    base = _base_state()
    save_base(base, str(tmp_path / "base.evd"))
    with pytest.raises(KeyError):
        save_delta({"ghost.weight": np.zeros((4,), dtype=np.float32)},
                   str(tmp_path / "base.evd"), str(tmp_path / "d.evd"))


def test_checkpointer_keep_last_overwrites(tmp_path):
    base = _base_state()
    ck = DeltaCheckpointer(base, str(tmp_path / "ckpts"), keep_last=2)
    for step in (10, 20, 30):
        sd = {k: v.copy() for k, v in base.items()}
        sd["norm.weight"][0] += step
        ck.save(step, sd)
    files = set(os.listdir(str(tmp_path / "ckpts")))
    assert "step-10.evd" not in files            # evicted
    assert {"step-20.evd", "step-30.evd", "base.evd"} <= files
    recon = ck.load(30)
    assert np.array_equal(recon["norm.weight"], base["norm.weight"] + np.float32(30) * (np.arange(256) == 0))


def test_many_checkpoints_fit_in_bounded_disk(tmp_path):
    base = _base_state()
    ck = DeltaCheckpointer(base, str(tmp_path / "ckpts"), keep_last=3)
    for step in range(20):
        sd = {k: v.copy() for k, v in base.items()}
        sd["layer0.weight"][0, 0] += step + 1
        ck.save(step, sd)
    deltas = [f for f in os.listdir(str(tmp_path / "ckpts")) if f.startswith("step-")]
    assert len(deltas) == 3                       # 20 saves, bounded to 3 on disk

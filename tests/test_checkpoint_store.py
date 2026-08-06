"""Tests for evsys_sdk.compute.checkpoint_store."""
import io

import pytest

from evsys_sdk.compute.checkpoint_store import (
    LocalDirStore,
    get_to_file,
    put_file,
    sha256_of,
    stream_copy,
)


def test_put_get_roundtrip(tmp_path):
    s = LocalDirStore(tmp_path / "a")
    n = s.put("base.evd", io.BytesIO(b"hello world"))
    assert n == 11
    assert s.exists("base.evd")
    with s.get("base.evd") as f:
        assert f.read() == b"hello world"
    assert s.keys() == ["base.evd"]


def test_get_missing_raises_keyerror(tmp_path):
    s = LocalDirStore(tmp_path / "a")
    with pytest.raises(KeyError):
        s.get("nope")


def test_put_is_atomic_no_part_files_visible(tmp_path):
    s = LocalDirStore(tmp_path / "a")
    s.put("k", io.BytesIO(b"x" * 1000))
    assert all(not k.endswith(".part") for k in s.keys())


def test_key_traversal_rejected(tmp_path):
    s = LocalDirStore(tmp_path / "a")
    for bad in ("../evil", "a/b", ".hidden", ""):
        with pytest.raises(ValueError):
            s.put(bad, io.BytesIO(b"x"))


def test_delete_idempotent(tmp_path):
    s = LocalDirStore(tmp_path / "a")
    s.put("k", io.BytesIO(b"x"))
    s.delete("k")
    s.delete("k")               # second delete is not an error
    assert not s.exists("k")


def test_stream_copy_moves_and_skips(tmp_path):
    a = LocalDirStore(tmp_path / "a")
    b = LocalDirStore(tmp_path / "b")
    a.put("base", io.BytesIO(b"B" * 100))
    a.put("d1", io.BytesIO(b"1"))
    b.put("base", io.BytesIO(b"B" * 100))       # already transferred earlier
    copied = stream_copy(a, b, ["base", "d1"])
    assert copied == ["d1"]                      # base skipped
    with b.get("d1") as f:
        assert f.read() == b"1"


def test_stream_copy_verifies_digest(tmp_path):
    a = LocalDirStore(tmp_path / "a")
    b = LocalDirStore(tmp_path / "b")
    a.put("k", io.BytesIO(b"payload"))
    good = sha256_of(a, "k")
    stream_copy(a, b, ["k"], verify={"k": good})           # passes
    b.delete("k")
    with pytest.raises(IOError):
        stream_copy(a, b, ["k"], verify={"k": "0" * 64})   # wrong digest
    assert not b.exists("k")                               # corrupt copy removed


def test_file_helpers(tmp_path):
    s = LocalDirStore(tmp_path / "a")
    src = tmp_path / "w.bin"
    src.write_bytes(b"weights")
    assert put_file(s, "w", src) == 7
    out = tmp_path / "out.bin"
    assert get_to_file(s, "w", out) == 7
    assert out.read_bytes() == b"weights"

"""Delta checkpointing — store base weights once, checkpoints as compressed XOR deltas.

Fine-tuning (especially LoRA / light SFT) changes few weights, and the ones that
change usually move by a small amount — so the **XOR of a checkpoint's raw bytes
against the base** is mostly zero, and mostly-zero data compresses to almost
nothing. This lets a run keep many checkpoints in the disk footprint of roughly
one, and pairs with persistent-storage nodes that overwrite a bounded set.

Two storage modes:

  * ``DeltaCheckpointer(base, dir)`` — writes ``base.evd`` once, then each
    ``save(step, sd)`` appends ``step-<n>.evd`` holding only ``compress(sd XOR base)``.
  * ``keep_last=k`` — overwrite-in-place: only the most recent ``k`` deltas are
    kept on disk (for nodes with small persistent volumes).

Reconstruction is exact and lossless (byte-identical), because XOR is its own
inverse: ``base XOR (base XOR ckpt) == ckpt``. No floating-point tolerance
needed — we never do arithmetic on the values, only on their bytes.

The on-disk delta format is a small self-describing container so a delta can be
loaded without the original Python objects; only the base file is needed.
"""
from __future__ import annotations

import json
import os
import struct
import zlib
from typing import Any

import numpy as np

MAGIC = b"EVD1"


def _as_u8(arr: Any) -> tuple[np.ndarray, str, tuple[int, ...]]:
    """Return (contiguous uint8 view, dtype-str, shape) for a numpy/torch tensor."""
    if hasattr(arr, "detach"):  # torch.Tensor
        arr = arr.detach().cpu().numpy()
    arr = np.ascontiguousarray(arr)
    return arr.view(np.uint8).reshape(-1), str(arr.dtype), tuple(arr.shape)


def _pack(entries: list[tuple[str, bytes, str, tuple[int, ...]]]) -> bytes:
    """Container: MAGIC | json header | concatenated blobs."""
    header, blobs, off = [], [], 0
    for name, blob, dt, shape in entries:
        header.append({"name": name, "dtype": dt, "shape": list(shape),
                       "off": off, "len": len(blob)})
        blobs.append(blob)
        off += len(blob)
    hj = json.dumps(header).encode()
    return MAGIC + struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def _unpack(raw: bytes) -> tuple[list[dict], bytes]:
    if raw[:4] != MAGIC:
        raise ValueError("not an EVD container")
    hlen = struct.unpack("<Q", raw[4:12])[0]
    header = json.loads(raw[12:12 + hlen].decode())
    return header, raw[12 + hlen:]


def save_base(state_dict: dict[str, Any], path: str) -> int:
    """Write the reference weights. Returns bytes written."""
    entries = []
    for name, t in state_dict.items():
        u8, dt, shape = _as_u8(t)
        entries.append((name, u8.tobytes(), dt, shape))
    data = _pack(entries)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def _load_base_raw(path: str) -> dict[str, tuple[np.ndarray, str, tuple[int, ...]]]:
    with open(path, "rb") as f:
        raw = f.read()
    header, body = _unpack(raw)
    out = {}
    for h in header:
        blob = body[h["off"]:h["off"] + h["len"]]
        out[h["name"]] = (np.frombuffer(blob, dtype=np.uint8), h["dtype"], tuple(h["shape"]))
    return out


def save_delta(state_dict: dict[str, Any], base_path: str, delta_path: str,
               level: int = 6) -> int:
    """Write ``compress(state_dict XOR base)``. Returns bytes written.

    Keys/shapes/dtypes must match the base. XOR is done on raw bytes, so the
    delta of an unchanged tensor is all-zero and compresses to a few bytes.
    """
    base = _load_base_raw(base_path)
    entries = []
    for name, t in state_dict.items():
        u8, dt, shape = _as_u8(t)
        if name not in base:
            raise KeyError(f"{name} not in base checkpoint")
        b_u8, b_dt, b_shape = base[name]
        if (dt, shape) != (b_dt, b_shape) or u8.shape != b_u8.shape:
            raise ValueError(f"{name}: shape/dtype differs from base ({dt}{shape} vs {b_dt}{b_shape})")
        xor = np.bitwise_xor(u8, b_u8)
        entries.append((name, zlib.compress(xor.tobytes(), level), dt, tuple(shape)))
    data = _pack(entries)
    tmp = delta_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, delta_path)  # atomic: safe for overwrite-in-place
    return len(data)


def load_delta(base_path: str, delta_path: str) -> dict[str, np.ndarray]:
    """Reconstruct the checkpoint exactly from base + delta."""
    base = _load_base_raw(base_path)
    with open(delta_path, "rb") as f:
        header, body = _unpack(f.read())
    out = {}
    for h in header:
        comp = body[h["off"]:h["off"] + h["len"]]
        xor = np.frombuffer(zlib.decompress(comp), dtype=np.uint8)
        b_u8, dt, shape = base[h["name"]]
        recon = np.bitwise_xor(xor, b_u8)
        out[h["name"]] = recon.view(np.dtype(dt)).reshape(shape)
    return out


class DeltaCheckpointer:
    """Keep many checkpoints in ~one checkpoint's disk footprint.

    ``keep_last=k`` bounds disk to the base + k newest deltas — the pattern for a
    persistent-storage node that overwrites rather than accumulating.
    """

    def __init__(self, base_state: dict[str, Any], directory: str,
                 keep_last: int | None = None, level: int = 6):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory
        self.base_path = os.path.join(directory, "base.evd")
        self.keep_last = keep_last
        self.level = level
        self.base_bytes = save_base(base_state, self.base_path)
        self.steps: list[int] = []

    def _delta_path(self, step: int) -> str:
        return os.path.join(self.dir, f"step-{step}.evd")

    def save(self, step: int, state_dict: dict[str, Any]) -> int:
        n = save_delta(state_dict, self.base_path, self._delta_path(step), self.level)
        self.steps.append(step)
        if self.keep_last is not None:
            while len(self.steps) > self.keep_last:
                old = self.steps.pop(0)
                try:
                    os.remove(self._delta_path(old))
                except FileNotFoundError:
                    pass
        return n

    def load(self, step: int) -> dict[str, np.ndarray]:
        return load_delta(self.base_path, self._delta_path(step))


__all__ = ["DeltaCheckpointer", "load_delta", "save_base", "save_delta"]

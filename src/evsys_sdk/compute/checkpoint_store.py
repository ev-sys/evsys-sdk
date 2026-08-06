"""Where checkpoints live, independent of which machine wrote them.

A spot node's persistent volume outlives the node — on Verda a volume keeps
existing (and billing) after its instance is preempted. That surviving disk is
the checkpoint store this module abstracts: a flat keyspace of named blobs on
*somebody's* storage, read and written in streams so a multi-GB base file never
has to fit in memory.

Two deliberate restrictions keep every implementation honest:

  * **Flat keys, bytes only.** No hierarchy, no metadata channel. What a
    checkpoint *means* (job, step, encoding) lives in
    :mod:`.checkpoint_map`, not in the store — so moving blobs between
    stores can never lose meaning.
  * **Streams, not paths.** ``get``/``put`` speak file-objects. That is what
    makes :func:`stream_copy` provider-agnostic: it pipes chunks from one
    store's reader into another's writer without a local staging copy, which
    is exactly the "transfer directly from one storage to another" a
    cross-provider restart needs.

``LocalDirStore`` is the only concrete store here: it covers both the node's
own persistent mount (``/data`` on the machine doing the training) and any
volume re-attached to a recovery node. Cloud-object stores register later —
the protocol is the extension point, mirroring how providers extend
:mod:`.availability`.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import tempfile
from collections.abc import Iterable
from typing import BinaryIO, Protocol, runtime_checkable

from ..logger import get_logger

log = get_logger(__name__)

#: 8 MiB: large enough that stream_copy is bandwidth-bound, small enough that
#: a chunk never matters to memory even with several transfers in flight.
CHUNK_BYTES = 8 * 1024 * 1024


@runtime_checkable
class CheckpointStore(Protocol):
    """A flat keyspace of blobs that survives the machine that wrote it."""

    def put(self, key: str, src: BinaryIO) -> int:
        """Store ``src``'s bytes under ``key``. Returns bytes written.

        Must be atomic per key: a reader never sees a half-written blob.
        """
        ...

    def get(self, key: str) -> BinaryIO:
        """Open ``key`` for reading. Raises ``KeyError`` if absent."""
        ...

    def exists(self, key: str) -> bool: ...

    def keys(self) -> list[str]: ...

    def delete(self, key: str) -> None:
        """Remove ``key``. Missing keys are not an error — deletes retry."""
        ...


class LocalDirStore:
    """A directory as a checkpoint store — a node's persistent mount.

    Writes go through a temp file and ``os.replace`` so a preemption mid-write
    leaves the previous blob intact rather than a torn one. That is the same
    atomicity choice as :meth:`~.checkpoint_delta.save_delta` and the queue's
    ``compact``, for the same reason: this file may be all that survives.
    """

    def __init__(self, root: str | os.PathLike):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> pathlib.Path:
        if not key or "/" in key or key.startswith("."):
            # Flat keyspace, enforced. Path traversal through a checkpoint key
            # would let a corrupt map overwrite files outside the store.
            raise ValueError(f"invalid store key: {key!r}")
        return self.root / key

    def put(self, key: str, src: BinaryIO) -> int:
        dst = self._path(key)
        fd, tmp = tempfile.mkstemp(dir=str(self.root), suffix=".part")
        n = 0
        try:
            with os.fdopen(fd, "wb") as f:
                while True:
                    chunk = src.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    f.write(chunk)
                    n += len(chunk)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, dst)
        finally:
            pathlib.Path(tmp).unlink(missing_ok=True)
        return n

    def get(self, key: str) -> BinaryIO:
        p = self._path(key)
        if not p.exists():
            raise KeyError(key)
        return p.open("rb")

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def keys(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_file() and not p.name.endswith(".part"))

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


def sha256_of(store: CheckpointStore, key: str) -> str:
    """Digest of a stored blob, streamed. The map records this at write time;
    verifying it after a transfer is what turns "the copy finished" into "the
    copy is the same bytes" — which matters, because a truncated base file
    reconstructs every delta wrongly and silently."""
    h = hashlib.sha256()
    with store.get(key) as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stream_copy(src: CheckpointStore, dst: CheckpointStore,
                keys: Iterable[str], *, skip_existing: bool = True,
                verify: dict[str, str] | None = None) -> list[str]:
    """Pipe blobs from one store to another, chunk by chunk.

    ``skip_existing`` makes re-runs cheap: a restart that already copied the
    base (by far the largest blob) only moves the new deltas. ``verify`` maps
    key -> expected sha256; a mismatch raises rather than resuming a job from
    corrupt weights.

    Returns the keys actually copied.
    """
    copied: list[str] = []
    for key in keys:
        if skip_existing and dst.exists(key):
            log.debug("[store] %s already at destination, skipped", key)
            continue
        with src.get(key) as f:
            n = dst.put(key, f)
        if verify and key in verify:
            got = sha256_of(dst, key)
            if got != verify[key]:
                dst.delete(key)
                raise OSError(f"transfer of {key!r} corrupt: "
                              f"sha256 {got[:12]} != {verify[key][:12]}")
        log.info("[store] copied %s (%d bytes)", key, n)
        copied.append(key)
    return copied


def put_file(store: CheckpointStore, key: str, path: str | os.PathLike) -> int:
    """Convenience: store a file that already exists on local disk."""
    with open(path, "rb") as f:
        return store.put(key, f)


def get_to_file(store: CheckpointStore, key: str, path: str | os.PathLike) -> int:
    """Convenience: materialise a blob to local disk (e.g. for load_delta)."""
    with store.get(key) as f, open(path, "wb") as out:
        shutil.copyfileobj(f, out, CHUNK_BYTES)
    return os.path.getsize(path)


__all__ = ["CHUNK_BYTES", "CheckpointStore", "LocalDirStore", "get_to_file",
           "put_file", "sha256_of", "stream_copy"]

"""Which checkpoint belongs to which job, and where its bytes physically are.

The queue knows a job died; the store holds anonymous blobs. Neither can
answer the question a restart asks: *"job 7f3a was at step 4200 — which blobs
reconstruct it, and on whose storage do they live right now?"* This map is
that answer, and it is the piece that makes a restart portable: the record
names a **provider and volume**, not a machine, so the same entry works
whether the job resumes on the cluster that wrote it or on a different
provider entirely (after :func:`~.checkpoint_store.stream_copy` moves the
blobs).

Durability follows :mod:`.queue` exactly — append-only JSONL, latest record
wins, torn final lines skipped. The map must survive everything the queue
survives, because a queue that remembers a job without its checkpoints can
only restart it from zero.

A checkpoint entry names blobs by *store key* and records each blob's sha256
at write time. The base is recorded once per (job, store) and shared by every
delta — that is the whole economy of XOR-delta checkpointing
(:mod:`..checkpoint_delta`): one big blob, then many tiny ones.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from ..logger import get_logger

log = get_logger(__name__)

MAP_PATH = "~/.evsys/checkpoint_map.jsonl"


@dataclass
class StoreRef:
    """Where blobs physically live: enough to find the storage again after
    the machine is gone. ``kind`` names the store implementation;
    ``provider``/``volume`` locate it in the world; ``path`` is the mount
    point or prefix within it."""

    kind: str                     # e.g. "local_dir" (a node's persistent mount)
    provider: str = ""            # e.g. "verda" — whose cloud holds the volume
    volume: str = ""              # provider's volume id/name, survives the node
    path: str = ""                # mount point / prefix

    def describe(self) -> str:
        loc = "/".join(x for x in (self.provider, self.volume) if x)
        return f"{self.kind}:{loc or self.path}"


@dataclass
class Checkpoint:
    """One restartable state of one job."""

    job_id: str
    step: int
    store: StoreRef
    base_key: str                 # blob holding base weights (shared)
    delta_key: str                # blob holding this step's XOR delta
    sha256: dict[str, str] = field(default_factory=dict)   # key -> digest
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)
    """Free-form resume context (data cursor, tenant name, config hash) —
    the things :class:`~.snapshot.ResumeManifest` tracks, carried alongside
    so a restart on a fresh provider needs exactly one lookup."""

    @property
    def keys(self) -> list[str]:
        return [self.base_key, self.delta_key]

    def describe(self) -> str:
        return (f"{self.job_id}@step{self.step} "
                f"[{self.store.describe()}] base={self.base_key} "
                f"delta={self.delta_key}")


class CheckpointMap:
    """Durable job -> checkpoints mapping. Append-only, latest-wins per id."""

    def __init__(self, path: str = MAP_PATH):
        self.path = pathlib.Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, ck: Checkpoint) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(ck)) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def record(self, job_id: str, step: int, store: StoreRef, *,
               base_key: str, delta_key: str,
               sha256: dict[str, str] | None = None,
               meta: dict[str, Any] | None = None) -> Checkpoint:
        ck = Checkpoint(job_id=job_id, step=step, store=store,
                        base_key=base_key, delta_key=delta_key,
                        sha256=dict(sha256 or {}), meta=dict(meta or {}))
        self._append(ck)
        log.info("[ckmap] %s", ck.describe())
        return ck

    def _all(self) -> list[Checkpoint]:
        if not self.path.exists():
            return []
        latest: dict[str, dict] = {}
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue        # torn final line: same policy as the queue
            if "id" in rec:
                latest[rec["id"]] = rec
        out = []
        for rec in latest.values():
            rec = dict(rec)
            rec["store"] = StoreRef(**rec.get("store") or {})
            known = {k: v for k, v in rec.items()
                     if k in Checkpoint.__annotations__}
            out.append(Checkpoint(**known))
        return sorted(out, key=lambda c: c.created_at)

    def for_job(self, job_id: str) -> list[Checkpoint]:
        return [c for c in self._all() if c.job_id == job_id]

    def latest(self, job_id: str) -> Checkpoint | None:
        """The checkpoint a restart should resume from — highest step, and
        newest within a step, so a re-written checkpoint supersedes cleanly."""
        cks = self.for_job(job_id)
        if not cks:
            return None
        return max(cks, key=lambda c: (c.step, c.created_at))

    def forget(self, job_id: str) -> int:
        """Drop a finished job's records (compacting rewrite). The blobs are
        the caller's to delete — the map never touches storage."""
        keep = [c for c in self._all() if c.job_id != job_id]
        dropped = len(self._all()) - len(keep)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            for c in keep:
                f.write(json.dumps(asdict(c)) + "\n")
        pathlib.Path(tmp).replace(self.path)
        return dropped


__all__ = ["MAP_PATH", "Checkpoint", "CheckpointMap", "StoreRef"]

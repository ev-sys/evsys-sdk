"""Jobs that survive their machines — the glue between queue, map and store.

The queue places work; :mod:`.checkpoint_delta` encodes state cheaply;
:mod:`.checkpoint_store` holds the bytes; :mod:`.checkpoint_map` remembers
which bytes are whose. This module is the policy that connects them:

  * **Only rent nodes that can checkpoint.** A spot box without persistent
    storage is a box whose work evaporates with it. ``storage_caps`` records,
    per provider, whether volumes survive the instance; the scheduler filter
    drops capacity that cannot make that promise. The table is a registry —
    new providers add one line, from their own module, exactly like
    availability probes register themselves.
  * **Preemption is a checkpoint event, not a failure.** ``on_preempted``
    requeues the job (the queue already counts attempts) and stamps it with
    its latest checkpoint id, so the next placement knows it is a resume.
  * **Restart anywhere.** ``restore`` takes the job's latest checkpoint and
    the store on the *new* node: same store — nothing to move; different
    store (other volume, other provider) — the blobs stream across, digests
    verified, before the job is told to resume. The base blob is skipped when
    already present, which is what makes repeated restarts cheap.

Everything here is pure policy over injected callables — no cloud SDK, no
HTTP — so the whole preempt/requeue/restore cycle is testable in memory, the
same design rule :mod:`.snapshot` follows.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..logger import get_logger
from . import availability as av
from .checkpoint_map import Checkpoint, CheckpointMap
from .checkpoint_store import CheckpointStore, sha256_of, stream_copy
from .queue import Job, Placement, Queue, Scheduler

log = get_logger(__name__)

#: Providers whose volumes outlive their instances. Verda is measured fact —
#: its disks survive preemption (and keep billing, which is the reaper's
#: problem, not this module's). Vast rents machines whose disk dies with the
#: container, so it cannot host a checkpointed job until an object-store
#: backend exists for it.
_STORAGE_CAPS: dict[str, bool] = {
    "verda": True,
    "vast": False,
}


def register_storage_caps(provider: str, persistent: bool) -> None:
    """New providers declare themselves here, from their own module."""
    _STORAGE_CAPS[provider] = persistent


def has_persistent_storage(provider: str) -> bool:
    """Unknown providers default to False: renting a node that silently
    cannot checkpoint is the expensive kind of surprise."""
    return _STORAGE_CAPS.get(provider, False)


class CheckpointingScheduler(Scheduler):
    """A scheduler that refuses capacity a checkpoint cannot survive on.

    Everything else — packing, pricing, retry accounting — is inherited.
    Filtering inside ``_cheapest`` (rather than re-sorting placements after)
    keeps the "cheapest acceptable" guarantee: the next-cheapest *storage-
    capable* vendor wins, instead of the job waiting behind a cheaper vendor
    it could never use.
    """

    def _cheapest(self, job: Job) -> av.Capacity | None:
        best: av.Capacity | None = None
        for gpu in job.gpus:
            for c in av.scan(gpu, job.count, spot=job.spot,
                             clouds_=self.vendors):
                if not c.ok:
                    continue
                if not has_persistent_storage(c.provider):
                    log.debug("[portable] %s has no persistent storage, "
                              "skipped for %s", c.provider, job.id)
                    continue
                if best is None or (c.usd_per_gpu_hr or 0) < (best.usd_per_gpu_hr or 0):
                    best = c
                break
        return best


def on_preempted(queue: Queue, cmap: CheckpointMap, job: Job) -> Job:
    """Requeue a job whose machine vanished, stamped with its resume point.

    The stamp goes in ``job.error`` — the queue's existing free-text field —
    as ``resume:<checkpoint-id>``, so no queue schema change is needed and a
    human reading the log sees exactly what the next placement will do.
    """
    ck = cmap.latest(job.id)
    reason = f"preempted; resume:{ck.id}" if ck else "preempted; no checkpoint"
    if ck:
        log.info("[portable] %s will resume from step %d (%s)",
                 job.id, ck.step, ck.store.describe())
    else:
        log.warning("[portable] %s preempted with no checkpoint — restarts "
                    "from zero", job.id)
    return queue.requeue(job, reason)


@dataclass
class RestorePlan:
    """What a restore did, returned for logging and for tests."""

    checkpoint: Checkpoint
    copied: list[str]           # keys streamed to the new store
    already_there: list[str]    # keys the destination had (base, usually)

    @property
    def cross_store(self) -> bool:
        return bool(self.copied)


def restore(cmap: CheckpointMap, job_id: str, dst: CheckpointStore,
            src: CheckpointStore | None = None, *,
            verify: bool = True) -> RestorePlan:
    """Bring a job's latest checkpoint to the node that will resume it.

    ``src`` is the store holding the blobs today — opened by the caller from
    the checkpoint's :class:`~.checkpoint_map.StoreRef` (how a ref becomes a
    live store is provider-specific: re-attach the volume, mount the clone).
    ``src=None`` asserts the destination *is* that store — same volume
    re-attached — and only verifies presence.

    Digests are checked when the map recorded them: a resume from a
    truncated base produces wrong weights on every delta, silently, which is
    strictly worse than failing here.
    """
    ck = cmap.latest(job_id)
    if ck is None:
        raise LookupError(f"no checkpoint recorded for job {job_id}")

    needed = ck.keys
    if src is None:
        missing = [k for k in needed if not dst.exists(k)]
        if missing:
            raise FileNotFoundError(
                f"resume store lacks {missing} for {ck.describe()} — "
                f"pass src= to stream them from {ck.store.describe()}")
        copied: list[str] = []
        already = list(needed)
    else:
        already = [k for k in needed if dst.exists(k)]
        expect = {k: ck.sha256[k] for k in needed if k in ck.sha256} if verify else None
        copied = stream_copy(src, dst, needed, skip_existing=True,
                             verify=expect)
    if verify:
        for k in needed:
            want = ck.sha256.get(k)
            if want and sha256_of(dst, k) != want:
                raise IOError(f"{k} on resume store does not match the "
                              f"digest recorded at step {ck.step}")
    log.info("[portable] %s restored at step %d (%d copied, %d present)",
             job_id, ck.step, len(copied), len(already))
    return RestorePlan(checkpoint=ck, copied=copied, already_there=already)


def describe_placements(placements: list[Placement]) -> str:
    """One line per decision — what a human tails while the router runs."""
    return "\n".join(p.describe() for p in placements)


__all__ = ["CheckpointingScheduler", "RestorePlan", "describe_placements",
           "has_persistent_storage", "on_preempted", "register_storage_caps",
           "restore"]

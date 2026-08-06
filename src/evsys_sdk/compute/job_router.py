"""The loop that runs jobs so nobody has to — queue in, surviving runs out.

Everything the E2E test did by hand (place a job, provision a node with a
persistent volume, track its checkpoints, notice the machine die, requeue,
re-place, re-attach or stream the checkpoints, resume) is one `tick()` here,
executed forever by `run()`. An operator's whole interface is ``submit``.

One tick, in order — each step automatic, none skippable:

  1. **Drain agent events.** Nodes report ``checkpoint`` events as they
     snapshot; each one lands in the :class:`~.checkpoint_map.CheckpointMap`.
     The map is maintained *by the router from agent reports*, never by hand,
     so it is always exactly as current as the newest event.
  2. **Reconcile.** Every RUNNING job's machine is polled. A machine that is
     gone triggers :func:`~.portability.on_preempted` — requeue, attempt
     counted, resume point stamped. Preemption is consumed here as an event,
     not discovered later as a mystery.
  3. **Place.** The storage-aware scheduler plans; every ``rent`` decision is
     provisioned with a :class:`VolumePlan` derived from the job's latest
     checkpoint:

       * same provider & the volume still exists  -> ``reuse``: attach the
         surviving volume (zero-copy restart — the E2E leg 2 path),
       * checkpoint exists elsewhere              -> ``stream``: the new
         node pulls the blobs from the old store (the E2E leg 1 path),
       * no checkpoint                            -> ``fresh``.

     The snapshot cadence handed to the node is Young/Daly optimal from
     :class:`~.snapshot.SnapshotPolicy` — cost and MTBF in, interval out.

Handles are durable (JSON file, atomic replace): a router process restart
re-reads them and the next tick reconciles as if nothing happened, the same
crash-model the queue and the map already share. All effects go through an
injected :class:`Provisioner`, so the entire lifecycle — including preemption
and cross-provider resume — is testable in memory.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from ..logger import get_logger
from . import availability as av
from .checkpoint_map import Checkpoint, CheckpointMap, StoreRef
from .portability import CheckpointingScheduler, on_preempted
from .queue import DONE, FAILED, RUNNING, Job, Queue
from .snapshot import SnapshotPolicy

log = get_logger(__name__)

HANDLES_PATH = "~/.evsys/router_handles.json"

#: Default MTBF for spot capacity when no measured figure exists yet: the
#: historical campaign saw roughly one preemption per successful multi-hour
#: launch. The lifetime ledger replaces this with data as it accumulates.
DEFAULT_MTBF_S = 3600.0


@dataclass
class VolumePlan:
    """How the new node gets the job's state. Decided by the router, executed
    by the provisioner."""

    mode: str                          # "fresh" | "reuse" | "stream"
    volume: str = ""                   # reuse: surviving volume to attach
    src: StoreRef | None = None        # stream: where the blobs live today
    checkpoint: Checkpoint | None = None   # resume point (None = from zero)

    def describe(self) -> str:
        if self.mode == "reuse":
            return f"reuse volume {self.volume} (zero-copy restart)"
        if self.mode == "stream":
            step = self.checkpoint.step if self.checkpoint else "?"
            return f"stream step-{step} from {self.src.describe() if self.src else '?'}"
        return "fresh volume"


@dataclass
class NodeHandle:
    """One provisioned machine running one job."""

    job_id: str
    provider: str
    machine_id: str
    region: str = ""
    gpu: str = ""
    volume: str = ""
    usd_hr: float = 0.0
    started_at: float = field(default_factory=time.time)


class Provisioner(Protocol):
    """The effectful edge. Everything above it is pure and tested."""

    def provision(self, job: Job, capacity: av.Capacity, plan: VolumePlan,
                  snapshot_interval_s: float) -> NodeHandle | None:
        """Bring up a node for the job. None means the launch failed and the
        job should stay queued for the next tick."""
        ...

    def alive(self, handle: NodeHandle) -> bool: ...

    def terminate(self, handle: NodeHandle, *, keep_volume: bool = True) -> None:
        """Tear a node down. ``keep_volume`` defaults True because the volume
        holding checkpoints is the part that must outlive the machine."""
        ...


def build_provisioner(kind: str, params: dict | None = None,
                      call: Callable[..., Any] | None = None) -> Provisioner:
    """Resolve a ``{kind, params}`` spec into a Provisioner — the same
    registry + Config convention as every other extension point.

    ``kind`` is looked up in the provisioner registry (built-ins register on
    import; a provider package registers its own with
    ``@register_provisioner("<name>")``). ``params`` are validated against
    the class's ``Config`` so a typo fails loudly. ``call`` optionally
    injects a transport (tests); ``None`` lets the class build its real
    authenticated one from credentials.
    """
    from ..registry import get_provisioner
    cls = get_provisioner(kind)
    cfg = getattr(cls, "Config", None)
    kw = cfg(**(params or {})).model_dump() if cfg is not None \
        else dict(params or {})
    return cls(call, **kw) if call is not None else cls(**kw)


class JobRouter:
    """Autonomous placement + snapshot-tracking + restart. See module doc."""

    def __init__(self, queue: Queue, cmap: CheckpointMap,
                 provisioner: Provisioner | dict[str, Provisioner], *,
                 events: Callable[[], list[dict]] | None = None,
                 vendors: list[str] | None = None,
                 clusters: Callable[[], list] | None = None,
                 snapshot_cost_s: float = 45.0,
                 mtbf_s: float = DEFAULT_MTBF_S,
                 handles_path: str = HANDLES_PATH,
                 poll_s: float = 60.0):
        self.queue = queue
        self.cmap = cmap
        # One provisioner (single-cloud) or a {provider: provisioner} map —
        # the router picks by Capacity.provider on the way up and by
        # NodeHandle.provider on the way down, so adding a cloud is one more
        # dict entry (register the class, list the vendor), no router change.
        if isinstance(provisioner, dict):
            self._provs: dict[str, Provisioner] = dict(provisioner)
            self.prov = next(iter(provisioner.values()), None)
            if vendors is None:
                vendors = list(provisioner)
        else:
            self._provs = {}
            self.prov = provisioner
        self._events = events or (lambda: [])
        self.sched = CheckpointingScheduler(queue, clusters=clusters,
                                            vendors=vendors)
        self.policy = SnapshotPolicy(snapshot_cost_s=snapshot_cost_s,
                                     mtbf_s=mtbf_s)
        self.poll_s = poll_s
        self._path = pathlib.Path(handles_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.handles: dict[str, NodeHandle] = self._load()

    # -- durable handles ---------------------------------------------------

    def _load(self) -> dict[str, NodeHandle]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text())
        except json.JSONDecodeError:
            return {}
        return {k: NodeHandle(**v) for k, v in raw.items()}

    def _save(self) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump({k: asdict(v) for k, v in self.handles.items()}, f)
        pathlib.Path(tmp).replace(self._path)

    # -- the automation ----------------------------------------------------

    def _prov_for(self, provider: str) -> Provisioner:
        """The provisioner owning ``provider``. Single-provisioner routers
        return their one provisioner for any name (back-compat)."""
        if self._provs:
            p = self._provs.get(provider)
            if p is None:
                raise KeyError(
                    f"no provisioner registered with the router for provider "
                    f"{provider!r} (have: {sorted(self._provs)})")
            return p
        return self.prov

    def submit(self, config: str, model: str, **kw: Any) -> Job:
        """Queue a job. **Spot by default**: snapshots + auto-restart make
        preemption a cost, not a risk, so the on-demand premium is opt-in —
        pass ``spot=False`` to force on-demand or ``spot=None`` to take
        whichever is cheapest."""
        kw.setdefault("spot", True)
        return self.queue.submit(config, model, **kw)

    def _drain_events(self) -> int:
        """Agent reports -> map. The only writer of checkpoint records."""
        n = 0
        for ev in self._events():
            kind = ev.get("kind")
            if kind == "checkpoint":
                self.cmap.record(
                    ev["job_id"], int(ev["step"]),
                    StoreRef(**ev["store"]),
                    base_key=ev["base_key"], delta_key=ev["delta_key"],
                    sha256=ev.get("sha256"), meta=ev.get("meta"))
                n += 1
            elif kind == "done":
                job = self._job(ev["job_id"])
                if job is not None:
                    self.queue.update(job, DONE)
                    h = self.handles.pop(job.id, None)
                    if h is not None:
                        self._prov_for(h.provider).terminate(
                            h, keep_volume=True)
                    self._save()
                n += 1
        return n

    def _job(self, job_id: str) -> Job | None:
        return next((j for j in self.queue.jobs() if j.id == job_id), None)

    def _reconcile(self) -> list[str]:
        """Dead machines become requeued jobs, automatically."""
        preempted = []
        for job_id, handle in list(self.handles.items()):
            if self._prov_for(handle.provider).alive(handle):
                continue
            job = self._job(job_id)
            if job is not None and job.state == RUNNING:
                on_preempted(self.queue, self.cmap, job)
                preempted.append(job_id)
            del self.handles[job_id]
            self._save()
        return preempted

    def _volume_plan(self, job: Job, capacity: av.Capacity) -> VolumePlan:
        """Where the new node's state comes from — the restart policy."""
        ck = self.cmap.latest(job.id)
        if ck is None:
            return VolumePlan("fresh")
        same_provider = (ck.store.provider == capacity.provider
                         and bool(ck.store.volume))
        if same_provider:
            return VolumePlan("reuse", volume=ck.store.volume, checkpoint=ck)
        return VolumePlan("stream", src=ck.store, checkpoint=ck)

    def tick(self) -> dict[str, Any]:
        """One full pass. Returns what happened, for logs and for tests."""
        drained = self._drain_events()
        preempted = self._reconcile()
        placed, waiting = [], []
        for placement in self.sched.plan():
            if placement.action != "rent" or placement.capacity is None:
                if placement.action == "wait":
                    waiting.append(placement.job.id)
                continue
            job, cap = placement.job, placement.capacity
            plan = self._volume_plan(job, cap)
            handle = self._prov_for(cap.provider).provision(
                job, cap, plan, self.policy.interval_s)
            if handle is None:
                log.warning("[router] launch failed for %s; stays queued",
                            job.id)
                continue
            self.handles[job.id] = handle
            self._save()
            self.queue.update(job, RUNNING,
                              cluster=f"{handle.provider}/{handle.machine_id}")
            log.info("[router] %s -> %s %s (%s)", job.id, handle.provider,
                     handle.machine_id, plan.describe())
            placed.append((job.id, plan.mode))
        return {"events": drained, "preempted": preempted,
                "placed": placed, "waiting": waiting}

    def run(self, *, timeout_s: float | None = None) -> None:
        """The daemon. Ticks until every job is terminal or time runs out."""
        start = time.time()
        while True:
            self.tick()
            jobs = self.queue.jobs()
            if jobs and all(j.state in (DONE, FAILED) for j in jobs):
                log.info("[router] all jobs terminal")
                return
            if timeout_s is not None and time.time() - start > timeout_s:
                log.info("[router] timeout reached")
                return
            time.sleep(self.poll_s)


__all__ = ["DEFAULT_MTBF_S", "JobRouter", "NodeHandle", "Provisioner",
           "VolumePlan", "build_provisioner"]

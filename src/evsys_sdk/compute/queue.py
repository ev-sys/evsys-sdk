"""A queue for SkyRL runs, so a job waits for capacity instead of failing on it.

Launching against Tinker always works: it is a hosted endpoint, so submitting a
config is the whole story. Self-hosted SkyRL has a step Tinker hides — somebody
has to find a GPU. Today that meant a human watching for spot capacity, and runs
died because nobody was watching at 3am.

This is that missing step. Submit a run and it is queued; a scheduler polls
:mod:`.availability` across the vendors you have enabled and places the job when
hardware appears. Nothing is lost to a launch that could not have succeeded.

The placement rule that matters is **packing, not spreading** — but for a
smaller reason than it first appears, and the size of the reason decides how
hard to pack.

Adapters on one GPU share it. Aggregate throughput barely moves as adapters are
added, so each adapter runs at roughly 1/n speed. Measured: an A100 at seq 1024
went 2,373 -> 2,411 tok/s from n=1 to n=8 (+1.6%, i.e. one adapter already
saturates that card), while an H100 at seq 4096 went 6,711 -> 7,643 at n=4
(+14%, so one adapter had left 14% of it idle) before declining again by n=16.

So packing is **not** free capacity, and it is not an n-times cost win. Against
the honest alternative — n separate machines — one H100 running 8 adapters
costs $0.0429/M against $0.0471/M for eight of them: about **9% cheaper per
token, with each experiment taking ~7x longer in wall clock**. Packing wins on
cost and on operational overhead (one setup, one preemption exposure instead of
eight), and loses on latency. That is the trade, and it is why packing is the
default rather than the only option: a job that needs its answer soon should
rent, not queue behind seven others.

Placement, in order:

  1. **Pack** onto a live cluster serving the same base model with a free
     adapter slot. Free.
  2. **Rent** the cheapest available accelerator the job will accept, across
     every enabled vendor, using live availability rather than a catalog.
  3. **Wait.** Not a failure — the job stays queued and is retried. Spot
     capacity that is gone now is usually back within the hour.

Durable by design: the queue is an append-only JSONL log, so a scheduler crash
loses nothing and the state is greppable while a run is in flight — the same
choice, for the same reason, as :mod:`.reliability`.

    from evsys_sdk.compute.queue import Queue, Scheduler

    q = Queue()
    q.submit("configs/mt_replay.yaml", model="Qwen/Qwen3-4B", gpus=["H200", "H100"])
    Scheduler(q).run()          # blocks, placing work as capacity appears
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

from ..logger import get_logger
from . import availability as av

log = get_logger(__name__)

QUEUE_PATH = "~/.evsys/queue.jsonl"

#: A job's life. PLACING is brief but real — a launch takes minutes, and
#: without its own state a crash mid-launch leaves a job that looks runnable
#: and is actually half-provisioned.
QUEUED, PLACING, RUNNING, DONE, FAILED = (
    "queued", "placing", "running", "done", "failed")

#: Accelerators we will take by default, best measured $/token first.
#: RTX PRO 6000 leads because it measured cheapest per token of anything
#: tested ($0.0320/M at 4,096 tokens), not because it is the biggest card.
DEFAULT_GPUS = ["RTXPRO6000", "H200", "H100", "A100"]


@dataclass
class Job:
    """One run waiting for, or holding, hardware."""

    config: str
    model: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    gpus: list[str] = field(default_factory=lambda: list(DEFAULT_GPUS))
    count: int = 1
    adapters: int = 1
    """Adapter slots this job needs on whatever cluster serves it."""
    priority: int = 0
    """Higher runs first. Ties break by submission order, so a burst of equal
    jobs stays FIFO instead of reordering itself every tick."""
    spot: bool | None = None
    """True forces spot, False forces on-demand, None takes whichever is
    cheaper. The *submission surfaces* (``JobRouter.submit``, ``evsys queue
    submit``) default to True — with checkpoint+restart in the loop,
    preemptible is the economically correct default and on-demand is the
    opt-in. The dataclass default stays None because spot-only search hid
    real capacity when used for *probing*: Verda had 1x and 2x machines
    purchasable on-demand while every spot probe
    came back empty. Preemption risk is real, but it is the snapshot system's
    problem, not a reason to refuse cheap hardware."""
    state: str = QUEUED
    cluster: str | None = None
    attempts: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str = ""

    @property
    def waiting_s(self) -> float:
        return time.time() - self.created_at

    def describe(self) -> str:
        where = f" on {self.cluster}" if self.cluster else ""
        return (f"{self.id} {self.model} [{'/'.join(self.gpus)}]x{self.count} "
                f"{self.state}{where}")


@dataclass
class Cluster:
    """A live SkyRL server that can accept more adapters.

    ``max_adapters`` is a hard ceiling, not a hint: Megatron allocates the slots
    up front, and asking for a slot beyond it fails inside ``swap_to_adapter``
    with a CUDA error that names nothing about adapters.
    """

    name: str
    model: str
    url: str = ""
    provider: str = ""
    gpu: str = ""
    count: int = 1
    max_adapters: int = 1
    used_adapters: int = 0
    usd_hr: float = 0.0
    launched_at: float = field(default_factory=time.time)

    @property
    def free_adapters(self) -> int:
        return max(self.max_adapters - self.used_adapters, 0)

    def can_take(self, job: Job) -> bool:
        """Same base model, and room for its adapters.

        The model check is not a nicety — adapters on one server share a base
        model, so placing a Qwen3-9B job on a 4B server silently trains against
        the wrong weights. That exact mix-up happened once and the run looked
        fine until the numbers made no sense.
        """
        return self.model == job.model and self.free_adapters >= job.adapters


@dataclass
class Placement:
    """What the scheduler decided, and why. Returned rather than executed so
    the decision can be tested and logged without renting anything."""

    job: Job
    action: str          # "pack" | "rent" | "wait"
    cluster: str | None = None
    capacity: av.Capacity | None = None
    reason: str = ""

    def describe(self) -> str:
        if self.action == "pack":
            return f"pack {self.job.id} onto {self.cluster} (no new GPU)"
        if self.action == "rent":
            c = self.capacity
            return (f"rent {c.provider} {c.gpu}x{c.count} {c.region} "
                    f"${c.usd_hr:.3f}/hr for {self.job.id}" if c else "rent")
        return f"wait: {self.job.id} — {self.reason}"


class Queue:
    """Durable job queue. Append-only log, rebuilt on read.

    Reads replay the log rather than trusting a cached snapshot, so an
    externally-edited or partially-written file converges to the same state a
    scheduler restart would see.
    """

    def __init__(self, path: str = QUEUE_PATH):
        self.path = pathlib.Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, job: Job) -> None:
        job.updated_at = time.time()
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(job)) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def submit(self, config: str, model: str, **kw: Any) -> Job:
        job = Job(config=config, model=model, **kw)
        self._append(job)
        log.info("[queue] submitted %s", job.describe())
        return job

    def jobs(self) -> list[Job]:
        """Current state of every job, latest record wins."""
        if not self.path.exists():
            return []
        latest: dict[str, dict] = {}
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A torn final line means the process died mid-write. Skipping
                # it is right: the job reverts to its previous state and gets
                # retried, which is safe, where guessing would not be.
                continue
            if "id" in rec:
                latest[rec["id"]] = rec
        out = []
        for rec in latest.values():
            known = {k: v for k, v in rec.items() if k in Job.__annotations__}
            out.append(Job(**known))
        return sorted(out, key=lambda j: j.created_at)

    def pending(self) -> list[Job]:
        """Queued work, highest priority first, FIFO within a priority."""
        q = [j for j in self.jobs() if j.state == QUEUED]
        return sorted(q, key=lambda j: (-j.priority, j.created_at))

    def update(self, job: Job, state: str, **kw: Any) -> Job:
        job.state = state
        for k, v in kw.items():
            setattr(job, k, v)
        self._append(job)
        log.info("[queue] %s -> %s", job.id, state)
        return job

    def requeue(self, job: Job, reason: str) -> Job:
        """Put a job back after its machine died. Attempts are counted so a job
        that fails forever is visible instead of silently looping."""
        job.attempts += 1
        return self.update(job, QUEUED, cluster=None, error=reason)

    def compact(self) -> int:
        """Rewrite the log with one record per job. Atomic — a truncated queue
        file is worse than a long one."""
        current = self.jobs()
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            for j in current:
                f.write(json.dumps(asdict(j)) + "\n")
        pathlib.Path(tmp).replace(self.path)
        return len(current)


class Scheduler:
    """Polls availability and places queued work.

    ``clusters`` is supplied by the caller because who owns live clusters
    differs by deployment — a SkyPilot target, a static partner list, or a
    registry. The scheduler only needs to be told what is running.
    """

    def __init__(self, queue: Queue,
                 clusters: Callable[[], list[Cluster]] | None = None,
                 launch: Callable[[Job, av.Capacity], Cluster | None] | None = None,
                 vendors: Iterable[str] | None = None,
                 poll_s: float = 60.0, max_attempts: int = 10):
        self.queue = queue
        self._clusters = clusters or (lambda: [])
        self._launch = launch
        self.vendors = list(vendors) if vendors is not None else None
        self.poll_s = poll_s
        self.max_attempts = max_attempts

    # -- deciding ----------------------------------------------------------

    def plan(self) -> list[Placement]:
        """Decide placements for everything queued, without acting.

        Packing is applied against a *working copy* of cluster occupancy, so
        two jobs planned in the same tick cannot both be promised the last
        adapter slot — a race that would only show up under load, as a CUDA
        error deep inside the second run.
        """
        live = {c.name: c for c in self._clusters()}
        out: list[Placement] = []
        for job in self.queue.pending():
            if job.attempts >= self.max_attempts:
                out.append(Placement(job, "wait",
                                     reason=f"{job.attempts} failed attempts"))
                continue
            host = next((c for c in live.values() if c.can_take(job)), None)
            if host is not None:
                host.used_adapters += job.adapters
                out.append(Placement(job, "pack", cluster=host.name,
                                     reason="model already served"))
                continue
            cap = self._cheapest(job)
            if cap is not None:
                out.append(Placement(job, "rent", capacity=cap))
            else:
                out.append(Placement(job, "wait",
                                     reason=f"no {'/'.join(job.gpus)} free"))
        return out

    def _cheapest(self, job: Job) -> av.Capacity | None:
        """Cheapest live capacity for any accelerator this job accepts.

        Ordered across vendors by price per GPU-hour, so the answer is a real
        comparison and not a preference for whoever we asked first.
        """
        best: av.Capacity | None = None
        for gpu in job.gpus:
            for c in av.scan(gpu, job.count, spot=job.spot,
                             clouds_=self.vendors):
                if not c.ok:
                    continue
                if best is None or (c.usd_per_gpu_hr or 0) < (best.usd_per_gpu_hr or 0):
                    best = c
                break  # scan() is already sorted; the first ok is this gpu's best
        return best

    # -- acting ------------------------------------------------------------

    def tick(self) -> list[Placement]:
        """One pass: decide, then execute what we can."""
        plans = self.plan()
        for p in plans:
            if p.action == "wait":
                log.info("[queue] %s", p.describe())
                continue
            log.info("[queue] %s", p.describe())
            if p.action == "pack":
                self.queue.update(p.job, RUNNING, cluster=p.cluster)
                continue
            if self._launch is None:
                # No launcher wired: the plan is still useful (this is how the
                # scheduler is dry-run), but nothing is rented.
                continue
            self.queue.update(p.job, PLACING)
            try:
                cluster = self._launch(p.job, p.capacity)  # type: ignore[arg-type]
            except Exception as e:  # noqa: BLE001
                self.queue.requeue(p.job, f"launch failed: {e}")
                continue
            if cluster is None:
                # Capacity vanished between the probe and the launch. Routine —
                # availability answers expire in about half a minute — so this
                # is a requeue, not a failure.
                self.queue.requeue(p.job, "capacity gone before launch")
            else:
                self.queue.update(p.job, RUNNING, cluster=cluster.name)
        return plans

    def run(self, *, timeout_s: float | None = None,
            once: bool = False) -> list[Placement]:
        """Keep placing work until the queue drains or the deadline passes.

        ``timeout_s=None`` runs until the queue is empty, which is what an
        overnight batch wants.
        """
        end = (time.time() + timeout_s) if timeout_s is not None else None
        last: list[Placement] = []
        while True:
            last = self.tick()
            if once or not self.queue.pending():
                return last
            if end is not None and time.time() >= end:
                log.info("[queue] deadline reached with %d job(s) still queued",
                         len(self.queue.pending()))
                return last
            time.sleep(self.poll_s)


def skypilot_launcher(**overrides: Any) -> Callable[[Job, av.Capacity], Cluster | None]:
    """A launcher that brings up a SkyRL server on the placed capacity.

    Pins the launch to the exact vendor and region the availability probe found
    free, rather than letting SkyPilot re-plan from its catalog and pick
    somewhere else — the probe is the only thing that looked at live stock.

    Multi-LoRA is on whenever a job asks for more than one adapter, because
    that is what makes packing pay: the cluster is sized for the adapters it
    will host, not the one job that opened it.
    """
    def launch(job: Job, cap: av.Capacity) -> Cluster | None:
        from .skypilot import SkyPilotCompute

        gpu = cap.gpu.replace(" ", "") or job.gpus[0]
        cfg = dict(
            cluster_name=f"skyrl-{job.id}",
            model=job.model,
            infra=f"{cap.provider}/{cap.region}" if cap.region else cap.provider,
            accelerators=f"{gpu}:{cap.count}",
            use_spot=cap.spot,
            multi_lora=job.adapters > 1,
            max_adapters=max(job.adapters, 1),
            retry_until_up=False,   # the queue owns retrying, not the launcher
        )
        cfg.update(overrides)
        target = SkyPilotCompute(**cfg)
        url = target.up()
        if not url:
            return None
        return Cluster(name=cfg["cluster_name"], model=job.model, url=url,
                       provider=cap.provider, gpu=cap.gpu, count=cap.count,
                       max_adapters=max(job.adapters, 1),
                       used_adapters=job.adapters, usd_hr=cap.usd_hr or 0.0)

    return launch


__all__ = ["Cluster", "DEFAULT_GPUS", "DONE", "FAILED", "Job", "PLACING",
           "Placement", "QUEUED", "QUEUE_PATH", "Queue", "RUNNING",
           "Scheduler", "skypilot_launcher"]

"""Surviving preemption — snapshot cadence, resume detection, tenant manifest.

A spot GPU disappears **without notice**. SkyPilot notices by polling cluster
status every 15 s and consumes no provider interruption signal (there is no
AWS ``spot/instance-action`` handler, no GCP metadata watcher, no Azure
scheduled-events poll; Verda and PrimeIntellect document no notice API at all).
So there is no "save on notice" hook to hang a flush off. Periodic snapshotting
is the entire defence, and the only question is *how often*.

Three separable problems, one module:

  * **Cadence** — :class:`SnapshotPolicy` answers "how often" from the classic
    Young/Daly optimum, and :class:`SnapshotScheduler` ticks it against a clock
    while learning the real snapshot cost as it observes it.
  * **Resume detection** — :func:`latest_resumable` picks which checkpoint a
    fresh server should be told to load, from the exact payload SkyRL's
    ``GET /api/v1/training_runs/{id}/checkpoints`` returns.
  * **The manifest** — SkyRL's ``models`` table has ``model_id``,
    ``base_model``, ``lora_config``, ``status``, ``request_id``, ``session_id``
    and ``created_at``, and **no user-metadata column**. Nothing on the server
    remembers that ``model_9fa2c1b0`` was tenant "policy-b" at step 4200 with
    the data cursor 118 k rows in. :class:`ResumeManifest` is where that lives,
    written next to the checkpoints so it dies with neither the instance nor
    the database.

Everything here is pure: no HTTP, no cloud SDK, no filesystem. Callers supply
the payloads and the storage. That is deliberate — it is the part that can be
tested without a GPU, and the part that must not be wrong.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Sequence

from ..logger import get_logger

log = get_logger(__name__)

#: Snapshotting faster than this is self-defeating: the server serialises
#: save_weights against the training queue, so the cost never amortises.
MIN_INTERVAL_S = 30.0

#: SkyRL constrains checkpoint ids to ``^[a-zA-Z0-9_-]+$``, max 255 chars
#: (``skyrl/tinker/api.py:53``). Every id this module mints obeys that.
ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

#: Checkpoint ids carry their step, zero-padded so lexical order matches
#: numeric order even when a reader has nothing else to sort by.
STEP_PREFIX = "step"
_STEP_RE = re.compile(rf"^{STEP_PREFIX}-(\d+)$")

#: SkyRL's two checkpoint types. Only ``training`` is loadable by
#: ``load_weights``; ``sampler`` checkpoints are inference-format exports and
#: resuming from one silently loses the optimizer.
TRAINING = "training"
SAMPLER = "sampler"


def checkpoint_id_for_step(step: int) -> str:
    """``4200`` → ``"step-0000004200"``.

    Zero-padded to ten digits so that sorting ids as strings agrees with
    sorting them as numbers. The server hands checkpoints back in no
    particular order and its only timestamp is ``completed_at`` — which is
    wall-clock, and therefore reorders under a clock skew or a slow write.
    Encoding the step in the id means resume never has to trust a clock.
    """
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    return f"{STEP_PREFIX}-{step:010d}"


def step_of(checkpoint_id: str) -> int | None:
    """The step encoded in a checkpoint id, or None if we did not mint it."""
    m = _STEP_RE.match(checkpoint_id or "")
    return int(m.group(1)) if m else None


# -- cadence ----------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotPolicy:
    """How often to snapshot, and what the answer costs.

    The optimum is Young/Daly. Per unit of wall time, a run pays two
    competing overheads:

      * *taking* snapshots — ``C/T`` (cost ``C`` every interval ``T``), and
      * *redoing* lost work — a preemption lands uniformly inside the
        interval, so it discards ``T/2`` on average, once per ``M`` seconds:
        ``T/(2M)``.

    Differentiating ``C/T + T/(2M)`` and setting it to zero gives
    ``T* = sqrt(2·C·M)``, with total overhead ``sqrt(2C/M)`` there.

    **The hourly rate does not appear.** Both terms are pure time, so both
    scale with $/hr identically and the price cancels. The rate decides
    whether to be on spot at all — never how often to snapshot. Getting this
    backwards ("it's expensive, snapshot more") buys overhead for nothing.

    **Restart latency does not appear either**, and for the opposite reason:
    ``R/M`` is paid per preemption whatever ``T`` is, so it cannot move the
    optimum. It can easily dominate the bill, though — a 15-minute respawn
    against a 2-hour MTBF is 12.5%, larger than everything ``T`` controls.
    That is an argument for a pre-baked image, not for a different cadence.
    """

    snapshot_cost_s: float
    """Wall time one snapshot costs the training loop, end to end.

    Not the upload time: ``save_weights`` is a barrier in SkyRL's queue
    (``_find_destructive_barriers`` treats optim_step/load_weights as
    barriers and the save runs on the same single engine thread), so the
    number that matters is how long the loop is stalled."""
    mtbf_s: float
    """Expected seconds between preemptions on this pool. A guess is fine —
    ``T*`` moves with the square root, so being wrong by 4x moves the answer
    by 2x, and the overhead near the optimum is very flat."""
    max_loss_s: float | None = None
    """Hard cap on work you are willing to redo. Clamps ``T`` regardless of
    what the optimum says — "at most a few minutes" is a requirement, not a
    quantity to be optimised away."""
    restart_s: float = 0.0
    """Time from preemption to a replacement serving again: provision, setup,
    model download, engine warmup. Does not affect ``T``; reported because it
    is usually the largest term in the total."""
    min_interval_s: float = MIN_INTERVAL_S

    def __post_init__(self) -> None:
        if self.snapshot_cost_s <= 0:
            raise ValueError("snapshot_cost_s must be > 0")
        if self.mtbf_s <= 0:
            raise ValueError("mtbf_s must be > 0")

    @property
    def optimal_interval_s(self) -> float:
        """The unclamped Young/Daly optimum, ``sqrt(2·C·M)``."""
        return math.sqrt(2.0 * self.snapshot_cost_s * self.mtbf_s)

    @property
    def interval_s(self) -> float:
        """The interval to actually use: the optimum, clamped both ways.

        ``max_loss_s`` wins over the optimum (a requirement outranks an
        optimisation) and ``min_interval_s`` wins over ``max_loss_s`` (asking
        for less loss than one snapshot costs is not a cadence, it is a
        server that only writes checkpoints).
        """
        t = self.optimal_interval_s
        if self.max_loss_s is not None:
            t = min(t, self.max_loss_s)
        return max(t, self.min_interval_s)

    def overhead_fraction(self, interval_s: float | None = None) -> float:
        """Fraction of wall time lost to snapshotting *and* redone work."""
        t = self.interval_s if interval_s is None else interval_s
        return self.snapshot_cost_s / t + t / (2.0 * self.mtbf_s)

    def restart_fraction(self) -> float:
        """Fraction of wall time lost waiting for replacement capacity."""
        return self.restart_s / self.mtbf_s

    def waste_fraction(self, interval_s: float | None = None) -> float:
        """Everything preemption costs, as a fraction of wall time."""
        return self.overhead_fraction(interval_s) + self.restart_fraction()

    def effective_hourly_cost(self, usd_hr: float) -> float:
        """What an hour of *useful* work really costs at this cadence.

        This is the number to compare against the on-demand rate. A spot
        instance at 40% of on-demand that wastes 60% of its wall time is not
        a discount.
        """
        return usd_hr * (1.0 + self.waste_fraction())

    def expected_loss_s(self) -> float:
        """Work discarded by an average preemption: half an interval."""
        return self.interval_s / 2.0

    def describe(self) -> str:
        clamped = ""
        if self.max_loss_s is not None and self.optimal_interval_s > self.interval_s:
            clamped = (f" (optimum was {self.optimal_interval_s / 60:.1f} min, "
                       f"clamped by max_loss_s)")
        return (f"snapshot every {self.interval_s / 60:.1f} min{clamped} — "
                f"expected loss {self.expected_loss_s() / 60:.1f} min per "
                f"preemption, {self.overhead_fraction() * 100:.1f}% cadence "
                f"overhead + {self.restart_fraction() * 100:.1f}% restart")


class SnapshotScheduler:
    """Ticks a :class:`SnapshotPolicy` against a clock, learning the real cost.

    The seeded ``snapshot_cost_s`` is a guess, and a wrong guess in the
    expensive direction is the bad one: a policy that believes snapshots are
    free will demand them every 30 s, and on a run where each costs two
    minutes that is not a cadence but a stall. So every completed snapshot is
    fed back, and the interval is re-derived from what actually happened.

    The clock is injected. Tests must not sleep, and a scheduler that reads
    the wall clock directly cannot be tested at all.
    """

    #: Weight on the newest observation. High enough to react to a bucket
    #: getting slower within a couple of snapshots, low enough that one
    #: unlucky write does not halve the cadence.
    ALPHA = 0.3

    def __init__(self, policy: SnapshotPolicy,
                 clock: Callable[[], float] | None = None) -> None:
        import time as _time

        self.policy = policy
        self._clock = clock or _time.monotonic
        self._last = self._clock()
        self._observations = 0

    @property
    def interval_s(self) -> float:
        return self.policy.interval_s

    def due(self) -> bool:
        """Has enough time passed to be worth another snapshot?"""
        return self._clock() - self._last >= self.policy.interval_s

    def since_last_s(self) -> float:
        return self._clock() - self._last

    def record(self, duration_s: float) -> SnapshotPolicy:
        """Report a completed snapshot: resets the timer, updates the cost.

        Called with the *measured* wall time, so the policy converges on the
        bucket, the model and the tenant count actually in play rather than
        on whatever the config guessed.
        """
        if duration_s <= 0:
            raise ValueError("duration_s must be > 0")
        prev = self.policy.snapshot_cost_s
        blended = (duration_s if self._observations == 0
                   else (1 - self.ALPHA) * prev + self.ALPHA * duration_s)
        self._observations += 1
        self.policy = replace(self.policy, snapshot_cost_s=blended)
        self._last = self._clock()
        if abs(blended - prev) / prev > 0.25:
            log.info("[snapshot] measured cost %.1fs (was %.1fs) — cadence now "
                     "%.1f min", duration_s, prev, self.policy.interval_s / 60)
        return self.policy

    def skipped(self) -> None:
        """A due snapshot was deliberately not taken; do not retry instantly.

        Without this a failed or declined snapshot leaves ``due()`` true
        forever and the loop tries again every step.
        """
        self._last = self._clock()


# -- resume detection -------------------------------------------------------


@dataclass(frozen=True)
class Checkpoint:
    """One entry of SkyRL's checkpoint listing.

    Mirrors the ``Checkpoint`` model that
    ``GET /api/v1/training_runs/{id}/checkpoints`` returns
    (``skyrl/tinker/api.py:1399``): ``checkpoint_id``, ``checkpoint_type``,
    ``time`` (the DB's ``completed_at``) and ``tinker_path``. That endpoint
    already filters to ``status == COMPLETED``, so a half-written checkpoint
    is never listed — but see :func:`latest_resumable` for why that is not
    the same as it being loadable.
    """

    checkpoint_id: str
    checkpoint_type: str
    tinker_path: str
    time: str | None = None

    @property
    def step(self) -> int | None:
        return step_of(self.checkpoint_id)

    @property
    def is_training(self) -> bool:
        return self.checkpoint_type == TRAINING


def parse_checkpoints(payload: Any) -> list[Checkpoint]:
    """Read the checkpoint-listing response into :class:`Checkpoint` objects.

    Accepts the whole ``{"checkpoints": [...]}`` response or just the list,
    and skips entries missing the two fields resume actually needs. Being
    lenient here is the right trade: a listing that gained a field must not
    make a run unresumable.
    """
    rows = payload.get("checkpoints", []) if isinstance(payload, dict) else payload
    out = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        cid, path = row.get("checkpoint_id"), row.get("tinker_path")
        if not cid or not path:
            continue
        out.append(Checkpoint(checkpoint_id=str(cid),
                              checkpoint_type=str(row.get("checkpoint_type") or ""),
                              tinker_path=str(path),
                              time=row.get("time") and str(row["time"])))
    return out


def latest_resumable(checkpoints: Iterable[Checkpoint]) -> Checkpoint | None:
    """The newest checkpoint a fresh server can actually be resumed from.

    Two filters, both load-bearing:

      * **Training checkpoints only.** ``load_weights`` rejects anything whose
        tinker path is not ``…/weights/…`` (``api.py:979``), and a sampler
        checkpoint is an inference-format export — resuming from one would at
        best restart the optimizer, at worst 400.
      * **Ours only.** A checkpoint whose id does not encode a step is one we
        did not mint, so we cannot say where it sits in the run. Guessing
        from ``completed_at`` means trusting a wall clock across a machine
        that has just been replaced; if two checkpoints disagree, the clock
        is exactly the witness not to believe.

    Returns None when there is nothing to resume from — which is the normal
    state of a run's first minute, not an error.
    """
    ours = [c for c in checkpoints if c.is_training and c.step is not None]
    if not ours:
        return None
    return max(ours, key=lambda c: c.step or 0)


def resume_target(payload: Any) -> tuple[str, int] | None:
    """``(tinker_path, step)`` to resume from, straight from the listing.

    The path goes to ``POST /api/v1/load_weights`` as ``path``, alongside the
    **new** model's id — not the old one. Resume is never "reattach to
    ``model_id``": ``create_model`` mints a fresh random id every time
    (``api.py:824``), and a fresh server's backend holds no models at all, so
    every request naming the dead id fails ``has_model``. The old id survives
    only inside the path, as ``load_weights``'s ``source_model_id``.
    """
    best = latest_resumable(parse_checkpoints(payload))
    return (best.tinker_path, best.step or 0) if best else None


# -- the manifest -----------------------------------------------------------


@dataclass(frozen=True)
class TenantState:
    """Everything about one LoRA tenant that the server does not remember.

    ``models`` stores ``model_id``, ``base_model``, ``lora_config``,
    ``status``, ``request_id``, ``session_id``, ``created_at`` — and no
    user-supplied name, no step, no data position. On a multi-tenant server
    that is the difference between "eight adapters exist" and "adapter three
    is the one that was training the router policy".
    """

    name: str
    """Our logical name for the tenant. Stable across preemptions; the
    server-side ``model_id`` is not."""
    model_id: str
    """The id this tenant had before the snapshot. Dead after a restart —
    kept because it is what forms the ``tinker://`` source path."""
    checkpoint_path: str
    """``tinker://<model_id>/weights/<checkpoint_id>`` — what to hand
    ``load_weights``."""
    step: int
    data_cursor: int = 0
    """Position in the training stream. Without it, resume replays data the
    optimizer has already seen, which is not a resume but a second epoch on
    a prefix."""
    seed: int | None = None
    lora_rank: int | None = None
    base_model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"name": self.name, "model_id": self.model_id,
             "checkpoint_path": self.checkpoint_path, "step": self.step,
             "data_cursor": self.data_cursor, "seed": self.seed,
             "lora_rank": self.lora_rank, "base_model": self.base_model}
        if self.extra:
            d["extra"] = dict(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TenantState:
        return cls(name=str(d["name"]), model_id=str(d["model_id"]),
                   checkpoint_path=str(d["checkpoint_path"]),
                   step=int(d["step"]), data_cursor=int(d.get("data_cursor") or 0),
                   seed=d.get("seed"), lora_rank=d.get("lora_rank"),
                   base_model=d.get("base_model"), extra=dict(d.get("extra") or {}))


class ManifestCorrupt(ValueError):
    """A manifest object did not survive the trip. Fall back a generation."""


@dataclass(frozen=True)
class ResumeManifest:
    """The run's own record of who was training what, and how far.

    Written to the same durable store as the checkpoints, because the two
    have to be lost together or not at all. A checkpoint the manifest does
    not name is an orphan; a manifest naming a checkpoint that is not there
    is worse.

    **Immutable per generation.** Object stores offer no atomic rename, so
    "overwrite manifest.json" has a window in which readers see a truncated
    object. A single PUT, on the other hand, is all-or-nothing: the object
    either exists whole or does not exist. So every snapshot writes a *new*
    object, ``manifest-<generation>.json``, and readers take the
    highest-numbered one that validates. A torn or half-uploaded top
    generation simply loses to the one below it.

    ``owner`` is a fencing token. Two servers can be alive at once — SkyPilot
    decides a cluster is gone by *polling*, so a control-plane hiccup can
    relaunch a job whose original is still running and still writing. Both
    then write generation N+1 and one silently wins. Recording who wrote a
    generation at least makes that detectable: a writer that reads back a
    generation it did not write knows it has been superseded and must stop.
    """

    generation: int
    run_id: str
    tenants: dict[str, TenantState]
    step: int = 0
    owner: str | None = None
    created_at: str | None = None

    #: Bumped only for changes a previous reader could not survive.
    VERSION = 1

    def payload(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "generation": self.generation,
            "run_id": self.run_id,
            "step": self.step,
            "owner": self.owner,
            "created_at": self.created_at,
            "tenants": [t.to_dict() for t in self.tenants.values()],
        }

    def to_json(self) -> str:
        """Serialise with a checksum over the payload.

        The checksum is not paranoia about bit rot; it is the cheapest way to
        recognise a *truncated* object, which is what a process killed
        mid-upload leaves behind on stores that do not guarantee atomic
        visibility (and what a local filesystem leaves behind always).
        """
        body = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(body.encode()).hexdigest()
        return json.dumps({"checksum": digest, "payload": json.loads(body)},
                          sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> ResumeManifest:
        """Parse and verify. Raises :class:`ManifestCorrupt` on any doubt."""
        try:
            outer = json.loads(text)
            payload = outer["payload"]
            claimed = outer["checksum"]
        except Exception as e:
            raise ManifestCorrupt(f"unreadable manifest: {e}") from e
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        actual = hashlib.sha256(body.encode()).hexdigest()
        if actual != claimed:
            raise ManifestCorrupt("checksum mismatch — the object is torn or "
                                  "was written by a different serialiser")
        if int(payload.get("version", 0)) > cls.VERSION:
            raise ManifestCorrupt(
                f"manifest version {payload['version']} is newer than this SDK "
                f"understands ({cls.VERSION})")
        tenants = {t["name"]: TenantState.from_dict(t)
                   for t in payload.get("tenants") or []}
        return cls(generation=int(payload["generation"]),
                   run_id=str(payload["run_id"]), tenants=tenants,
                   step=int(payload.get("step") or 0),
                   owner=payload.get("owner"),
                   created_at=payload.get("created_at"))

    def succeed(self, tenants: dict[str, TenantState], *, step: int | None = None,
                owner: str | None = None, created_at: str | None = None) -> ResumeManifest:
        """The next generation, with the state as of now."""
        return ResumeManifest(generation=self.generation + 1, run_id=self.run_id,
                              tenants=dict(tenants),
                              step=self.step if step is None else step,
                              owner=owner if owner is not None else self.owner,
                              created_at=created_at)

    def object_name(self) -> str:
        return manifest_name(self.generation)


#: Manifest objects are ``manifest-<generation>.json``, zero-padded so a
#: plain lexical listing is already in generation order.
_MANIFEST_RE = re.compile(r"^manifest-(\d{10})\.json$")


def manifest_name(generation: int) -> str:
    if generation < 0:
        raise ValueError(f"generation must be non-negative, got {generation}")
    return f"manifest-{generation:010d}.json"


def manifest_generation(name: str) -> int | None:
    """The generation encoded in an object name, or None if it is not one."""
    m = _MANIFEST_RE.match((name or "").rsplit("/", 1)[-1])
    return int(m.group(1)) if m else None


def latest_manifest(names: Sequence[str],
                    load: Callable[[str], str]) -> ResumeManifest | None:
    """Newest *valid* manifest from a listing, walking down on corruption.

    A crash during the upload of generation N leaves an object that lists but
    does not parse. Refusing to resume at that point would mean losing a run
    to the one write that was interrupted — precisely the moment the manifest
    exists to cover. So a bad generation is logged and skipped, and the run
    resumes from N-1, redoing one interval's work. Returns None only when
    nothing valid remains, which is also the answer for a fresh run.

    ``load`` is injected so this works against a bucket, a filesystem or a
    dict without this module knowing which.
    """
    candidates = sorted(
        ((g, n) for n in names if (g := manifest_generation(n)) is not None),
        reverse=True)
    for generation, name in candidates:
        try:
            return ResumeManifest.from_json(load(name))
        except ManifestCorrupt as e:
            log.warning("[snapshot] manifest generation %d is unusable (%s) — "
                        "falling back to the one before it", generation, e)
        except Exception as e:
            log.warning("[snapshot] could not read manifest %s: %s", name, e)
    return None


__all__ = [
    "MIN_INTERVAL_S", "SAMPLER", "TRAINING",
    "Checkpoint", "ManifestCorrupt", "ResumeManifest", "SnapshotPolicy",
    "SnapshotScheduler", "TenantState",
    "checkpoint_id_for_step", "latest_manifest", "latest_resumable",
    "manifest_generation", "manifest_name", "parse_checkpoints",
    "resume_target", "step_of",
]

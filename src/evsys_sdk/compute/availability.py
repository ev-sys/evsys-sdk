"""Is there actually a GPU to rent right now — asked per vendor, answered honestly.

SkyPilot has no abstraction for this, and the omission is deliberate on its
part. Its catalog (``~/.sky/catalogs/<v>/<cloud>/vms.csv``) carries prices and
machine shapes only; ``regions_with_offering()`` answers "which regions *sell*
this SKU", not "which have one free". Capacity is discovered inside each cloud's
``run_instances`` — you find out by failing, then failing over.

That is defensible for a tool that must work across twenty clouds. It is not
enough for us: we chase spot capacity across vendors continuously, and "launch
and see" costs a provisioning round-trip per guess, burns quota, and gives no
signal to order the next attempt by. So availability becomes its own extension
point, alongside `pricing` and `liveness`.

Three design decisions, each of which cost real money to learn:

  * **Three states, never a bool.** ``UNKNOWN`` is not ``UNAVAILABLE``. A probe
    that times out must not look identical to a provider saying "nothing free" —
    that conflation is what had monitors reporting "no capacity" for an hour
    while the real answer was "we could not reach the API". Same lesson as
    :mod:`.liveness`.
  * **Answers expire.** Every SKU reported free was refused ~30s later, over and
    over. A capacity answer is a *hint with a timestamp*, so it carries one and
    :meth:`Capacity.fresh` is the only way to trust it. Nothing here is
    authoritative — only the launch call is. This layer decides *what to try
    first* and *when to bother trying*.
  * **Ask the vendor the question you mean.** Verda's ``is_spot`` defaults to
    ``"false"``, so the obvious call asks about on-demand stock and disagreed
    with spot launches all afternoon. The mirror of that mistake is just as
    expensive: searching *only* spot hid every on-demand 1x and 2x machine
    while we reported nothing free. Hence ``spot=None`` — both modes — is the
    default here, and price decides.

Scaling to a new vendor is deliberately near-free. Anything already in
``pricing.PROBES`` gets listing-derived availability with no code at all, via
:class:`OffersProbe`. A vendor with a real capacity endpoint overrides one
method::

    @register_availability("lambda")
    class LambdaAvailability(AvailabilityProbe):
        name = "lambda"
        def probe(self, gpu, count, region, spot):
            ...  # return AVAILABLE / UNAVAILABLE / UNKNOWN

Usage::

    from evsys_sdk.compute import availability as av

    av.check("verda", "H200", 1)            # one vendor, both purchase modes
    av.scan("H100", 2, spot=True)           # every vendor, spot only
    av.wait_for("H200", 1, timeout_s=3600)  # block until something frees up
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..logger import get_logger
from ..registry import get_availability, list_availabilities, register_availability
from . import pricing

log = get_logger(__name__)

#: There is stock we can buy.
AVAILABLE = "available"
#: The vendor says no. A real answer, not a failure to get one.
UNAVAILABLE = "unavailable"
#: We could not find out. Never treat this as UNAVAILABLE — see module docstring.
UNKNOWN = "unknown"

#: How long a capacity answer is worth anything. Measured behaviour: SKUs that
#: probed free were refused roughly half a minute later, repeatedly. Anything
#: longer than this is a guess wearing a timestamp.
DEFAULT_TTL_S = 45.0


@dataclass(frozen=True)
class Capacity:
    """One vendor's answer about one shape, with the time it was true."""

    provider: str
    gpu: str
    count: int
    state: str
    region: str | None = None
    sku: str | None = None
    spot: bool = True
    usd_hr: float | None = None
    checked_at: float = field(default_factory=time.time)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.state == AVAILABLE

    def fresh(self, ttl_s: float = DEFAULT_TTL_S) -> bool:
        """Whether this answer is recent enough to act on."""
        return (time.time() - self.checked_at) < ttl_s

    @property
    def usd_per_gpu_hr(self) -> float | None:
        if self.usd_hr is None:
            return None
        return self.usd_hr / max(self.count, 1)

    def describe(self) -> str:
        where = f" {self.region}" if self.region else ""
        price = f" ${self.usd_hr:.3f}/hr" if self.usd_hr else ""
        kind = "spot" if self.spot else "on-demand"
        tail = f" ({self.detail})" if self.detail else ""
        return (f"{self.provider} {self.gpu}x{self.count}{where} {kind}: "
                f"{self.state}{price}{tail}")


class AvailabilityProbe:
    """What a vendor must answer to participate in capacity search.

    Subclasses override :meth:`probe` only. Everything else — never raising,
    stamping answers, turning exceptions into ``UNKNOWN`` — is handled by
    :meth:`check` so a flaky vendor cannot take down a multi-vendor scan.
    """

    name: str = ""

    def probe(self, gpu: str, count: int, region: str | None,
              spot: bool | None) -> list[Capacity]:
        """Return one :class:`Capacity` per (region, sku) considered.

        ``spot=None`` means both purchase modes. Raise freely — :meth:`check`
        converts failures into ``UNKNOWN``.
        """
        raise NotImplementedError

    def check(self, gpu: str, count: int = 1, region: str | None = None,
              spot: bool | None = None) -> list[Capacity]:
        """Safe wrapper: never raises, never returns an empty-means-no answer."""
        try:
            out = self.probe(gpu, count, region, spot)
        except Exception as e:
            log.debug("[availability] %s probe failed: %s", self.name, e)
            return [Capacity(self.name, gpu, count, UNKNOWN, region=region,
                             spot=bool(spot),
                             detail=f"probe failed: {type(e).__name__}")]
        if not out:
            # An empty list is ambiguous — the vendor may sell nothing of this
            # shape, or may have returned nothing. Say UNAVAILABLE explicitly so
            # callers never have to guess what empty meant.
            return [Capacity(self.name, gpu, count, UNAVAILABLE, region=region,
                             spot=bool(spot), detail="no matching offers")]
        return out


class OffersProbe(AvailabilityProbe):
    """Availability derived from a vendor's price listing.

    The zero-work path for a new vendor: anything with a `pricing` probe gets
    capacity search for free. Weaker than a real endpoint — a listing tells you
    what is *sold*, and only some vendors flag what is *free* — so where a
    vendor exposes a capacity API, override :meth:`probe` instead.
    """

    def probe(self, gpu: str, count: int, region: str | None,
              spot: bool | None) -> list[Capacity]:
        offers = pricing.live_offers(self.name, gpu, count, spot=spot)
        out = []
        for o in offers:
            if region and o.region != region:
                continue
            out.append(Capacity(
                provider=self.name, gpu=gpu, count=count,
                state=AVAILABLE if o.available else UNAVAILABLE,
                region=o.region, usd_hr=o.usd_hr, spot=o.spot,
                detail="from price listing"))
        return out


@register_availability("verda")
class VerdaAvailability(AvailabilityProbe):
    """Verda's authoritative per-SKU endpoint.

    ``GET /instance-availability/{sku}?is_spot=…&location_code=…`` returns a
    bare JSON boolean. The aggregate listing at the same path *without* a SKU
    answers about on-demand stock unless ``is_spot`` is passed, which is why it
    disagreed with every spot launch attempt until that parameter was added.
    """

    name = "verda"

    def probe(self, gpu: str, count: int, region: str | None,
              spot: bool | None) -> list[Capacity]:
        from .providers_verda import VerdaProvider

        p = VerdaProvider()
        out: list[Capacity] = []
        for o in p.offers(gpu=gpu, count=count, spot=spot):
            if region and o.region != region:
                continue
            try:
                free = p.available(o.sku, o.region, spot=o.spot)
                state = AVAILABLE if free else UNAVAILABLE
                detail = "per-sku endpoint"
            except Exception as e:
                state, detail = UNKNOWN, f"sku check failed: {type(e).__name__}"
            out.append(Capacity(provider=self.name, gpu=o.gpu, count=o.count,
                                state=state, region=o.region, sku=o.sku,
                                usd_hr=o.usd_hr, spot=o.spot, detail=detail))
        return out


@register_availability("primeintellect")
class PrimeIntellectAvailability(OffersProbe):
    """PrimeIntellect publishes stock in its offer listing, so the generic
    listing-derived probe is already the best answer available."""

    name = "primeintellect"


@register_availability("nebius")
class NebiusAvailability(AvailabilityProbe):
    """Nebius's capacity advisor — the strongest answer any vendor here gives.

    ``ResourceAdviceService.List`` reports, per (region, platform, preset),
    how many on-demand and preemptible VMs the tenant could launch right now,
    with a confidence level. ``LIMIT_REACHED``/0 maps to UNAVAILABLE, stale
    data to UNKNOWN, anything launchable to AVAILABLE. Prices come from the
    published per-GPU-hour catalog (flat across counts, like Verda).
    """

    name = "nebius"

    def probe(self, gpu: str, count: int, region: str | None,
              spot: bool | None) -> list[Capacity]:
        from .providers_nebius import PLATFORMS, NebiusProvider

        p = NebiusProvider()
        want = "".join(c for c in (gpu or "").lower() if c.isalnum())
        out: list[Capacity] = []
        for item in p.advice():
            spec = item.get("spec") or {}
            ci = spec.get("computeInstance") or spec.get("compute_instance") or {}
            platform = ci.get("platform") or ""
            cat = PLATFORMS.get(platform)
            if cat is None or not str(cat["gpu"]).lower().startswith(want):
                continue
            preset = (ci.get("preset") or {}).get("name") or ""
            n_gpus = ((ci.get("preset") or {}).get("resources") or {}).get(
                "gpuCount") or next(
                (n for n, name in cat["presets"].items() if name == preset), 0)
            if count and n_gpus != count:
                continue
            reg = spec.get("region") or ""
            if region and reg != region:
                continue
            status = item.get("status") or {}
            for is_spot in ((True, False) if spot is None else (spot,)):
                adv = status.get("preemptible" if is_spot else "onDemand") \
                    or status.get("preemptible" if is_spot else "on_demand") \
                    or {}
                level = adv.get("availabilityLevel") \
                    or adv.get("availability_level") or ""
                free = int(adv.get("available") or 0)
                if adv.get("dataState", adv.get("data_state")) \
                        == "DATA_STATE_UNKNOWN":
                    state = UNKNOWN
                elif free > 0 and "LIMIT_REACHED" not in level:
                    state = AVAILABLE
                else:
                    state = UNAVAILABLE
                price = cat["usd_gpu_hr_spot" if is_spot else "usd_gpu_hr"]
                out.append(Capacity(
                    provider=self.name, gpu=cat["gpu"], count=n_gpus,
                    state=state, region=reg, sku=f"{platform}/{preset}",
                    usd_hr=price * n_gpus, spot=is_spot,
                    detail=f"advisor: {free} launchable, {level or '?'}"))
        return out


def _probe(cloud: str) -> AvailabilityProbe:
    """Resolve a probe, falling back to the listing-derived one.

    A vendor that has pricing but no registered availability probe still gets
    searched — that fallback is the whole scalability story.
    """
    try:
        return get_availability(cloud)()
    except KeyError:
        if cloud in pricing.PROBES:
            p = OffersProbe()
            p.name = cloud
            return p
        raise


def clouds() -> list[str]:
    """Every vendor we can ask, registered probes plus pricing fallbacks."""
    return sorted(set(list_availabilities()) | set(pricing.PROBES))


def check(cloud: str, gpu: str, count: int = 1, region: str | None = None,
          spot: bool | None = None) -> list[Capacity]:
    """Ask one vendor. Never raises for a reachability problem.

    ``spot=None`` (the default) asks about **both** purchase modes. Defaulting
    to spot-only hid real capacity: Verda's API defaults ``is_spot`` to false,
    and on-demand 1x and 2x machines were purchasable the whole time we were
    reporting nothing free.
    """
    return _probe(cloud).check(gpu, count, region, spot)


def scan(gpu: str, count: int = 1, *, spot: bool | None = None,
         clouds_: Iterable[str] | None = None,
         region: str | None = None) -> list[Capacity]:
    """Ask every vendor, cheapest purchasable first.

    Ordering is the point. ``AVAILABLE`` sorts ahead of ``UNKNOWN``, which sorts
    ahead of ``UNAVAILABLE`` — an unknown is worth an attempt, a refusal is not
    — and within a state, cheapest per GPU-hour wins. Feed the result straight
    into a launch loop.
    """
    rank = {AVAILABLE: 0, UNKNOWN: 1, UNAVAILABLE: 2}
    out: list[Capacity] = []
    for c in (clouds_ if clouds_ is not None else clouds()):
        out.extend(check(c, gpu, count, region, spot))
    return sorted(out, key=lambda c: (rank.get(c.state, 3),
                                      c.usd_per_gpu_hr if c.usd_per_gpu_hr
                                      else float("inf")))


def available(gpu: str, count: int = 1, **kw: Any) -> list[Capacity]:
    """Just the purchasable ones, cheapest first."""
    return [c for c in scan(gpu, count, **kw) if c.ok]


def wait_for(gpu: str, count: int = 1, *, timeout_s: float | None = 3600,
             poll_s: float = 60, spot: bool | None = None,
             clouds_: Iterable[str] | None = None,
             region: str | None = None,
             on_poll: Callable[[list[Capacity]], None] | None = None,
             ) -> list[Capacity]:
    """Poll until something is purchasable, or the deadline passes.

    Returns the ranked candidates as soon as any vendor says yes — plural,
    because by the time you launch, the first one may be gone and the caller
    should walk the list rather than re-poll from scratch.

    ``timeout_s=None`` waits forever, which is what an overnight capacity hunt
    wants. ``timeout_s=0`` is a single immediate check, not an infinite loop —
    the same ``if deadline`` bug that :mod:`.liveness` had.
    """
    end = (time.time() + timeout_s) if timeout_s is not None else None
    n = 0
    while True:
        n += 1
        found = scan(gpu, count, spot=spot, clouds_=clouds_, region=region)
        if on_poll:
            on_poll(found)
        if any(c.ok for c in found):
            log.info("[availability] %s x%d found after %d poll(s): %s",
                     gpu, count, n, found[0].describe())
            return found
        if end is not None and time.time() >= end:
            log.info("[availability] gave up on %s x%d after %d poll(s)",
                     gpu, count, n)
            return found
        time.sleep(poll_s)


__all__ = ["AVAILABLE", "DEFAULT_TTL_S", "UNAVAILABLE", "UNKNOWN",
           "AvailabilityProbe", "Capacity", "OffersProbe",
           "PrimeIntellectAvailability", "VerdaAvailability", "available",
           "check", "clouds", "register_availability", "scan", "wait_for"]

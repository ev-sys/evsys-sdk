"""Live GPU prices and availability, straight from the provider.

SkyPilot picks instances from a **pre-generated catalog** — a hosted CSV, not a
live query. It is the right structure for planning across a dozen clouds, but
it goes stale, and stale in both directions:

  * *Price.* The catalog offered PrimeIntellect H100 at $1.97/hr and H200 at
    $2.00/hr; the provider's own API said $3.25 and $4.00 — a 65–100% error, in
    the direction that flatters the plan. Any "can we beat the hosted service"
    arithmetic done on catalog prices is wrong before it starts.

  * *Existence.* The catalog happily routes to capacity that is gone. Asking
    for an H100 sent SkyPilot to hyperstack in CA, which had no stock, and it
    reported ``ResourcesUnavailableError`` — the same error it raises for an
    unpayable wallet or an unsupported feature. Three very different failures,
    one indistinguishable message.

So we ask the provider directly before spending anything. This does not
replace the catalog: SkyPilot still plans and provisions. It is a pre-flight
check that answers three questions the catalog cannot — is there really
capacity, what does it really cost, and which region actually has it.

Providers are looked up by SkyPilot's cloud name; an unknown one returns no
offers and callers fall back to catalog behaviour unchanged.
"""

from __future__ import annotations

import json
import pathlib
import urllib.request
from dataclasses import dataclass
from typing import Callable

from ..logger import get_logger

log = get_logger(__name__)

REQUEST_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class Offer:
    """One purchasable configuration, as the provider reports it right now."""

    provider: str
    region: str
    gpu: str
    count: int
    usd_hr: float
    spot: bool
    available: bool

    def describe(self) -> str:
        kind = "spot" if self.spot else "on-demand"
        stock = "available" if self.available else "OUT OF STOCK"
        return (f"${self.usd_hr:.4f}/hr {self.gpu}x{self.count} {kind} "
                f"{self.provider}/{self.region} ({stock})")


class PricingUnavailable(RuntimeError):
    """The provider could not be asked — not the same as 'no capacity'."""


# -- PrimeIntellect ---------------------------------------------------------

PRIME_API = "https://api.primeintellect.ai/api/v1/availability/"
PRIME_CREDENTIALS = "~/.prime/config.json"

#: SkyPilot spells accelerators one way, PrimeIntellect another, and the
#: memory suffix is not optional on their side: a bare "A100" silently means
#: the 40 GB part, which is a different machine at a different price.
PRIME_GPU_NAMES = {
    "A100": "A100_40GB",
    "A100-80GB": "A100_80GB",
    "H100": "H100_80GB",
    "H200": "H200_141GB",
    "A10": "A10_24GB",
    "A40": "A40_48GB",
    "A6000": "A6000_48GB",
    "L40S": "L40S_48GB",
    "RTX6000Ada": "RTX6000Ada_48GB",
}


def _prime_key() -> str:
    path = pathlib.Path(PRIME_CREDENTIALS).expanduser()
    try:
        return json.loads(path.read_text())["api_key"]
    except Exception as e:
        raise PricingUnavailable(
            f"could not read a PrimeIntellect API key from {PRIME_CREDENTIALS}: {e}"
        ) from e


def _prime_offers(gpu: str, count: int) -> list[Offer]:
    gpu_type = PRIME_GPU_NAMES.get(gpu)
    if gpu_type is None:
        raise PricingUnavailable(f"no PrimeIntellect name known for GPU {gpu!r}")
    req = urllib.request.Request(
        f"{PRIME_API}?gpu_type={gpu_type}",
        headers={"Authorization": f"Bearer {_prime_key()}"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
            payload = json.load(r)
    except Exception as e:
        raise PricingUnavailable(f"PrimeIntellect availability query failed: {e}") from e

    offers: list[Offer] = []
    for entries in (payload or {}).values():
        for o in entries:
            if o.get("gpuCount") != count:
                continue
            prices = o.get("prices") or {}
            # `price` does not exist on this payload — reading it yields None
            # for every offer and silently reports "no capacity".
            usd = prices.get("onDemand") or prices.get("communityPrice")
            if usd is None:
                continue
            # Spot is not a flag: it is a separate SKU whose id ends in _SPOT,
            # priced in the same `onDemand` field as everything else.
            spot = str(o.get("cloudId") or "").upper().endswith("_SPOT")
            offers.append(Offer(
                provider=str(o.get("provider") or "?"),
                region=str(o.get("dataCenter") or o.get("country") or "?"),
                gpu=gpu, count=count, usd_hr=float(usd), spot=spot,
                available=str(o.get("stockStatus") or "").lower() == "available",
            ))
    return sorted(offers, key=lambda o: o.usd_hr)


#: Keyed by SkyPilot's cloud name. Add a probe here and every compute target
#: that consults live pricing picks it up.
PROBES: dict[str, Callable[[str, int], list[Offer]]] = {
    "primeintellect": _prime_offers,
}


def live_offers(cloud: str, gpu: str, count: int = 1,
                spot: bool | None = None) -> list[Offer]:
    """Current offers for ``gpu`` on ``cloud``, cheapest first.

    Returns ``[]`` for a provider with no probe, so callers degrade to plain
    catalog behaviour rather than refusing to launch. Raises
    :class:`PricingUnavailable` only when a probe exists but could not answer —
    a distinction that matters, because "I could not ask" must never be
    reported to a user as "there is none".
    """
    probe = PROBES.get(cloud.split("/")[0].lower())
    if probe is None:
        return []
    offers = probe(gpu, count)
    if spot is not None:
        offers = [o for o in offers if o.spot == spot]
    return offers


def cheapest_available(cloud: str, gpu: str, count: int = 1,
                       spot: bool | None = None) -> Offer | None:
    """The cheapest offer that is actually in stock, or None."""
    return next((o for o in live_offers(cloud, gpu, count, spot) if o.available), None)


__all__ = ["Offer", "PricingUnavailable", "cheapest_available", "live_offers"]

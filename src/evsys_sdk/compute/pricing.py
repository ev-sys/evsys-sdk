"""Live GPU prices and availability, straight from the provider.

SkyPilot picks instances from a **pre-generated catalog** — a hosted CSV, not a
live query. It is the right structure for planning across a dozen clouds, but
it goes stale, and stale in both directions:

  * *Price.* The catalog offered PrimeIntellect H100 at $1.97/hr and H200 at
    $2.00/hr; the provider's own API said $3.25 and $4.00 — a 65-100% error, in
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
from collections.abc import Callable
from dataclasses import dataclass

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


# -- Verda (formerly DataCrunch) --------------------------------------------

VERDA_API = "https://api.verda.com/v1"
VERDA_CREDENTIALS = "~/.verda/config.json"

#: Verda names a GPU inside a free-text instance description, so match on the
#: model field instead of parsing strings.
VERDA_GPU_MODELS = {
    "A100": "A100", "A100-80GB": "A100", "H100": "H100", "H200": "H200",
    "L40S": "L40S", "A6000": "A6000", "V100": "V100", "B200": "B200",
    "RTX6000Ada": "RTX6000ADA",
}

#: GPU memory in GB implied by a SkyPilot name, where the name carries it.
#: Verda lists an 80 GB and a 40 GB A100 as separate types under one model.
VERDA_GPU_MEMORY = {"A100-80GB": 80, "A100": 40}


def _verda_token() -> str:
    """OAuth2 client-credentials exchange. Verda issues short-lived tokens
    (~10 min), so this is done per call rather than cached."""
    path = pathlib.Path(VERDA_CREDENTIALS).expanduser()
    try:
        cfg = json.loads(path.read_text())
        body = json.dumps({"grant_type": "client_credentials",
                           "client_id": cfg["client_id"],
                           "client_secret": cfg["client_secret"]}).encode()
    except Exception as e:
        raise PricingUnavailable(
            f"could not read Verda credentials from {VERDA_CREDENTIALS}: {e}") from e
    req = urllib.request.Request(f"{VERDA_API}/oauth2/token", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
            return json.load(r)["access_token"]
    except Exception as e:
        raise PricingUnavailable(f"Verda token exchange failed: {e}") from e


def _verda_get(path: str, token: str):
    req = urllib.request.Request(f"{VERDA_API}/{path}",
                                 headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
            return json.load(r)
    except Exception as e:
        raise PricingUnavailable(f"Verda {path} query failed: {e}") from e


def _verda_offers(gpu: str, count: int) -> list[Offer]:
    model = VERDA_GPU_MODELS.get(gpu)
    if model is None:
        raise PricingUnavailable(f"no Verda model known for GPU {gpu!r}")
    token = _verda_token()
    types = _verda_get("instance-types", token)
    # Availability is per location and lists instance-type ids, so stock has to
    # be joined on rather than read off the type.
    stock: dict[str, list[str]] = {}
    for loc in _verda_get("instance-availability", token):
        for it in loc.get("availabilities") or []:
            stock.setdefault(it, []).append(loc["location_code"])

    want_mem = VERDA_GPU_MEMORY.get(gpu)
    offers: list[Offer] = []
    for t in types:
        g = t.get("gpu") or {}
        if t.get("model") != model or g.get("number_of_gpus") != count:
            continue
        if want_mem is not None:
            per_gpu = (t.get("gpu_memory") or {}).get("size_in_gigabytes")
            if per_gpu and count and round(per_gpu / count) != want_mem:
                continue
        it = t["instance_type"]
        regions = stock.get(it) or ["-"]
        for kind, price in (("spot", t.get("spot_price")),
                            ("ondemand", t.get("price_per_hour"))):
            if not price or float(price) <= 0:
                continue
            for region in regions:
                offers.append(Offer(
                    provider="verda", region=region, gpu=gpu, count=count,
                    usd_hr=float(price), spot=(kind == "spot"),
                    available=it in stock,
                ))
    return sorted(offers, key=lambda o: o.usd_hr)


#: Keyed by SkyPilot's cloud name. Add a probe here and every compute target
#: that consults live pricing picks it up.
PROBES: dict[str, Callable[[str, int], list[Offer]]] = {
    "primeintellect": _prime_offers,
    "verda": _verda_offers,
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

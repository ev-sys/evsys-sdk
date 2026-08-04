"""Vast.ai — a marketplace, so availability and price are the same question.

Verda sells fixed SKUs at a fixed spot price and answers "is one free" with a
boolean. Vast is an auction over other people's machines, and that changes the
model in ways worth spelling out, because reusing Verda's assumptions here
produces confidently wrong numbers:

  * **The search IS the availability query.** Every offer returned with
    ``rentable`` is available right now. There is no separate endpoint to ask,
    and no aggregate listing to disagree with.
  * **``dph_total`` is per-offer, not per-GPU.** Measured: the cheapest 1x H100
    offer was $0.99/hr and the cheapest 8x was $18.38/hr — $0.99 vs $2.30 per
    GPU-hour. Comparing raw ``dph_total`` across counts ranks an 8x box as 18x
    worse than it is.
  * **The interruptible price is a decision, not a fact.** ``min_bid`` is the
    floor you must clear, and bidding higher makes you *less* likely to be
    outbid and preempted. So the price we record for spot is the floor, and
    what you actually pay is a risk choice — unlike Verda, where the spot price
    is simply the price.
  * **Hosts are individually unreliable and Vast says so.** ``reliability2`` is
    a per-host score; a marketplace has no uniform SLA, so a cheap offer from a
    0.90 host is a different product from the same price at 0.999.

Searching needs no credentials. **Renting does** — set ``VAST_API_KEY`` or
write ``~/.vast/config.json`` with ``{"api_key": "..."}``, and register an SSH
key with the account first (``GET /api/v0/ssh/``), or the instance comes up
with no way in.
"""

from __future__ import annotations

import json
import os
import pathlib
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..logger import get_logger
from ..registry import register_availability
from .availability import (AVAILABLE, UNAVAILABLE, AvailabilityProbe, Capacity)
from .provider import (GONE, PENDING, RUNNING, LaunchFailed, Machine, Offer,
                       Provider)

log = get_logger(__name__)

API = "https://console.vast.ai/api/v0"
CREDENTIALS = "~/.vast/config.json"

#: Vast writes GPU names its own way, and ``gpu_name`` filters on **exact
#: equality** — there is no prefix or substring match. So a family maps to
#: every exact spelling it ships under, queried with ``in``. Asking for
#: ``{"eq": "H100"}`` matches nothing at all, because no offer is named that;
#: it silently reported "Vast sells no H100s" while 30 were rentable.
#:
#: The variants are genuinely different cards at different prices — H100 PCIE
#: bid at $0.400/hr against H100 SXM at $1.480 — so the full name is kept on
#: every Offer rather than collapsed into the family.
#:
#: Verified empirically against the live marketplace. RTX PRO 6000 is
#: deliberately absent: Vast returns zero offers for it under any spelling,
#: so there is no name to guess.
GPU_ALIASES = {
    "H100": ["H100 SXM", "H100 PCIE", "H100 NVL"],
    "H200": ["H200", "H200 NVL"],
    "A100": ["A100 SXM4", "A100 PCIE"],
    "B200": ["B200"],
    "RTX6000ADA": ["RTX 6000Ada"],
    "A6000": ["RTX A6000"],
    "L40S": ["L40S"],
    "L40": ["L40"],
    "4090": ["RTX 4090"],
    "5090": ["RTX 5090"],
    "V100": ["Tesla V100"],
}

#: The API caps ``limit`` server-side at 64 no matter what is asked for, so a
#: query is always a *sample* of the market rather than the whole of it.
#: Ordering by price ascending makes that sample the cheapest 64, which is the
#: only sample that answers "what is the cheapest way to run this".
MAX_OFFERS = 64

#: Below this, a host has a track record of dropping jobs. Vast reports the
#: score precisely because marketplace hosts are not interchangeable.
MIN_RELIABILITY = 0.90

#: How far above ``min_bid`` to bid when no explicit bid is given. Bidding the
#: floor exactly means being outbid by the very next person who wants the
#: machine, which turns a rental into a preemption.
BID_MARGIN = 1.15


def _norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


class VastProvider(Provider):
    name = "vast"

    def __init__(self, credentials: str = CREDENTIALS,
                 min_reliability: float = MIN_RELIABILITY,
                 image: str = "pytorch/pytorch:latest"):
        self.credentials = credentials
        self.min_reliability = min_reliability
        self.image = image

    # -- transport ---------------------------------------------------------

    def _key(self) -> str:
        key = os.environ.get("VAST_API_KEY", "")
        if key:
            return key
        p = pathlib.Path(self.credentials).expanduser()
        if p.exists():
            try:
                return json.loads(p.read_text()).get("api_key", "")
            except json.JSONDecodeError:
                return ""
        return ""

    def _get(self, path: str, params: dict | None = None, auth: bool = False) -> Any:
        url = f"{API}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {"Accept": "application/json"}
        if auth:
            key = self._key()
            if not key:
                raise LaunchFailed(
                    "vast needs an api key to rent: set VAST_API_KEY or write "
                    f"{self.credentials}", "no_credentials")
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, headers=headers)
        return self._send(req)

    def _post(self, path: str, body: Any, method: str = "PUT") -> Any:
        key = self._key()
        if not key:
            raise LaunchFailed(
                "vast needs an api key to rent: set VAST_API_KEY or write "
                f"{self.credentials}", "no_credentials")
        req = urllib.request.Request(
            f"{API}/{path}", data=json.dumps(body).encode(), method=method,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json",
                     "Accept": "application/json"})
        return self._send(req)

    @staticmethod
    def _send(req: urllib.request.Request) -> Any:
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode().strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            raise LaunchFailed(f"{e.code} {detail}",
                               VastProvider._reason(e.code, detail))
        return json.loads(raw) if raw else {}

    @staticmethod
    def _reason(code: int, detail: str) -> str:
        d = detail.lower()
        if code in (401, 403):
            return "no_credentials"
        if "no longer available" in d or "not available" in d or code == 404:
            return "no_capacity"
        if "insufficient" in d or "credit" in d or "balance" in d:
            return "no_funds"
        return f"http_{code}"

    # -- the five verbs ----------------------------------------------------

    @staticmethod
    def _gpu_filter(gpu: str) -> dict[str, Any]:
        """Vast query clause selecting every spelling of a GPU family.

        Falls back to exact match on whatever was passed, so an unmapped or
        brand-new card is still reachable by its literal marketplace name.
        """
        names = GPU_ALIASES.get(_norm(gpu).upper()) or GPU_ALIASES.get(gpu)
        return {"in": names} if names else {"eq": gpu}

    def offers(self, gpu: str | None = None, count: int = 1,
               spot: bool | None = None) -> list[Offer]:
        """Search the marketplace. Everything returned is rentable now."""
        q: dict[str, Any] = {"rentable": {"eq": True}, "rented": {"eq": False},
                             "order": [["dph_total", "asc"]],
                             "limit": MAX_OFFERS}
        if count:
            q["num_gpus"] = {"eq": count}
        if gpu:
            q["gpu_name"] = self._gpu_filter(gpu)
        raw = self._get("bundles/", {"q": json.dumps(q)}).get("offers") or []
        out: list[Offer] = []
        for o in raw:
            n = int(o.get("num_gpus") or 0)
            if not n:
                continue
            if float(o.get("reliability2") or 0) < self.min_reliability:
                continue
            name = o.get("gpu_name") or "?"
            region = (o.get("geolocation") or "?").strip(", ") or "?"
            for is_spot in ((True, False) if spot is None else (spot,)):
                # Spot is a bid: min_bid is the floor to clear, not a fixed
                # price. On-demand is dph_total, which is what you actually pay.
                price = o.get("min_bid") if is_spot else o.get("dph_total")
                if not price or float(price) <= 0:
                    continue
                out.append(Offer(sku=str(o["id"]), region=region, gpu=name,
                                 count=n, usd_hr=float(price), spot=is_spot))
        return sorted(out, key=lambda x: x.usd_per_gpu_hr)

    def available(self, sku: str, region: str, spot: bool = True) -> bool:
        """Is this specific offer still rentable?

        Offer ids are ephemeral — a machine someone else rents disappears from
        the market entirely — so this asks about the id rather than a shape.
        """
        q = {"id": {"eq": int(sku)}, "rentable": {"eq": True},
             "rented": {"eq": False}, "limit": 1}
        got = self._get("bundles/", {"q": json.dumps(q)}).get("offers") or []
        return bool(got)

    def launch(self, sku: str, region: str, *, spot: bool = True,
               name: str = "evsys", bid: float | None = None) -> Machine:
        """Rent an offer.

        For an interruptible rental the bid must clear ``min_bid``; we bid a
        margin above it by default, because bidding exactly the floor means
        being outbid by the next person who wants the machine.
        """
        # Look the offer up by id, not by scanning a search page: the listing
        # is capped at 64 rows sorted by price, so any offer outside the
        # cheapest 64 is simply absent from it.
        rows = self._get("bundles/", {"q": json.dumps(
            {"id": {"eq": int(sku)}, "limit": 1})}).get("offers") or []
        if not rows:
            raise LaunchFailed(f"offer {sku} is gone", "no_capacity")
        row = rows[0]
        price = bid
        if price is None:
            price = float(row.get("min_bid") or row.get("dph_total") or 0)
            if spot:
                price *= BID_MARGIN
        body: dict[str, Any] = {"client_id": "me", "image": self.image,
                                "disk": 100, "label": name, "runtype": "ssh"}
        if spot:
            body["price"] = price
        res = self._post(f"asks/{sku}/", body)
        if not res.get("success", True):
            raise LaunchFailed(str(res)[:200], "no_capacity")
        iid = res.get("new_contract")
        if not iid:
            raise LaunchFailed(f"unexpected launch response: {res!r}", "unknown")
        return Machine(id=str(iid), provider=self.name, sku=sku,
                       region=(row.get("geolocation") or region or "?").strip(", "),
                       gpu=row.get("gpu_name") or "?",
                       count=int(row.get("num_gpus") or 1), usd_hr=float(price))

    def poll(self, machine: Machine) -> Machine:
        rows = (self._get("instances/", auth=True) or {}).get("instances") or []
        for i in rows:
            if str(i.get("id")) == machine.id:
                machine.ip = i.get("public_ipaddr") or machine.ip
                machine.state = (RUNNING if i.get("actual_status") == "running"
                                 else PENDING)
                return machine
        machine.state = GONE
        return machine

    def terminate(self, machine: Machine) -> None:
        try:
            self._post(f"instances/{machine.id}/", {}, method="DELETE")
        except Exception as e:  # noqa: BLE001
            log.warning("[vast] could not terminate %s: %s", machine.id, e)

    def ssh_keys(self) -> list[dict]:
        """Keys registered with the account.

        Worth checking before renting: an instance launched with no registered
        key comes up with no way to log in, and is billed regardless.
        """
        got = self._get("ssh/", auth=True)
        return got.get("ssh_keys", got) if isinstance(got, dict) else got


@register_availability("vast")
class VastAvailability(AvailabilityProbe):
    """Marketplace availability. The search result *is* the answer."""

    name = "vast"

    def probe(self, gpu: str, count: int, region: str | None,
              spot: bool | None) -> list[Capacity]:
        offers = VastProvider().offers(gpu=gpu, count=count, spot=spot)
        out: list[Capacity] = []
        for o in offers:
            if region and _norm(region) not in _norm(o.region):
                continue
            out.append(Capacity(
                provider=self.name, gpu=o.gpu, count=o.count, state=AVAILABLE,
                region=o.region, sku=o.sku, usd_hr=o.usd_hr, spot=o.spot,
                detail="min bid — actual price is your bid" if o.spot
                       else "on-demand"))
        if not out:
            return [Capacity(self.name, gpu, count, UNAVAILABLE, region=region,
                             spot=bool(spot), detail="no rentable offers")]
        return out


__all__ = ["API", "BID_MARGIN", "GPU_ALIASES", "MAX_OFFERS", "MIN_RELIABILITY",
           "VastAvailability", "VastProvider"]

"""Verda (formerly DataCrunch) as a :class:`~.provider.Provider`.

Implements the five verbs against Verda's REST API. Everything here was learned
the expensive way and is written down so the next provider does not repeat it:

  * OAuth2 client-credentials with ~10 minute tokens, cached — minting one per
    call doubled every request and made a full availability sweep time out.
  * ``GET /instance-availability`` takes ``is_spot``, which **defaults to
    "false"**. Omitting it reads on-demand stock while you launch spot, and
    they disagree constantly. It also takes ``location_code`` (and a deprecated
    ``locationCode`` alias). Both are validated: a bad value is a 400.
  * That listing is **complete**, not a summary. Verified by probing all 47 GPU
    SKUs x 3 locations x both modes against the per-SKU endpoint: it omitted
    none. So enumerate from the listing (2 calls) and reserve the per-SKU
    boolean for confirming the one machine you are about to launch.
  * The dashboard's ``1x / 2x / 4x / 8x`` buttons are the shapes a GPU is
    **sold** in, not the shapes that are free. Availability is per-SKU: with
    B200 showing as available, only ``8B200.240V`` was actually purchasable.
    Reading those buttons as availability means launching a shape that does
    not exist.
  * Prices are flat per GPU across counts — 1x and 8x cost the same per
    GPU-hour — so count is a capacity decision, never a price optimisation.
  * ``POST /instances`` returns a bare instance-id string, not JSON.
  * There is **no snapshot API and no object storage**. The only image-like
    primitive is ``PUT /volumes`` with ``action: clone``, which does work
    across regions.
  * The OS volume **outlives its instance** and keeps billing at ~$0.08/hr per
    300 GB. Nothing reclaims it. Terminate must delete it explicitly, and a
    volume in ``cloning`` state accepts a delete and ignores it.
"""

from __future__ import annotations

import json
import pathlib
import time
import urllib.error
import urllib.request
from typing import Any

from ..logger import get_logger
from .provider import (GONE, PENDING, RUNNING, LaunchFailed, Machine, Offer,
                       Provider)

log = get_logger(__name__)

API = "https://api.verda.com/v1"
CREDENTIALS = "~/.verda/config.json"
#: Ubuntu 22.04 + CUDA 12.8, matching the torch cu128 wheels the stack pins.
DEFAULT_IMAGE = "aaaaaaaa-3dd9-4d09-9512-52d8032fff6e"

def _norm(s: str) -> str:
    """Fold a GPU name to something comparable.

    Verda's ``model`` field is free text with spaces and memory suffixes —
    ``"A100 80GB"``, ``"RTX PRO 6000"``, ``"Tesla V100"``. Comparing those to
    the names people actually type (``A100``, ``RTXPRO6000``) by equality
    matches nothing, which silently reported *no A100s exist* while Verda was
    selling four A100 SKUs.
    """
    return "".join(c for c in s.lower() if c.isalnum())


def _matches(model: str, want: str | None) -> bool:
    """Whether a Verda model name is the GPU the caller asked for.

    Prefix match against the whole name *and* against each word, so ``A100``
    finds ``A100 80GB``, ``A6000`` finds ``RTX A6000``, and ``RTXPRO6000``
    finds ``RTX PRO 6000``. Deliberately a prefix and not a substring: plain
    containment makes ``B300`` match ``GB300``, which is a different machine at
    a different price.
    """
    if not want:
        return True
    w = _norm(want)
    if not w:
        return True
    parts = [_norm(model)] + [_norm(p) for p in model.split()]
    return any(p.startswith(w) for p in parts if p)


class VerdaProvider(Provider):
    name = "verda"

    def __init__(self, credentials: str = CREDENTIALS,
                 image: str = DEFAULT_IMAGE, volume_gb: int = 300):
        self.credentials, self.image, self.volume_gb = credentials, image, volume_gb
        self._owned_volumes: set[str] = set()
        self._tok: str = ""
        self._tok_exp: float = 0.0
        self._cache: dict[str, tuple[float, Any]] = {}

    #: How long the catalogue and stock listing may be reused. Deliberately
    #: shorter than ``availability.DEFAULT_TTL_S`` so a cached listing can
    #: never outlive the answer built from it. Sweeping eight GPU families
    #: across four counts refetched the 47-SKU catalogue 32 times and took
    #: 132s; a scheduler polling every 60s cannot afford that.
    CACHE_TTL_S = 20.0

    def _cached(self, path: str) -> Any:
        hit = self._cache.get(path)
        if hit and time.time() - hit[0] < self.CACHE_TTL_S:
            return hit[1]
        val = self._call(path)
        self._cache[path] = (time.time(), val)
        return val

    # -- transport ---------------------------------------------------------

    def _token(self) -> str:
        """Cached bearer token.

        Tokens last ~10 minutes, so fetching one per call doubled every
        request and made an availability sweep across 47 SKUs time out. Cached
        with a safety margin, since a token that expires mid-flight fails the
        call it was minted for.
        """
        if self._tok and time.time() < self._tok_exp:
            return self._tok
        cfg = json.loads(pathlib.Path(self.credentials).expanduser().read_text())
        body = json.dumps({"grant_type": "client_credentials",
                           "client_id": cfg["client_id"],
                           "client_secret": cfg["client_secret"]}).encode()
        req = urllib.request.Request(f"{API}/oauth2/token", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        self._tok = d["access_token"]
        self._tok_exp = time.time() + max(int(d.get("expires_in", 600)) - 60, 30)
        return self._tok

    def _call(self, path: str, body: Any = None, method: str | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{API}/{path}", data=data, method=method,
            headers={"Authorization": f"Bearer {self._token()}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode().strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            raise LaunchFailed(f"{e.code} {detail}", self._reason(e.code, detail))
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # POST /instances answers with a bare id string, not JSON.
            return raw.strip('"')

    @staticmethod
    def _reason(code: int, detail: str) -> str:
        d = detail.lower()
        if "not enough resources" in d or "no capacity" in d:
            return "no_capacity"
        if "insufficient" in d or "balance" in d or "funds" in d:
            return "no_funds"
        if "not supported" in d or "decommission" in d:
            return "unsupported"
        return f"http_{code}"

    # -- the five verbs ----------------------------------------------------

    def offers(self, gpu: str | None = None, count: int = 1,
               spot: bool | None = None) -> list[Offer]:
        types = self._cached("instance-types")
        out: list[Offer] = []
        for is_spot in ((True, False) if spot is None else (spot,)):
            stock = self._stock(is_spot)
            for t in types:
                g = t.get("gpu") or {}
                if not g.get("number_of_gpus"):
                    continue
                if count and g["number_of_gpus"] != count:
                    continue
                if not _matches(t.get("model") or "", gpu):
                    continue
                price = t.get("spot_price") if is_spot else t.get("price_per_hour")
                if not price or float(price) <= 0:
                    continue
                for region in stock.get(t["instance_type"], []):
                    out.append(Offer(sku=t["instance_type"], region=region,
                                     gpu=t.get("model") or "?",
                                     count=g["number_of_gpus"],
                                     usd_hr=float(price), spot=is_spot))
        return sorted(out, key=lambda o: o.usd_per_gpu_hr)

    def _stock(self, spot: bool, region: str | None = None) -> dict[str, list[str]]:
        """SKU -> regions holding stock, for **this** purchase mode.

        ``is_spot`` defaults to ``"false"`` at the API, so omitting it asks
        about on-demand. Reading that and launching spot disagreed constantly:
        Verda had 8xH200 free on spot in FIN-03 while this reported H200 as not
        sold at all.

        This listing is the complete picture, not a summary — verified by
        probing all 47 GPU SKUs x 3 locations x both modes against the per-SKU
        endpoint and finding **zero** free SKUs the listing omitted. So one
        call per mode replaces 282, and the per-SKU endpoint is reserved for
        confirming a specific choice immediately before launching it.
        """
        q = f"instance-availability?is_spot={'true' if spot else 'false'}"
        if region:
            q += f"&location_code={region}"
        stock: dict[str, list[str]] = {}
        for loc in self._cached(q) or []:
            for it in loc.get("availabilities") or []:
                stock.setdefault(it, []).append(loc["location_code"])
        return stock

    def available(self, sku: str, region: str, spot: bool = True) -> bool:
        # The per-SKU endpoint is authoritative; the aggregate listing answers
        # about on-demand stock unless is_spot is passed, and disagrees.
        q = f"instance-availability/{sku}?is_spot={'true' if spot else 'false'}"
        return self._call(f"{q}&location_code={region}") is True

    def launch(self, sku: str, region: str, *, spot: bool = True,
               name: str = "evsys") -> Machine:
        keys = self._call("sshkeys")
        if not keys:
            raise LaunchFailed("no ssh key registered with verda", "unsupported")
        vol = f"{name}-{int(time.time())}"
        iid = self._call("instances", {
            "instance_type": sku, "image": self.image,
            "ssh_key_ids": [keys[0]["id"]], "hostname": name,
            "description": f"{name} (evsys)", "location_code": region,
            "is_spot": spot, "contract": "SPOT" if spot else "PAY_AS_YOU_GO",
            "os_volume": {"name": vol, "size": self.volume_gb}})
        if not isinstance(iid, str) or not iid:
            raise LaunchFailed(f"unexpected launch response: {iid!r}", "unknown")
        self._owned_volumes.add(vol)
        gpu, cnt, price = self._describe(sku, spot)
        return Machine(id=iid, provider=self.name, sku=sku, region=region,
                       gpu=gpu, count=cnt, usd_hr=price)

    def _describe(self, sku: str, spot: bool = True) -> tuple[str, int, float]:
        """Shape and hourly price of a SKU, for the mode actually purchased.

        This used to return ``spot_price or price_per_hour`` regardless of what
        was launched, so an on-demand machine was recorded at its spot price —
        an RTX PRO 6000 rented at $1.890/hr was logged as $0.6615. That number
        feeds the reliability ledger and every cost comparison downstream, so it
        being 2.9x low is worse than it being absent.
        """
        for t in self._cached("instance-types"):
            if t.get("instance_type") == sku:
                g = t.get("gpu") or {}
                price = t.get("spot_price") if spot else t.get("price_per_hour")
                return (t.get("model") or "?", g.get("number_of_gpus") or 1,
                        float(price or t.get("price_per_hour") or 0))
        return ("?", 1, 0.0)

    def poll(self, machine: Machine) -> Machine:
        for i in self._call("instances") or []:
            if i.get("id") == machine.id:
                machine.ip = i.get("ip") or machine.ip
                machine.state = RUNNING if i.get("status") == "running" else PENDING
                return machine
        machine.state = GONE
        return machine

    def terminate(self, machine: Machine) -> None:
        try:
            self._call("instances", {"id": machine.id, "action": "delete"},
                       method="PUT")
        except Exception as e:
            log.warning("[verda] could not terminate %s: %s", machine.id, e)
        # The disk survives the instance and keeps billing. Nothing else
        # reclaims it, so sweep here even if the terminate above failed.
        for vid, vname, _ in self.orphans():
            if vname in self._owned_volumes:
                self._delete_volume(vid, vname)

    def _delete_volume(self, vid: str, vname: str) -> None:
        try:
            self._call("volumes", {"action": "delete", "id": vid,
                                   "is_permanent": True}, method="PUT")
            self._owned_volumes.discard(vname)
        except Exception as e:
            log.warning("[verda] could not delete volume %s: %s", vname, e)

    def orphans(self) -> list[tuple[str, str, float]]:
        out = []
        for v in self._call("volumes") or []:
            # A volume mid-clone accepts a delete and silently ignores it, so
            # do not report it as reclaimable yet.
            if v.get("instance_id") or v.get("status") == "cloning":
                continue
            out.append((v["id"], v.get("name", "?"),
                        float(v.get("monthly_price") or 0)))
        return out


__all__ = ["API", "DEFAULT_IMAGE", "VerdaProvider"]

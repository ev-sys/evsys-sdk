"""Keep SkyPilot's own price catalog honest, instead of working around it.

SkyPilot plans from a CSV it downloads periodically
(``~/.sky/catalogs/<schema>/<cloud>/vms.csv``). That file is the single input
to every price comparison and every failover ordering it makes — and it is
generated ahead of time, so it drifts. It priced PrimeIntellect's H100 at
$1.97/hr when the provider was charging $3.25, and listed regions with no
stock at all.

The obvious responses are both wrong. Forking SkyPilot to add live pricing
means maintaining a fork. Building a parallel provisioning abstraction throws
away twenty clouds of battle-tested provisioning to fix a pricing problem.

The right response is to write fresh numbers into the file SkyPilot already
reads. Nothing is patched, nothing is bypassed: `sky launch`, `sky show-gpus`,
the optimizer and the failover ordering all keep working exactly as designed,
they just stop working from stale data.

    from evsys_sdk.compute import catalog
    catalog.refresh("verda")          # rewrite prices from the live API
    catalog.refresh_all()             # every provider we can price

Two properties this must have, because a catalog is load-bearing:

  * **Never lose rows.** A SKU we cannot price live keeps its catalogued price
    rather than vanishing — a missing row means SkyPilot will not consider that
    machine at all, which is worse than an out-of-date price.
  * **Never write a partial file.** Write to a temporary file and rename, so a
    crash mid-write leaves the previous catalog intact rather than a truncated
    one that breaks every subsequent launch.
"""

from __future__ import annotations

import csv
import os
import pathlib
import shutil
import tempfile
import time
from typing import Any, Callable

from ..logger import get_logger
from .pricing import PricingUnavailable, live_offers

log = get_logger(__name__)

#: SkyPilot's catalog root. The schema version is a directory level, so it is
#: discovered rather than hardcoded — pinning it would silently target the
#: wrong directory after a SkyPilot upgrade.
CATALOG_ROOT = "~/.sky/catalogs"

PRICE_COL = "Price"
SPOT_COL = "SpotPrice"
GPU_COL = "AcceleratorName"
COUNT_COL = "AcceleratorCount"
REGION_COL = "Region"


def catalog_dir() -> pathlib.Path | None:
    """Newest schema directory under the catalog root, or None."""
    root = pathlib.Path(CATALOG_ROOT).expanduser()
    if not root.is_dir():
        return None
    versions = sorted((d for d in root.iterdir() if d.is_dir()),
                      key=lambda d: d.stat().st_mtime, reverse=True)
    return versions[0] if versions else None


def catalog_path(cloud: str) -> pathlib.Path | None:
    d = catalog_dir()
    if d is None:
        return None
    p = d / cloud / "vms.csv"
    return p if p.exists() else None


def _price_index(cloud: str, gpus: set[str],
                 counts: set[int]) -> dict[tuple[str, int, str], tuple[float | None, float | None]]:
    """Live (on-demand, spot) price by (gpu, count, region).

    Queries only the (gpu, count) pairs the catalog actually contains, so a
    provider with fifty SKUs is not swept exhaustively to refresh sixteen rows.
    """
    out: dict[tuple[str, int, str], tuple[float | None, float | None]] = {}
    for gpu in sorted(gpus):
        for count in sorted(counts):
            for spot in (True, False):
                try:
                    offers = live_offers(cloud, gpu, count, spot=spot)
                except PricingUnavailable as e:
                    log.debug("[catalog] %s %s x%d spot=%s: %s",
                              cloud, gpu, count, spot, e)
                    continue
                except Exception as e:  # noqa: BLE001
                    log.info("[catalog] %s unreachable: %s", cloud, e)
                    return out
                for o in offers:
                    key = (gpu, count, o.region)
                    od, sp = out.get(key, (None, None))
                    if o.spot:
                        sp = o.usd_hr if sp is None else min(sp, o.usd_hr)
                    else:
                        od = o.usd_hr if od is None else min(od, o.usd_hr)
                    out[key] = (od, sp)
    return out


def refresh(cloud: str, *, path: pathlib.Path | None = None,
            index: Callable[..., Any] | None = None) -> dict[str, int]:
    """Rewrite ``cloud``'s catalog prices from its live API.

    Returns counts: rows seen, prices updated, rows left untouched. Raises
    nothing — a provider that cannot be reached leaves the catalog as it was,
    which is the correct failure: stale prices beat no catalog.
    """
    p = path or catalog_path(cloud)
    if p is None:
        log.info("[catalog] no local catalog for %s — nothing to refresh", cloud)
        return {"rows": 0, "updated": 0, "unchanged": 0}

    with p.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"rows": 0, "updated": 0, "unchanged": 0}
    fields = list(rows[0].keys())
    if PRICE_COL not in fields and SPOT_COL not in fields:
        log.warning("[catalog] %s has no price columns — refusing to touch it", p)
        return {"rows": len(rows), "updated": 0, "unchanged": len(rows)}

    gpus = {r[GPU_COL] for r in rows if r.get(GPU_COL)}
    counts = set()
    for r in rows:
        try:
            counts.add(int(float(r.get(COUNT_COL) or 0)))
        except ValueError:
            continue
    counts.discard(0)

    lookup = (index or _price_index)(cloud, gpus, counts)
    if not lookup:
        log.info("[catalog] no live prices for %s — catalog left as-is", cloud)
        return {"rows": len(rows), "updated": 0, "unchanged": len(rows)}

    updated = 0
    for r in rows:
        try:
            key = (r.get(GPU_COL), int(float(r.get(COUNT_COL) or 0)),
                   r.get(REGION_COL))
        except ValueError:
            continue
        if key not in lookup:
            # Unpriced SKUs keep their catalogued price. Dropping the row would
            # hide the machine from SkyPilot entirely, which is worse than an
            # out-of-date number.
            continue
        od, sp = lookup[key]
        before = (r.get(PRICE_COL), r.get(SPOT_COL))
        if od is not None and PRICE_COL in r:
            r[PRICE_COL] = f"{od:g}"
        if sp is not None and SPOT_COL in r:
            r[SPOT_COL] = f"{sp:g}"
        if (r.get(PRICE_COL), r.get(SPOT_COL)) != before:
            updated += 1

    # Atomic replace: a crash mid-write must not leave a truncated catalog,
    # because every later launch reads this file.
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        shutil.move(tmp, p)
    except Exception:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise

    log.info("[catalog] %s: %d/%d prices refreshed from the live API",
             cloud, updated, len(rows))
    return {"rows": len(rows), "updated": updated,
            "unchanged": len(rows) - updated}


def refresh_all(clouds: list[str] | None = None) -> dict[str, dict[str, int]]:
    """Refresh every cloud we have a live pricing probe for."""
    from .pricing import PROBES
    names = clouds if clouds is not None else sorted(PROBES)
    return {c: refresh(c) for c in names}


#: How stale a catalog may be before a launch bothers to refresh it. Prices
#: move on the order of hours; what moves minute-to-minute is *availability*,
#: which this file does not carry — so a per-minute refresh would be thousands
#: of API calls a day re-deriving numbers that did not change. Refresh when the
#: number is about to be used, and put a floor under how old it can be.
DEFAULT_MAX_AGE_S = 900.0


def refresh_if_stale(cloud: str, max_age_s: float = DEFAULT_MAX_AGE_S,
                     **kw: Any) -> dict[str, int] | None:
    """Refresh only if the catalog is older than ``max_age_s``.

    Cheap enough to call before every launch: a fresh catalog costs one
    ``stat``. Returns None when no refresh was needed.
    """
    age = age_s(cloud)
    if age is not None and age < max_age_s:
        log.debug("[catalog] %s is %.0fs old — fresh enough", cloud, age)
        return None
    return refresh(cloud, **kw)


def age_s(cloud: str) -> float | None:
    """Seconds since this catalog was last written, or None if absent."""
    p = catalog_path(cloud)
    return (time.time() - p.stat().st_mtime) if p else None


__all__ = ["CATALOG_ROOT", "DEFAULT_MAX_AGE_S", "age_s", "catalog_dir",
           "catalog_path", "refresh", "refresh_all", "refresh_if_stale"]

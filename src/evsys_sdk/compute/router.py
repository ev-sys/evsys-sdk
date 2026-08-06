"""What GPUs can I rent right now, and what do they cost?

A thin poller over the provider probes in :mod:`evsys_sdk.compute.pricing`.
It answers one question — which accelerators are actually purchasable at this
moment, and at what price per GPU-hour — across every provider we hold
credentials for, cheapest first.

Deliberately no throughput or cost-per-token modelling. Tokens/sec depends on
the model, sequence length, batch size, LoRA rank, whether the job is training
or RL, and a handful of server flags; a table baked in here is wrong the moment
any of them changes. Benchmark numbers belong with the benchmark. This returns
prices and availability: facts with a short shelf life and no caveats.

    from evsys_sdk.compute import router

    print(router.report(gpu="H200"))
    best = router.cheapest(min_memory_gib=96)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..logger import get_logger
from .credentials import PROVIDERS
from .pricing import Offer, PricingUnavailable, live_offers

log = get_logger(__name__)

#: Device memory per GPU, GiB. Used only to filter on "will my model fit",
#: which is a property of the hardware rather than of any benchmark.
GPU_MEMORY_GIB = {
    "A100": 40.0, "A100-80GB": 80.0, "H100": 80.0, "H200": 141.0,
    "B200": 180.0, "B300": 268.0, "RTXPRO6000": 96.0, "L40S": 48.0,
    "A6000": 48.0, "A40": 48.0, "V100": 16.0, "RTX6000Ada": 48.0,
}

DEFAULT_GPUS = ("H200", "H100", "A100-80GB", "RTXPRO6000", "B200")


@dataclass(frozen=True)
class Quote:
    """One purchasable option, priced per GPU so sizes compare honestly."""

    provider: str
    gpu: str
    count: int
    region: str
    usd_hr: float
    usd_per_gpu_hr: float
    spot: bool
    available: bool
    memory_gib: float | None

    def describe(self) -> str:
        mem = f", {self.memory_gib:.0f} GiB/gpu" if self.memory_gib else ""
        return (f"${self.usd_per_gpu_hr:.4f}/gpu-hr  {self.gpu}x{self.count} "
                f"({'spot' if self.spot else 'on-demand'}) "
                f"{self.provider}/{self.region} "
                f"[${self.usd_hr:.3f}/hr total{mem}]"
                f"{'' if self.available else '  OUT OF STOCK'}")


def _quote(o: Offer) -> Quote:
    return Quote(provider=o.provider, gpu=o.gpu, count=o.count, region=o.region,
                 usd_hr=o.usd_hr, usd_per_gpu_hr=o.usd_hr / max(o.count, 1),
                 spot=o.spot, available=o.available,
                 memory_gib=GPU_MEMORY_GIB.get(o.gpu))


def offers(*, gpu: str | None = None, gpus: tuple[str, ...] = DEFAULT_GPUS,
           counts: tuple[int, ...] = (1, 2, 4, 8),
           spot: bool | None = None, providers: list[str] | None = None,
           available_only: bool = True,
           min_memory_gib: float | None = None) -> list[Quote]:
    """Live quotes across authenticated providers, cheapest per GPU first.

    Priced per GPU because that is the only way an 8-GPU box compares fairly
    with a single card. A provider that cannot be reached is skipped with a log
    line rather than failing the call: a partial answer beats none when you are
    hunting for capacity.
    """
    want = (gpu,) if gpu else gpus
    names = providers or [p.name for p in PROVIDERS.values() if p.authenticated()]
    if not names:
        log.warning("[router] no authenticated providers — see "
                    "evsys_sdk.compute.credentials.report()")
        return []

    out: list[Quote] = []
    for prov in names:
        for g in want:
            for n in counts:
                try:
                    got = live_offers(prov, g, n, spot=spot)
                except PricingUnavailable as e:
                    log.debug("[router] %s/%s x%d: %s", prov, g, n, e)
                    continue
                except Exception as e:
                    log.info("[router] %s unreachable: %s", prov, e)
                    break
                out.extend(_quote(o) for o in got)

    if available_only:
        out = [q for q in out if q.available]
    if min_memory_gib is not None:
        # Unknown memory is kept rather than filtered out: dropping it would
        # hide real capacity just because the lookup table is incomplete.
        out = [q for q in out
               if q.memory_gib is None or q.memory_gib >= min_memory_gib]
    return sorted(out, key=lambda q: q.usd_per_gpu_hr)


def cheapest(**kw) -> Quote | None:
    """Cheapest available option per GPU-hour, or None if nothing is free."""
    got = offers(**kw)
    if not got:
        log.info("[router] nothing available matching %s", kw or "the defaults")
    return got[0] if got else None


def report(**kw) -> str:
    got = offers(**kw)
    if not got:
        return "no capacity available matching that request"
    return "\n".join(q.describe() for q in got)


__all__ = ["DEFAULT_GPUS", "GPU_MEMORY_GIB", "Quote", "cheapest", "offers",
           "report"]

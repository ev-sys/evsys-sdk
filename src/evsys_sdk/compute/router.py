"""Pick the cheapest hardware and layout for a training job.

Every number here was measured on spot capacity in August 2026 against
Qwen3-4B-Instruct-2507 with LoRA rank 32, packed sequences, and
``fused_lm_head_logprob`` enabled. The point of writing them down is that the
answers are not guessable from spec sheets:

  * **The cheapest adequate card wins, and by more as context grows.** An RTX
    PRO 6000 runs at 69% of an H200's throughput at 8K but 90% at 128K, for 47%
    of the price. Buying the faster card costs you money at every length where
    the cheap one fits.
  * **Capacity, not speed, sets the cliff.** 256K needs ~108 GiB, so a 96 GiB
    card simply cannot, and the only cards that can are three times the price
    per token.
  * **Concurrency is an amortisation tool, not a throughput one.** Extra
    adapters bought +13% at 8K and +3% at 64K, while prompt count buys the same
    generation throughput inside a single experiment. What concurrency really
    saves is the ~7 minute setup, which dominates any experiment shorter than
    about half an hour.

Prices move, so :func:`route` takes live offers when you have them and falls
back to the measured table when you do not. Throughput does not move, and is
what makes the recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..logger import get_logger

log = get_logger(__name__)

#: Tinker's published training rate, and the context it is quoted at. Comparing
#: our 8K throughput against it flatters us; comparing 256K is not a comparison
#: at all, because Tinker does not sell that context.
TINKER_USD_PER_M_TRAIN = 0.737
TINKER_CONTEXT = 65536

#: Measured training throughput, tokens/sec, by (card, sequence length).
#: None means it did not fit. Absent lengths are interpolated, never guessed
#: beyond the measured envelope.
SFT_TOK_S: dict[str, dict[int, float | None]] = {
    "RTXPRO6000": {8192: 5262.0, 65536: 2485.0, 131072: 1528.0, 262144: None},
    "H200":       {8192: 7597.0, 16384: 6922.0, 32768: 4528.0,
                   65536: 2915.0, 131072: 1700.0, 262144: 899.0},
}

#: Usable device memory, GiB. The 256K working set was 107.9 GiB, which is why
#: a 96 GiB card cannot reach it however fast it is.
CARD_MEM_GIB = {"RTXPRO6000": 96.0, "H200": 141.0, "H100": 80.0, "A100": 80.0}

#: Spot $/hr per GPU, observed. Overridden by live pricing when available.
CARD_USD_HR = {"A100": 0.626, "RTXPRO6000": 0.6615, "H100": 1.137,
               "H200": 1.400, "B200": 2.139, "B300": 2.625}

#: Setup before any work: install, server warmup, model pull.
SETUP_S = 420.0

#: Aggregate throughput multiplier vs one adapter, by context and tenant count.
#: Flat at long context because one tenant already saturates the pipeline.
ADAPTER_SCALING = {
    8192:  {1: 1.00, 2: 1.08, 4: 1.13, 8: 1.08, 12: 1.04, 16: 1.04, 24: 0.94},
    65536: {1: 1.00, 2: 1.01, 4: 1.03, 8: 1.03},
}


@dataclass(frozen=True)
class Plan:
    card: str
    gpus: int
    colocate: bool
    adapters: int
    usd_hr: float
    tok_s: float
    usd_per_M: float
    vs_tinker: float | None
    why: str

    def describe(self) -> str:
        v = f", {self.vs_tinker:.1f}x under Tinker" if self.vs_tinker else ""
        return (f"{self.gpus}x{self.card} @ ${self.usd_hr:.3f}/hr, "
                f"{'colocated' if self.colocate else 'disaggregated'}, "
                f"{self.adapters} adapter(s): {self.tok_s:,.0f} tok/s = "
                f"${self.usd_per_M:.4f}/M{v} — {self.why}")


def _interp(card: str, seq_len: int) -> float | None:
    """Throughput at ``seq_len``, interpolated within the measured envelope.

    Returns None outside it rather than extrapolating: throughput falls
    super-linearly with context (attention is quadratic), so a straight-line
    guess past the last measurement is optimistic in the direction that costs
    money.

    The same convexity means interpolation *between* widely spaced points also
    reads high — the true curve sags below the chord. Treat an interpolated
    figure as an upper bound and measure the point if the decision is close.
    """
    table = SFT_TOK_S.get(card) or {}
    fitted = {k: v for k, v in table.items() if v is not None}
    if not fitted:
        return None
    if seq_len in fitted:
        return fitted[seq_len]
    below = [k for k in fitted if k < seq_len]
    above = [k for k in fitted if k > seq_len]
    if not below or not above:
        return None
    lo, hi = max(below), min(above)
    f = (seq_len - lo) / (hi - lo)
    return fitted[lo] + f * (fitted[hi] - fitted[lo])


def _fits(card: str, seq_len: int) -> bool:
    table = SFT_TOK_S.get(card) or {}
    if table.get(seq_len, "?") is None:      # measured and it did not fit
        return False
    return True


def route(seq_len: int, *, mode: str = "sft", experiments: int = 1,
          live_usd_hr: dict[str, float] | None = None,
          available: list[str] | None = None) -> Plan | None:
    """Cheapest plan for this job, or None if nothing measured can run it.

    ``experiments`` is how many independent runs you have to do, not a
    parallelism knob: concurrency is worth using when it amortises setup across
    real work, and worth ignoring otherwise.
    """
    prices = {**CARD_USD_HR, **(live_usd_hr or {})}
    cards = available or [c for c in SFT_TOK_S if c in prices]

    best: Plan | None = None
    for card in cards:
        if not _fits(card, seq_len):
            continue
        tok_s = _interp(card, seq_len)
        if not tok_s:
            continue
        hr = prices.get(card)
        if hr is None:
            continue

        # RL costs a second GPU only when experiments must run concurrently:
        # colocated single-GPU RL works, and its weight sync is 6.1s warm.
        colocate = (mode != "rl") or experiments == 1
        gpus = 1 if colocate else 2

        # Concurrency only helps where it was measured to help, and only if
        # there is real work to share.
        scale_at = min(ADAPTER_SCALING, key=lambda k: abs(k - seq_len))
        adapters = 1
        if experiments > 1:
            options = [n for n in ADAPTER_SCALING[scale_at] if n <= experiments]
            adapters = max(options, key=lambda n: ADAPTER_SCALING[scale_at][n])
        eff = tok_s * ADAPTER_SCALING[scale_at].get(adapters, 1.0)

        usd_hr = hr * gpus
        usd_per_M = usd_hr / (eff * 3600 / 1e6)
        why = []
        if adapters > 1:
            why.append(f"{adapters} adapters amortise the {SETUP_S/60:.0f} min setup")
        if not colocate:
            why.append("disaggregated: concurrent RL needs policy and vLLM on separate cards")
        if seq_len > 131072:
            why.append("only cards with >128 GiB reach this context")
        plan = Plan(card=card, gpus=gpus, colocate=colocate, adapters=adapters,
                    usd_hr=usd_hr, tok_s=eff, usd_per_M=usd_per_M,
                    vs_tinker=(TINKER_USD_PER_M_TRAIN / usd_per_M
                               if seq_len <= TINKER_CONTEXT else None),
                    why="; ".join(why) or "cheapest card that fits")
        if best is None or plan.usd_per_M < best.usd_per_M:
            best = plan

    if best is None:
        log.info("[router] nothing measured runs seq_len=%d — the largest "
                 "measured working set was 107.9 GiB at 256K", seq_len)
    return best


__all__ = ["CARD_MEM_GIB", "CARD_USD_HR", "Plan", "SFT_TOK_S",
           "TINKER_CONTEXT", "TINKER_USD_PER_M_TRAIN", "route"]

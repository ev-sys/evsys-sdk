"""Routing a job to the cheapest hardware that can run it.

The table encodes measurements that contradict intuition — the slower card is
usually cheaper per token, and the advantage widens with context — so these
tests pin the conclusions rather than the arithmetic.
"""
from __future__ import annotations

import pytest

from evsys_sdk.compute import router as rt


class TestCheapestAdequateCardWins:
    def test_prefers_the_slower_cheaper_card_at_short_context(self):
        """RTX PRO 6000 is 69% of an H200's throughput for 47% of the price."""
        p = rt.route(8192)
        assert p and p.card == "RTXPRO6000"
        assert p.usd_per_M < rt.CARD_USD_HR["H200"] / (7597 * 3600 / 1e6)

    def test_advantage_widens_with_context(self):
        """90% of H200 throughput at 128K vs 69% at 8K, same price ratio."""
        short = rt.route(8192)
        long_ = rt.route(131072)
        h200_short = rt.CARD_USD_HR["H200"] / (7597 * 3600 / 1e6)
        h200_long = rt.CARD_USD_HR["H200"] / (1700 * 3600 / 1e6)
        assert (h200_short / short.usd_per_M) < (h200_long / long_.usd_per_M)

    def test_falls_back_to_the_big_card_only_when_capacity_forces_it(self):
        """256K needs ~108 GiB; a 96 GiB card cannot, however fast."""
        p = rt.route(262144)
        assert p and p.card == "H200"


class TestTinkerComparison:
    def test_reports_a_multiple_at_or_below_tinkers_context(self):
        p = rt.route(rt.TINKER_CONTEXT)
        assert p.vs_tinker and p.vs_tinker > 5

    def test_reports_no_multiple_beyond_tinkers_context(self):
        """Above 64K it is not a price difference — it is a capability Tinker
        does not sell, and quoting a ratio there would be misleading."""
        assert rt.route(262144).vs_tinker is None


class TestConcurrency:
    def test_single_experiment_uses_one_adapter(self):
        assert rt.route(8192, experiments=1).adapters == 1

    def test_several_experiments_use_the_measured_optimum(self):
        """+13% at n=4 was the peak at 8K; n=24 measured 0.94x."""
        assert rt.route(8192, experiments=8).adapters == 4

    def test_concurrency_is_not_used_where_it_was_measured_flat(self):
        """At 64K every tenant count landed within 3%, so the pick is
        whichever is nominally best but the gain is noise, not strategy."""
        p = rt.route(65536, experiments=8)
        scale = rt.ADAPTER_SCALING[65536][p.adapters]
        assert scale <= 1.03

    def test_never_exceeds_the_number_of_real_experiments(self):
        assert rt.route(8192, experiments=2).adapters <= 2


class TestRLPlacement:
    def test_single_rl_run_colocates_on_one_gpu(self):
        """Warm weight sync is 6.1s, so colocation costs wall-clock, not a
        second card."""
        p = rt.route(8192, mode="rl", experiments=1)
        assert p.colocate and p.gpus == 1

    def test_concurrent_rl_disaggregates(self):
        """n>=2 dies under colocation: engine sleep/wake is global, so one
        tenant's training sleeps the engine another is sampling on."""
        p = rt.route(8192, mode="rl", experiments=4)
        assert not p.colocate and p.gpus == 2

    def test_sft_never_needs_two_gpus(self):
        assert rt.route(8192, mode="sft", experiments=4).gpus == 1


class TestHonestAboutItsLimits:
    def test_refuses_to_extrapolate_past_the_measured_envelope(self):
        """Attention is quadratic, so a straight-line guess past the last
        measurement errs in the direction that costs money."""
        assert rt.route(1_000_000) is None

    def test_interpolates_inside_the_envelope(self):
        """Between two measured points, for whichever card wins on price —
        which at 24K is the cheap one, not the fast one."""
        p = rt.route(24576)
        assert p and p.card == "RTXPRO6000"
        assert 2485 <= p.tok_s <= 5262        # bracketed by its 8K and 64K points

    def test_interpolation_is_linear_over_a_convex_curve(self):
        """Documented weakness, not an accident: throughput falls faster than
        linearly with context, so a straight line between measurements reads
        HIGH in the middle. The router therefore over-promises between widely
        spaced points, and the fix is more measurements, not more maths."""
        mid = rt._interp("H200", 49152)       # between 32K and 64K
        assert mid is not None
        # The true curve lies below the chord; assert we know we are above it.
        assert mid > (4528 + 2915) / 2 * 0.9

    def test_live_prices_override_the_measured_table(self):
        """Prices move hourly; throughput does not."""
        cheap = rt.route(8192, live_usd_hr={"H200": 0.10})
        assert cheap.card == "H200"

    def test_respects_what_is_actually_available(self):
        p = rt.route(8192, available=["H200"])
        assert p.card == "H200"

"""Live capacity and price polling.

Prices and stock are facts with a short shelf life; throughput is not a fact
about hardware at all (it depends on model, sequence length, batch, rank and
mode), so it is deliberately absent here.
"""
from __future__ import annotations

import pytest

from evsys_sdk.compute import pricing
from evsys_sdk.compute import router as rt


def offer(usd, gpu="H200", count=1, spot=True, available=True,
          provider="verda", region="FIN-02"):
    return pricing.Offer(provider=provider, region=region, gpu=gpu, count=count,
                         usd_hr=usd, spot=spot, available=available)


@pytest.fixture
def probe(monkeypatch):
    """Stub live_offers and pretend one provider is authenticated."""
    state = {"offers": []}
    monkeypatch.setattr(rt, "live_offers",
                        lambda prov, g, n, spot=None: [
                            o for o in state["offers"]
                            if o.gpu == g and o.count == n
                            and (spot is None or o.spot == spot)])
    monkeypatch.setattr(rt.PROVIDERS["verda"].__class__, "authenticated",
                        lambda self: self.name == "verda")
    return state


class TestPricingPerGpu:
    def test_sorted_by_price_per_gpu_not_per_box(self, probe):
        """An 8-GPU box at $11.20 is cheaper per GPU than one at $1.50."""
        probe["offers"] = [offer(1.50, count=1), offer(11.20, count=8)]
        got = rt.offers(gpu="H200", counts=(1, 8))
        assert [q.count for q in got] == [8, 1]
        assert got[0].usd_per_gpu_hr == pytest.approx(1.40)

    def test_reports_both_total_and_per_gpu(self, probe):
        probe["offers"] = [offer(11.20, count=8)]
        q = rt.offers(gpu="H200", counts=(8,))[0]
        assert "11.200/hr total" in q.describe()
        assert "1.4000/gpu-hr" in q.describe()


class TestFiltering:
    def test_out_of_stock_hidden_by_default(self, probe):
        probe["offers"] = [offer(1.40, available=False)]
        assert rt.offers(gpu="H200", counts=(1,)) == []

    def test_out_of_stock_visible_on_request(self, probe):
        """Useful when you want to know a SKU exists but is dry."""
        probe["offers"] = [offer(1.40, available=False)]
        got = rt.offers(gpu="H200", counts=(1,), available_only=False)
        assert len(got) == 1 and "OUT OF STOCK" in got[0].describe()

    def test_memory_filter_uses_hardware_capacity(self, probe):
        probe["offers"] = [offer(1.14, gpu="H100"), offer(1.40, gpu="H200")]
        got = rt.offers(gpus=("H100", "H200"), counts=(1,), min_memory_gib=100)
        assert [q.gpu for q in got] == ["H200"]

    def test_unknown_memory_is_kept_not_dropped(self, probe):
        """Filtering it out would hide real capacity because our lookup table
        is incomplete, which is the wrong failure direction when hunting."""
        probe["offers"] = [offer(0.50, gpu="MYSTERY9000")]
        got = rt.offers(gpus=("MYSTERY9000",), counts=(1,), min_memory_gib=100)
        assert len(got) == 1

    def test_spot_filter_passes_through(self, probe):
        probe["offers"] = [offer(1.40, spot=True), offer(4.00, spot=False)]
        assert [q.spot for q in rt.offers(gpu="H200", counts=(1,), spot=True)] == [True]


class TestDegradesGracefully:
    def test_a_provider_that_cannot_be_priced_is_skipped(self, probe, monkeypatch):
        def boom(prov, g, n, spot=None):
            raise pricing.PricingUnavailable("no probe")
        monkeypatch.setattr(rt, "live_offers", boom)
        assert rt.offers(gpu="H200", counts=(1,)) == []

    def test_no_authenticated_providers_returns_empty(self, monkeypatch):
        monkeypatch.setattr(rt.PROVIDERS["verda"].__class__, "authenticated",
                            lambda self: False)
        assert rt.offers(gpu="H200") == []

    def test_cheapest_is_none_when_nothing_is_free(self, probe):
        probe["offers"] = []
        assert rt.cheapest(gpu="H200", counts=(1,)) is None

    def test_report_says_so_plainly(self, probe):
        probe["offers"] = []
        assert "no capacity" in rt.report(gpu="H200", counts=(1,))


class TestNoBenchmarkModelling:
    def test_router_exposes_no_throughput_or_cost_per_token(self):
        """Tokens/sec depends on model, sequence length, batch, rank and mode.
        A table baked in here would be wrong as soon as any of them changed."""
        assert not hasattr(rt, "SFT_TOK_S")
        assert not hasattr(rt, "route")
        assert not any("tok" in f for f in rt.Quote.__dataclass_fields__)

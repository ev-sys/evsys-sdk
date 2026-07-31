"""Live pricing probes.

The catalog SkyPilot plans from is a pre-generated CSV. It was wrong about
PrimeIntellect H100 by 65% and routed to a region with no stock at all, so
these tests pin the two things that made those failures silent: reading the
right price field, and telling "could not ask" apart from "there is none".
"""
from __future__ import annotations

import json
import pytest

from evsys_sdk.compute import pricing


def _payload(*offers):
    return {"H100_80GB": list(offers)}


def _offer(price=3.25, count=1, cloud_id="1H100.30V", stock="Available",
           provider="datacrunch", dc="FIN-02"):
    return {"gpuCount": count, "prices": {"onDemand": price}, "cloudId": cloud_id,
            "stockStatus": stock, "provider": provider, "dataCenter": dc}


@pytest.fixture
def prime(monkeypatch):
    """Stub the HTTP layer and the key file, keeping the parsing real."""
    state = {"payload": _payload(_offer())}

    class _Resp:
        def __init__(self, data): self._d = data
        def read(self): return json.dumps(self._d).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(pricing, "_prime_key", lambda: "pit_test")
    monkeypatch.setattr(pricing.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(state["payload"]))
    return state


class TestPrimeIntellectParsing:
    def test_reads_the_price_field_that_exists(self, prime):
        """`prices.price` is absent from this payload; reading it yields None
        for every offer, which reports as 'no capacity anywhere'."""
        offers = pricing.live_offers("primeintellect", "H100")
        assert [o.usd_hr for o in offers] == [3.25]

    def test_spot_is_a_separate_sku_not_a_flag(self, prime):
        prime["payload"] = _payload(_offer(3.25, cloud_id="1H100.30V"),
                                    _offer(0.63, cloud_id="1H100.30V_SPOT"))
        spot = pricing.live_offers("primeintellect", "H100", spot=True)
        assert [o.usd_hr for o in spot] == [0.63]
        assert all(o.spot for o in spot)
        assert [o.usd_hr for o in pricing.live_offers("primeintellect", "H100",
                                                      spot=False)] == [3.25]

    def test_cheapest_first(self, prime):
        prime["payload"] = _payload(_offer(4.5), _offer(3.25), _offer(4.0))
        assert [o.usd_hr for o in pricing.live_offers("primeintellect", "H100")] == \
            [3.25, 4.0, 4.5]

    def test_out_of_stock_is_reported_not_dropped(self, prime):
        """Knowing an offer exists but is unavailable is the whole point —
        that is the case the catalog cannot express."""
        prime["payload"] = _payload(_offer(1.9, stock="Unavailable"))
        offers = pricing.live_offers("primeintellect", "H100")
        assert len(offers) == 1 and offers[0].available is False
        assert pricing.cheapest_available("primeintellect", "H100") is None

    def test_gpu_count_is_respected(self, prime):
        prime["payload"] = _payload(_offer(3.25, count=1), _offer(26.0, count=8))
        assert [o.usd_hr for o in pricing.live_offers("primeintellect", "H100", 1)] == [3.25]
        assert [o.usd_hr for o in pricing.live_offers("primeintellect", "H100", 8)] == [26.0]

    def test_a100_without_a_memory_suffix_is_a_different_machine(self):
        """A bare A100 means the 40 GB part on their side — a different box at
        a different price, which is not what `A100-80GB:1` asked for."""
        assert pricing.PRIME_GPU_NAMES["A100"] == "A100_40GB"
        assert pricing.PRIME_GPU_NAMES["A100-80GB"] == "A100_80GB"


class TestDegradesQuietly:
    def test_unknown_cloud_returns_nothing_rather_than_raising(self):
        assert pricing.live_offers("aws", "H100") == []

    def test_unknown_gpu_is_could_not_ask_not_none_available(self, prime):
        with pytest.raises(pricing.PricingUnavailable):
            pricing.live_offers("primeintellect", "TPUv5")

    def test_a_failed_query_is_could_not_ask(self, monkeypatch):
        monkeypatch.setattr(pricing, "_prime_key", lambda: "pit_test")

        def boom(req, timeout=None):
            raise OSError("connection reset")

        monkeypatch.setattr(pricing.urllib.request, "urlopen", boom)
        with pytest.raises(pricing.PricingUnavailable):
            pricing.live_offers("primeintellect", "H100")

    def test_missing_credentials_is_could_not_ask(self, monkeypatch):
        monkeypatch.setattr(pricing, "PRIME_CREDENTIALS", "/nonexistent/prime.json")
        with pytest.raises(pricing.PricingUnavailable):
            pricing.live_offers("primeintellect", "H100")

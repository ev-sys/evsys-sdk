"""Availability probes: the three states, the ordering, and the waiting.

The bugs these guard against are all ones we actually hit: `unknown` being
read as `no`, a `while` loop that never exits because `if deadline` treats 0
as absent, and one flaky vendor taking down a multi-vendor scan.
"""

from __future__ import annotations

import time

import pytest

from evsys_sdk.compute import availability as av
from evsys_sdk.compute import pricing
from evsys_sdk.registry import get_availability, list_availabilities


class FakeProbe(av.AvailabilityProbe):
    name = "fake"

    def __init__(self, result=None, boom=None):
        self._result, self._boom = result, boom
        self.calls = 0

    def probe(self, gpu, count, region, spot):
        self.calls += 1
        if self._boom:
            raise self._boom
        out = self._result() if callable(self._result) else self._result
        return list(out or [])


def cap(provider="fake", state=av.AVAILABLE, usd=1.0, region="a", count=1):
    return av.Capacity(provider=provider, gpu="H100", count=count, state=state,
                       region=region, usd_hr=usd)


# -- the three states --------------------------------------------------------


def test_probe_failure_is_unknown_not_unavailable():
    """The distinction the whole module exists for: a probe we could not run
    must never look like a vendor saying no."""
    got = FakeProbe(boom=RuntimeError("timeout")).check("H100")
    assert [c.state for c in got] == [av.UNKNOWN]
    assert not got[0].ok


def test_check_never_raises():
    assert FakeProbe(boom=ValueError("bad json")).check("H100")[0].state == av.UNKNOWN


def test_empty_result_is_an_explicit_unavailable():
    """Empty is ambiguous at the call site, so it is resolved here."""
    got = FakeProbe(result=[]).check("H100")
    assert [c.state for c in got] == [av.UNAVAILABLE]
    assert "no matching offers" in got[0].detail


def test_available_passes_through():
    assert FakeProbe(result=[cap()]).check("H100")[0].ok


# -- staleness ---------------------------------------------------------------


def test_answers_expire():
    """Capacity is a hint with a timestamp — SKUs that probed free were refused
    ~30s later all afternoon."""
    stale = av.Capacity("v", "H100", 1, av.AVAILABLE,
                        checked_at=time.time() - 600)
    assert stale.ok and not stale.fresh()
    assert av.Capacity("v", "H100", 1, av.AVAILABLE).fresh()


def test_ttl_is_seconds_not_minutes():
    assert 10 <= av.DEFAULT_TTL_S <= 120


def test_usd_per_gpu_hr_divides_by_count():
    assert av.Capacity("v", "H100", 8, av.AVAILABLE, usd_hr=16.0).usd_per_gpu_hr == 2.0
    assert av.Capacity("v", "H100", 8, av.AVAILABLE).usd_per_gpu_hr is None


# -- ordering ----------------------------------------------------------------


def test_scan_ranks_available_then_unknown_then_unavailable(monkeypatch):
    """Unknown is worth an attempt; a refusal is not. Same principle as
    _ordered_candidates: we would rather try than refuse."""
    table = {
        "no": [cap(provider="no", state=av.UNAVAILABLE, usd=0.1)],
        "maybe": [cap(provider="maybe", state=av.UNKNOWN, usd=9.0)],
        "yes": [cap(provider="yes", state=av.AVAILABLE, usd=5.0)],
    }
    monkeypatch.setattr(av, "clouds", lambda: list(table))
    monkeypatch.setattr(av, "check",
                        lambda c, *a, **k: table[c], raising=True)
    assert [c.provider for c in av.scan("H100")] == ["yes", "maybe", "no"]


def test_scan_orders_by_price_per_gpu_within_a_state(monkeypatch):
    table = {
        "pricey": [cap(provider="pricey", usd=8.0, count=1)],
        "bulk": [cap(provider="bulk", usd=16.0, count=8)],  # $2/gpu-hr
    }
    monkeypatch.setattr(av, "clouds", lambda: list(table))
    monkeypatch.setattr(av, "check", lambda c, *a, **k: table[c])
    assert [c.provider for c in av.scan("H100")] == ["bulk", "pricey"]


def test_one_broken_vendor_does_not_break_the_scan(monkeypatch):
    def flaky(cloud, *a, **k):
        if cloud == "down":
            p = FakeProbe(boom=OSError("connection refused"))
            p.name = cloud
            return p.check("H100")
        return [cap(provider=cloud)]

    monkeypatch.setattr(av, "clouds", lambda: ["down", "up"])
    monkeypatch.setattr(av, "check", flaky)
    got = av.scan("H100")
    assert got[0].provider == "up" and got[0].ok
    assert {c.provider for c in got} == {"up", "down"}


def test_available_filters_to_purchasable(monkeypatch):
    monkeypatch.setattr(av, "clouds", lambda: ["a", "b"])
    monkeypatch.setattr(av, "check", lambda c, *a, **k: [
        cap(provider=c, state=av.AVAILABLE if c == "a" else av.UNKNOWN)])
    assert [c.provider for c in av.available("H100")] == ["a"]


# -- waiting -----------------------------------------------------------------


def test_wait_for_returns_as_soon_as_capacity_appears(monkeypatch):
    seq = [[cap(state=av.UNAVAILABLE)], [cap(state=av.UNAVAILABLE)], [cap()]]
    monkeypatch.setattr(av, "scan", lambda *a, **k: seq.pop(0))
    monkeypatch.setattr(av.time, "sleep", lambda s: None)
    got = av.wait_for("H100", poll_s=0)
    assert got[0].ok and not seq


def test_wait_for_zero_timeout_checks_once_and_stops(monkeypatch):
    """`if deadline` treats 0 as no-deadline and loops forever. This is the
    exact bug liveness.py shipped with; it is a test, not a comment."""
    calls = []
    monkeypatch.setattr(av, "scan",
                        lambda *a, **k: calls.append(1) or [cap(state=av.UNAVAILABLE)])
    monkeypatch.setattr(av.time, "sleep", lambda s: pytest.fail("slept on a 0 timeout"))
    got = av.wait_for("H100", timeout_s=0)
    assert len(calls) == 1
    assert not any(c.ok for c in got)


def test_wait_for_gives_up_and_returns_what_it_saw(monkeypatch):
    monkeypatch.setattr(av, "scan", lambda *a, **k: [cap(state=av.UNAVAILABLE)])
    monkeypatch.setattr(av.time, "sleep", lambda s: None)
    got = av.wait_for("H100", timeout_s=0.01, poll_s=0)
    assert got and not any(c.ok for c in got)


def test_wait_for_reports_each_poll(monkeypatch):
    seen = []
    seq = [[cap(state=av.UNAVAILABLE)], [cap()]]
    monkeypatch.setattr(av, "scan", lambda *a, **k: seq.pop(0))
    monkeypatch.setattr(av.time, "sleep", lambda s: None)
    av.wait_for("H100", poll_s=0, on_poll=seen.append)
    assert len(seen) == 2


# -- vendor scaling ----------------------------------------------------------


def test_builtin_vendors_are_registered():
    assert "verda" in list_availabilities()
    assert get_availability("verda") is av.VerdaAvailability


def test_vendor_with_pricing_but_no_probe_still_gets_searched(monkeypatch):
    """The scalability claim: a new vendor needs zero availability code."""
    monkeypatch.setitem(pricing.PROBES, "newcloud", lambda g, c: [])
    p = av._probe("newcloud")
    assert isinstance(p, av.OffersProbe) and p.name == "newcloud"
    assert "newcloud" in av.clouds()


def test_unknown_vendor_is_a_loud_keyerror():
    with pytest.raises(KeyError):
        av._probe("not-a-cloud")


def test_registering_a_vendor_needs_one_method():
    @av.register_availability("demo-vendor")
    class Demo(av.AvailabilityProbe):
        name = "demo-vendor"

        def probe(self, gpu, count, region, spot):
            return [cap(provider="demo-vendor")]

    assert av.check("demo-vendor", "H100")[0].ok


def test_offers_probe_maps_the_stock_flag(monkeypatch):
    def fake(cloud, gpu, count, spot=True):
        return [pricing.Offer(cloud, "fin-01", gpu, count, 2.0, spot, True),
                pricing.Offer(cloud, "fin-02", gpu, count, 1.0, spot, False)]

    monkeypatch.setattr(pricing, "live_offers", fake)
    p = av.OffersProbe()
    p.name = "x"
    got = {c.region: c.state for c in p.check("H100")}
    assert got == {"fin-01": av.AVAILABLE, "fin-02": av.UNAVAILABLE}


def test_region_filter_narrows_the_answer(monkeypatch):
    def fake(cloud, gpu, count, spot=True):
        return [pricing.Offer(cloud, "fin-01", gpu, count, 2.0, spot, True),
                pricing.Offer(cloud, "ice-01", gpu, count, 2.0, spot, True)]

    monkeypatch.setattr(pricing, "live_offers", fake)
    p = av.OffersProbe()
    p.name = "x"
    assert [c.region for c in p.check("H100", region="ice-01")] == ["ice-01"]


# -- verda, where the expensive lesson lives ---------------------------------


class FakeVerda:
    """Records how availability was asked, because asking it wrong is what
    made the aggregate listing disagree with spot launches all afternoon."""

    def __init__(self, answer=True):
        self.answer, self.asked = answer, []

    def offers(self, gpu=None, count=1, spot=None):
        from evsys_sdk.compute.provider import Offer as POffer
        return [POffer(sku="1H200.141S.30V", region="FIN-02", gpu="H200",
                       count=1, usd_hr=2.19, spot=bool(spot))]

    def available(self, sku, region, spot=True):
        self.asked.append((sku, region, spot))
        return self.answer


def _patch_verda(monkeypatch, fake):
    import evsys_sdk.compute.providers_verda as pv
    monkeypatch.setattr(pv, "VerdaProvider", lambda *a, **k: fake)


def test_verda_asks_per_sku_with_the_spot_flag(monkeypatch):
    fake = FakeVerda(answer=True)
    _patch_verda(monkeypatch, fake)
    got = av.check("verda", "H200", 1, spot=True)
    assert got[0].ok and got[0].sku == "1H200.141S.30V"
    assert fake.asked == [("1H200.141S.30V", "FIN-02", True)]


def test_verda_spot_and_ondemand_are_different_questions(monkeypatch):
    fake = FakeVerda()
    _patch_verda(monkeypatch, fake)
    av.check("verda", "H200", 1, spot=False)
    assert fake.asked[0][2] is False


def test_verda_no_capacity_is_unavailable_not_unknown(monkeypatch):
    _patch_verda(monkeypatch, FakeVerda(answer=False))
    got = av.check("verda", "H200", 1)
    assert [c.state for c in got] == [av.UNAVAILABLE]


def test_verda_sku_check_failing_is_unknown(monkeypatch):
    fake = FakeVerda()
    fake.available = lambda *a, **k: (_ for _ in ()).throw(OSError("503"))
    _patch_verda(monkeypatch, fake)
    got = av.check("verda", "H200", 1)
    assert [c.state for c in got] == [av.UNKNOWN]


def test_describe_is_readable():
    s = av.Capacity("verda", "H200", 1, av.AVAILABLE, region="FIN-02",
                    usd_hr=2.19).describe()
    assert "verda" in s and "FIN-02" in s and "available" in s and "spot" in s


class TestBothPurchaseModes:
    """Verda's is_spot defaults to "false" (documented). Both mistakes are
    expensive: omitting it reads on-demand stock while launching spot, and
    searching spot-only hides on-demand 1x/2x machines that were purchasable
    the whole time."""

    def test_default_asks_about_both_modes(self, monkeypatch):
        seen = []

        def fake(cloud, gpu, count, spot=True):
            seen.append(spot)
            return []

        monkeypatch.setattr(pricing, "live_offers", fake)
        p = av.OffersProbe()
        p.name = "x"
        p.check("H100")
        assert seen == [None], "None must reach the provider as 'either mode'"

    def test_an_explicit_mode_is_still_honoured(self, monkeypatch):
        seen = []

        def fake(cloud, gpu, count, spot=True):
            seen.append(spot)
            return []

        monkeypatch.setattr(pricing, "live_offers", fake)
        p = av.OffersProbe()
        p.name = "x"
        p.check("H100", spot=True)
        assert seen == [True]

    def test_on_demand_capacity_is_purchasable_not_ignored(self, monkeypatch):
        """1xH100 on-demand at $3.25 is worse than spot at $1.14 — but it is
        infinitely better than nothing, which is what spot-only returned."""
        def fake(cloud, gpu, count, spot=None):
            return [pricing.Offer(cloud, "FIN-02", gpu, count, 3.25, False, True)]

        monkeypatch.setitem(pricing.PROBES, "x", lambda g, c: [])
        monkeypatch.setattr(pricing, "live_offers", fake)
        monkeypatch.setattr(av, "clouds", lambda: ["x"])
        got = av.available("H100")
        assert [(c.spot, c.usd_hr) for c in got] == [(False, 3.25)]

    def test_spot_is_preferred_when_both_are_free(self, monkeypatch):
        """Cheaper per GPU-hour wins, which on Verda is always spot."""
        def fake(cloud, gpu, count, spot=None):
            return [pricing.Offer(cloud, "FIN-02", gpu, count, 3.25, False, True),
                    pricing.Offer(cloud, "FIN-02", gpu, count, 1.138, True, True)]

        monkeypatch.setitem(pricing.PROBES, "x", lambda g, c: [])
        monkeypatch.setattr(pricing, "live_offers", fake)
        monkeypatch.setattr(av, "clouds", lambda: ["x"])
        assert av.available("H100")[0].spot is True

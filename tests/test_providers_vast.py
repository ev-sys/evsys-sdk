"""Vast.ai's implementation of the provider contract.

Vast is a marketplace, not a fixed price list, and each test here pins a place
where treating it like one produces a confidently wrong number.
"""
from __future__ import annotations

import json
import urllib.parse

import pytest

from evsys_sdk.compute import provider as pv
from evsys_sdk.compute.providers_vast import VastProvider


@pytest.fixture
def api(monkeypatch):
    """Stub the transport only.

    Deliberately stubs ``_send`` rather than ``_get``: the query is built by
    ``_get`` as urlencoded JSON, and the multi-variant ``in`` clause is exactly
    the thing that was broken, so that construction must stay real and
    inspectable.
    """
    state = {"offers": [], "instances": [], "post": {"success": True,
                                                     "new_contract": 991},
             "calls": [], "queries": []}

    def fake_send(req):
        url = req.full_url
        body = json.loads(req.data.decode()) if req.data else None
        state["calls"].append((url, body, req.get_method()))
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if "q" in qs:
            state["queries"].append(json.loads(qs["q"][0]))
        if req.data is not None or req.get_method() in ("PUT", "DELETE"):
            return state["post"]
        if "/instances" in url:
            return {"instances": state["instances"]}
        if "/ssh" in url:
            return {"ssh_keys": [{"id": 1}]}
        q = state["queries"][-1] if state["queries"] else {}
        rows = state["offers"]
        if "id" in q:                       # id lookup used by launch()
            rows = [r for r in rows if r["id"] == q["id"]["eq"]]
        return {"offers": rows}

    monkeypatch.setattr(VastProvider, "_send", staticmethod(fake_send))
    monkeypatch.setenv("VAST_API_KEY", "k")
    return state


def offer(**kw):
    base = dict(id=1, num_gpus=1, gpu_name="H100 SXM", dph_total=1.491,
                min_bid=1.480, reliability2=0.995, geolocation="Montana, US",
                rentable=True, rented=False)
    base.update(kw)
    return base


class TestPriceIsPerOfferNotPerGpu:
    """Measured on the live market: the cheapest 1x H100 was $0.99/hr and the
    cheapest 8x was $18.38/hr. Read as per-GPU that ranks the 8x box as 18x
    worse; it is actually $0.99 vs $2.30 per GPU-hour."""

    def test_usd_hr_is_the_whole_offer(self, api):
        api["offers"] = [offer(num_gpus=8, dph_total=18.378, min_bid=17.280)]
        o = VastProvider().offers(gpu="H100", count=8, spot=False)[0]
        assert o.usd_hr == 18.378

    def test_comparison_happens_per_gpu(self, api):
        api["offers"] = [offer(num_gpus=8, dph_total=18.378, min_bid=17.280)]
        o = VastProvider().offers(gpu="H100", count=8, spot=False)[0]
        assert round(o.usd_per_gpu_hr, 3) == 2.297

    def test_a_bigger_box_can_win_on_price_per_gpu(self, api):
        """The ordering must not simply prefer small machines."""
        api["offers"] = [offer(id=1, num_gpus=1, dph_total=4.00, min_bid=4.00),
                         offer(id=2, num_gpus=8, dph_total=16.0, min_bid=16.0)]
        got = VastProvider().offers(gpu="H100", count=0, spot=False)
        assert [o.sku for o in got] == ["2", "1"]      # $2.00 vs $4.00 per gpu


class TestBidIsAFloorNotAPrice:
    """On Verda the spot price is the price. On Vast min_bid is the floor you
    must clear, and bidding higher makes you less likely to be outbid — so
    what you pay is a risk decision, not a quoted fact."""

    def test_spot_price_recorded_is_the_min_bid(self, api):
        api["offers"] = [offer(dph_total=0.990, min_bid=0.400)]
        o = VastProvider().offers(gpu="H100", count=1, spot=True)[0]
        assert o.usd_hr == 0.400 and o.spot is True

    def test_on_demand_price_is_dph_total(self, api):
        api["offers"] = [offer(dph_total=0.990, min_bid=0.400)]
        o = VastProvider().offers(gpu="H100", count=1, spot=False)[0]
        assert o.usd_hr == 0.990 and o.spot is False

    def test_one_offer_yields_both_modes_when_unspecified(self, api):
        """The same machine is purchasable two ways at two prices."""
        api["offers"] = [offer(dph_total=0.990, min_bid=0.400)]
        got = VastProvider().offers(gpu="H100", count=1, spot=None)
        assert sorted((o.spot, o.usd_hr) for o in got) == [(False, 0.990),
                                                           (True, 0.400)]

    def test_default_bid_clears_the_floor_with_margin(self, api):
        """Bidding exactly min_bid means the next bidder takes the machine."""
        api["offers"] = [offer(id=7, min_bid=1.00)]
        m = VastProvider().launch("7", "US", spot=True)
        assert m.usd_hr > 1.00
        body = next(b for u, b, _ in api["calls"] if "asks/7" in u)
        assert body["price"] > 1.00

    def test_an_explicit_bid_is_honoured(self, api):
        api["offers"] = [offer(id=7, min_bid=1.00)]
        m = VastProvider().launch("7", "US", spot=True, bid=2.5)
        assert m.usd_hr == 2.5

    def test_on_demand_rental_sends_no_bid(self, api):
        api["offers"] = [offer(id=7, min_bid=1.00, dph_total=3.0)]
        VastProvider().launch("7", "US", spot=False)
        body = next(b for u, b, _ in api["calls"] if "asks/7" in u)
        assert "price" not in body


class TestGpuNamesNeedTheInOperator:
    """gpu_name filters on exact equality — there is no prefix match. Asking
    for {"eq": "H100"} matched nothing while 30 H100s were rentable, because
    no offer is literally named "H100"."""

    def test_a_family_queries_every_spelling(self, api):
        api["offers"] = []
        VastProvider().offers(gpu="H100", count=1)
        assert api["queries"][-1]["gpu_name"] == {
            "in": ["H100 SXM", "H100 PCIE", "H100 NVL"]}

    def test_h100_does_not_match_h200(self, api):
        """Two very different cards at very different prices."""
        names = api["queries"]
        VastProvider().offers(gpu="H100", count=1)
        assert not any("H200" in n for n in names[-1]["gpu_name"]["in"])
        VastProvider().offers(gpu="H200", count=1)
        assert not any("H100" in n for n in names[-1]["gpu_name"]["in"])

    @pytest.mark.parametrize("typed,expected", [
        ("A100", ["A100 SXM4", "A100 PCIE"]),
        ("a100", ["A100 SXM4", "A100 PCIE"]),          # case-insensitive
        ("RTX 6000Ada", ["RTX 6000Ada"]),              # spaces and mixed case
        ("4090", ["RTX 4090"]),                        # vendor prefix implied
        ("V100", ["Tesla V100"]),                      # vendor prefix implied
    ])
    def test_names_people_type_reach_the_marketplace_spelling(
            self, api, typed, expected):
        api["offers"] = []
        VastProvider().offers(gpu=typed, count=1)
        assert api["queries"][-1]["gpu_name"] == {"in": expected}

    def test_an_unmapped_card_falls_back_to_exact_match(self, api):
        """A brand-new card must stay reachable by its literal name rather
        than silently returning nothing."""
        api["offers"] = []
        VastProvider().offers(gpu="RTX 5080 Ti", count=1)
        assert api["queries"][-1]["gpu_name"] == {"eq": "RTX 5080 Ti"}

    def test_the_variant_is_kept_not_collapsed(self, api):
        """H100 PCIE bid at $0.40 against H100 SXM at $1.48 — reporting both
        as "H100" hides which machine the price belongs to."""
        api["offers"] = [offer(id=1, gpu_name="H100 PCIE", min_bid=0.400),
                         offer(id=2, gpu_name="H100 SXM", min_bid=1.480)]
        got = VastProvider().offers(gpu="H100", count=1, spot=True)
        assert {o.gpu for o in got} == {"H100 PCIE", "H100 SXM"}


class TestTheSearchIsTheAvailabilityQuery:
    """There is no separate capacity endpoint. Anything returned rentable is
    rentable now, so the listing cannot disagree with itself the way Verda's
    aggregate did."""

    def test_offers_are_filtered_to_rentable_and_unrented(self, api):
        api["offers"] = []
        VastProvider().offers(gpu="H100", count=1)
        q = api["queries"][-1]
        assert q["rentable"] == {"eq": True} and q["rented"] == {"eq": False}

    def test_available_asks_about_the_offer_id(self, api):
        """Offer ids are ephemeral: a machine someone else rents leaves the
        market entirely, so the question is about the id, not the shape."""
        api["offers"] = [offer(id=42)]
        assert VastProvider().available("42", "US") is True
        assert api["queries"][-1]["id"] == {"eq": 42}

    def test_a_vanished_offer_is_unavailable(self, api):
        api["offers"] = []
        assert VastProvider().available("42", "US") is False

    def test_probe_reports_available_for_every_live_offer(self, api):
        from evsys_sdk.compute import availability as av
        api["offers"] = [offer()]
        got = av.check("vast", "H100", 1, spot=True)
        assert [c.state for c in got] == [av.AVAILABLE]

    def test_probe_says_unavailable_rather_than_empty(self, api):
        from evsys_sdk.compute import availability as av
        api["offers"] = []
        got = av.check("vast", "H100", 1, spot=True)
        assert [c.state for c in got] == [av.UNAVAILABLE]

    def test_a_bid_capacity_is_labelled_as_a_bid(self, api):
        """A price that is really a floor must not read as a quote."""
        from evsys_sdk.compute import availability as av
        api["offers"] = [offer()]
        c = av.check("vast", "H100", 1, spot=True)[0]
        assert "bid" in c.detail


class TestHostReliability:
    """A marketplace has no uniform SLA. Vast scores each host precisely
    because they are not interchangeable, and a cheap offer from a bad host
    is a different product."""

    def test_unreliable_hosts_are_dropped(self, api):
        api["offers"] = [offer(id=1, reliability2=0.50, min_bid=0.01),
                         offer(id=2, reliability2=0.995, min_bid=1.00)]
        got = VastProvider().offers(gpu="H100", count=1, spot=True)
        assert [o.sku for o in got] == ["2"]

    def test_the_bar_is_configurable(self, api):
        api["offers"] = [offer(id=1, reliability2=0.50, min_bid=0.01)]
        got = VastProvider(min_reliability=0.0).offers(gpu="H100", count=1,
                                                       spot=True)
        assert [o.sku for o in got] == ["1"]

    def test_a_missing_score_is_treated_as_unreliable(self, api):
        """Absent evidence of reliability is not evidence of reliability."""
        api["offers"] = [offer(id=1, reliability2=None)]
        assert VastProvider().offers(gpu="H100", count=1, spot=True) == []


class TestRentingNeedsCredentials:
    """Search is public; renting is not. Failing loudly matters because the
    alternative is a launch loop that silently never provisions anything."""

    def _no_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VAST_API_KEY", raising=False)
        return VastProvider(credentials=str(tmp_path / "absent.json"))

    def test_launch_without_a_key_raises_with_a_reason(self, api, monkeypatch,
                                                       tmp_path):
        api["offers"] = [offer(id=7)]
        p = self._no_key(monkeypatch, tmp_path)
        with pytest.raises(pv.LaunchFailed) as e:
            p.launch("7", "US")
        assert e.value.reason == "no_credentials"

    def test_polling_without_a_key_raises_rather_than_reporting_gone(
            self, api, monkeypatch, tmp_path):
        """Reporting GONE would look like a preemption and trigger a rebuy."""
        p = self._no_key(monkeypatch, tmp_path)
        m = pv.Machine("1", "vast", "7", "US", "H100 SXM", 1, 1.0)
        with pytest.raises(pv.LaunchFailed):
            p.poll(m)

    def test_searching_still_works_without_a_key(self, api, monkeypatch,
                                                 tmp_path):
        api["offers"] = [offer()]
        p = self._no_key(monkeypatch, tmp_path)
        assert p.offers(gpu="H100", count=1, spot=True)

    def test_a_malformed_credentials_file_is_not_a_crash(self, monkeypatch,
                                                         tmp_path):
        f = tmp_path / "c.json"
        f.write_text("{not json")
        monkeypatch.delenv("VAST_API_KEY", raising=False)
        assert VastProvider(credentials=str(f))._key() == ""

    def test_an_auth_rejection_is_classified_as_credentials(self):
        assert VastProvider._reason(401, "unauthorized") == "no_credentials"

    def test_a_gone_offer_is_capacity_not_a_mystery(self):
        assert VastProvider._reason(404, "not found") == "no_capacity"
        assert VastProvider._reason(400, "insufficient credit") == "no_funds"


class TestListingIsCappedAtSixtyFour:
    """The API caps limit server-side regardless of what is asked, so every
    query is a sample. Ordering by price makes it the cheapest sample, which
    is the only one that answers "what is the cheapest way to run this"."""

    def test_asks_for_the_cap_and_orders_by_price(self, api):
        from evsys_sdk.compute.providers_vast import MAX_OFFERS
        api["offers"] = []
        VastProvider().offers(gpu="H100", count=1)
        q = api["queries"][-1]
        assert q["limit"] == MAX_OFFERS == 64
        assert q["order"] == [["dph_total", "asc"]]

    def test_launch_finds_an_offer_outside_the_cheapest_page(self, api):
        """Looking the id up in a 64-row price-sorted page would miss any
        expensive offer, and report a live machine as gone."""
        api["offers"] = [offer(id=999, dph_total=99.0, min_bid=98.0)]
        m = VastProvider().launch("999", "US", spot=True)
        assert m.id == "991" and m.gpu == "H100 SXM"

    def test_launching_a_gone_offer_says_so(self, api):
        api["offers"] = []
        with pytest.raises(pv.LaunchFailed) as e:
            VastProvider().launch("999", "US")
        assert e.value.reason == "no_capacity"


class TestPoll:
    def test_running_with_a_public_ip(self, api):
        api["instances"] = [{"id": 991, "actual_status": "running",
                             "public_ipaddr": "1.2.3.4"}]
        m = pv.Machine("991", "vast", "7", "US", "H100 SXM", 1, 1.0)
        assert VastProvider().poll(m).state == pv.RUNNING and m.ip == "1.2.3.4"

    def test_absent_means_gone(self, api):
        api["instances"] = []
        m = pv.Machine("991", "vast", "7", "US", "H100 SXM", 1, 1.0,
                       state=pv.RUNNING)
        assert VastProvider().poll(m).state == pv.GONE

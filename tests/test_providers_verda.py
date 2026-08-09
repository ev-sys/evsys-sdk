"""Verda's implementation of the provider contract.

Each test pins a quirk of the real API that broke a launch today.
"""
from __future__ import annotations

import json
import time

import pytest

from evsys_sdk.compute import provider as pv
from evsys_sdk.compute.providers_verda import VerdaProvider


@pytest.fixture
def api(monkeypatch):
    """Stub the HTTP layer, keep every line of parsing real."""
    state = {"responses": {}, "calls": []}

    def fake_call(self, path, body=None, method=None):
        state["calls"].append((path, body, method))
        for k, v in state["responses"].items():
            if path.startswith(k):
                return v(body) if callable(v) else v
        return {}

    monkeypatch.setattr(VerdaProvider, "_call", fake_call)
    return state


class TestAvailabilityIsAuthoritative:
    def test_asks_the_per_sku_endpoint_with_is_spot(self, api):
        """The aggregate listing answers about ON-DEMAND stock unless is_spot
        is passed. Reading it and launching spot disagreed all afternoon."""
        api["responses"]["instance-availability/1H200"] = True
        assert VerdaProvider().available("1H200", "FIN-02", spot=True) is True
        path = api["calls"][-1][0]
        assert path.startswith("instance-availability/1H200")
        assert "is_spot=true" in path and "location_code=FIN-02" in path

    def test_spot_false_asks_for_on_demand(self, api):
        api["responses"]["instance-availability/1H200"] = False
        VerdaProvider().available("1H200", "FIN-02", spot=False)
        assert "is_spot=false" in api["calls"][-1][0]

    def test_anything_other_than_true_is_unavailable(self, api):
        """The endpoint returns a bare boolean; a dict or None is not a yes."""
        for val in (False, None, {}, "true"):
            api["responses"]["instance-availability/x"] = val
            assert VerdaProvider().available("x", "R") is False


class TestLaunch:
    def _ok(self, api):
        api["responses"]["sshkeys"] = [{"id": "k1"}]
        api["responses"]["instance-types"] = [
            {"instance_type": "1H200", "model": "H200",
             "gpu": {"number_of_gpus": 1}, "spot_price": "1.40",
             "price_per_hour": "4.00"}]
        api["responses"]["instances"] = "i-abc123"

    def test_returns_the_bare_id_string_the_api_gives(self, api):
        """POST /instances answers with a raw id, not JSON."""
        self._ok(api)
        m = VerdaProvider().launch("1H200", "FIN-02")
        assert m.id == "i-abc123" and m.gpu == "H200" and m.usd_hr == 1.40

    def test_sends_the_contract_enum_not_just_is_spot(self, api):
        self._ok(api)
        VerdaProvider().launch("1H200", "FIN-02", spot=True)
        body = next(b for p, b, _ in api["calls"] if p == "instances" and b)
        assert body["is_spot"] is True and body["contract"] == "SPOT"

    def test_refuses_without_a_registered_key(self, api):
        api["responses"]["sshkeys"] = []
        with pytest.raises(pv.LaunchFailed) as e:
            VerdaProvider().launch("1H200", "FIN-02")
        assert e.value.reason == "unsupported"

    def test_tracks_the_volume_it_created(self, api):
        """The disk outlives the instance; forgetting it means it bills on."""
        self._ok(api)
        p = VerdaProvider()
        p.launch("1H200", "FIN-02", name="run")
        assert any(v.startswith("run-") for v in p._owned_volumes)


class TestErrorClassification:
    def test_no_capacity_and_no_funds_are_distinguished(self):
        """SkyPilot flattens both into ResourcesUnavailableError. They are
        opposite signals: one is about the provider, one about your wallet."""
        assert VerdaProvider._reason(503, "Not enough resources to deploy") == "no_capacity"
        assert VerdaProvider._reason(400, "Insufficient funds in the wallet") == "no_funds"

    def test_decommissioned_hardware_is_unsupported_not_capacity(self):
        """HDD is listed in the price table and rejects every clone."""
        assert VerdaProvider._reason(400, "HDD storage is being decommissioned") == "unsupported"

    def test_unknown_codes_keep_the_status(self):
        assert VerdaProvider._reason(418, "teapot") == "http_418"


class TestPoll:
    def test_running_with_an_ip(self, api):
        api["responses"]["instances"] = [{"id": "i-1", "status": "running",
                                          "ip": "1.2.3.4"}]
        m = pv.Machine("i-1", "verda", "s", "R", "H200", 1, 1.4)
        assert VerdaProvider().poll(m).state == pv.RUNNING and m.ip == "1.2.3.4"

    def test_absent_from_the_list_means_gone(self, api):
        """Preemption shows up as the instance simply not being there."""
        api["responses"]["instances"] = []
        m = pv.Machine("i-1", "verda", "s", "R", "H200", 1, 1.4, state=pv.RUNNING)
        assert VerdaProvider().poll(m).state == pv.GONE


class TestOrphanedDisks:
    def test_detached_volumes_are_reported(self, api):
        api["responses"]["volumes"] = [
            {"id": "v1", "name": "dead", "instance_id": None, "monthly_price": 60},
            {"id": "v2", "name": "live", "instance_id": "i-9", "monthly_price": 60}]
        assert [n for _, n, _ in VerdaProvider().orphans()] == ["dead"]

    def test_a_cloning_volume_is_not_reported_as_reclaimable(self, api):
        """It accepts a delete request and silently ignores it, so reporting
        it invites a fire-and-forget delete that never happens."""
        api["responses"]["volumes"] = [
            {"id": "v1", "name": "mid", "instance_id": None,
             "status": "cloning", "monthly_price": 60}]
        assert VerdaProvider().orphans() == []

    def test_terminate_releases_the_disk_it_created(self, api):
        api["responses"]["sshkeys"] = [{"id": "k1"}]
        api["responses"]["instance-types"] = [
            {"instance_type": "1H200", "model": "H200",
             "gpu": {"number_of_gpus": 1}, "spot_price": "1.40"}]
        api["responses"]["instances"] = "i-abc"
        p = VerdaProvider()
        m = p.launch("1H200", "FIN-02", name="run")
        vol = next(iter(p._owned_volumes))
        api["responses"]["volumes"] = [{"id": "v1", "name": vol,
                                        "instance_id": None, "monthly_price": 60}]
        p.terminate(m)
        deletes = [b for pth, b, mth in api["calls"]
                   if pth == "volumes" and mth == "PUT"]
        assert deletes and deletes[0]["is_permanent"] is True


class TestOffers:
    def test_priced_per_gpu_and_only_where_stock_exists(self, api):
        api["responses"]["instance-types"] = [
            {"instance_type": "1H200", "model": "H200",
             "gpu": {"number_of_gpus": 1}, "spot_price": "1.40"},
            {"instance_type": "1H100", "model": "H100",
             "gpu": {"number_of_gpus": 1}, "spot_price": "1.14"}]
        api["responses"]["instance-availability"] = [
            {"location_code": "FIN-02", "availabilities": ["1H200"]}]
        got = VerdaProvider().offers(count=1, spot=True)
        assert [o.sku for o in got] == ["1H200"]      # 1H100 has no stock
        assert got[0].region == "FIN-02"


class TestGpuNamesAreFreeText:
    """Verda's `model` field is prose: "A100 80GB", "RTX PRO 6000",
    "Tesla V100". Equality matching against what people type found nothing and
    reported it as "this provider does not sell A100s"."""

    TYPES = [
        {"instance_type": "1A100.22V", "model": "A100 80GB",
         "gpu": {"number_of_gpus": 1}, "spot_price": "0.8"},
        {"instance_type": "1A6000.10V", "model": "RTX A6000",
         "gpu": {"number_of_gpus": 1}, "spot_price": "0.3"},
        {"instance_type": "4RTXPRO6000.120V", "model": "RTX PRO 6000",
         "gpu": {"number_of_gpus": 4}, "spot_price": "2.646"},
        {"instance_type": "1B300.30V", "model": "B300",
         "gpu": {"number_of_gpus": 1}, "spot_price": "3.0"},
        {"instance_type": "1GB300.32V", "model": "GB300",
         "gpu": {"number_of_gpus": 1}, "spot_price": "4.0"},
    ]

    def _stocked(self, api, skus):
        api["responses"]["instance-types"] = self.TYPES
        api["responses"]["instance-availability?is_spot=true"] = [
            {"location_code": "FIN-03", "availabilities": skus}]

    @pytest.mark.parametrize("want,sku", [
        ("A100", "1A100.22V"),            # memory suffix in the model name
        ("A6000", "1A6000.10V"),          # vendor prefix in the model name
        ("RTXPRO6000", "4RTXPRO6000.120V"),  # spaces in the model name
    ])
    def test_finds_gpus_whose_model_name_is_not_the_name_people_type(
            self, api, want, sku):
        self._stocked(api, [s["instance_type"] for s in self.TYPES])
        count = next(t["gpu"]["number_of_gpus"] for t in self.TYPES
                     if t["instance_type"] == sku)
        got = VerdaProvider().offers(gpu=want, count=count, spot=True)
        assert [o.sku for o in got] == [sku]

    def test_b300_is_not_gb300(self, api):
        """Substring matching must not merge two different machines."""
        self._stocked(api, ["1B300.30V", "1GB300.32V"])
        got = VerdaProvider().offers(gpu="B300", count=1, spot=True)
        assert [o.sku for o in got] == ["1B300.30V"]


class TestStockIsPerPurchaseMode:
    """The bug that hid live capacity: regions were enumerated from the
    aggregate listing with no is_spot, so spot-only stock was invisible.
    Verda had 8xH200 free on spot in FIN-03 while we reported none."""

    TYPES = [{"instance_type": "8H200.141S.176V", "model": "H200",
              "gpu": {"number_of_gpus": 8}, "spot_price": "11.2",
              "price_per_hour": "26.0"}]

    def test_spot_stock_is_queried_with_is_spot_true(self, api):
        api["responses"]["instance-types"] = self.TYPES
        api["responses"]["instance-availability?is_spot=true"] = [
            {"location_code": "FIN-03", "availabilities": ["8H200.141S.176V"]}]
        api["responses"]["instance-availability?is_spot=false"] = []
        got = VerdaProvider().offers(gpu="H200", count=8, spot=True)
        assert [(o.region, o.usd_hr) for o in got] == [("FIN-03", 11.2)]

    def test_on_demand_stock_is_a_different_question(self, api):
        api["responses"]["instance-types"] = self.TYPES
        api["responses"]["instance-availability?is_spot=true"] = [
            {"location_code": "FIN-03", "availabilities": ["8H200.141S.176V"]}]
        api["responses"]["instance-availability?is_spot=false"] = []
        assert VerdaProvider().offers(gpu="H200", count=8, spot=False) == []

    def test_unspecified_mode_asks_both(self, api):
        api["responses"]["instance-types"] = self.TYPES
        api["responses"]["instance-availability?is_spot=true"] = [
            {"location_code": "FIN-03", "availabilities": ["8H200.141S.176V"]}]
        api["responses"]["instance-availability?is_spot=false"] = [
            {"location_code": "FIN-01", "availabilities": ["8H200.141S.176V"]}]
        got = VerdaProvider().offers(gpu="H200", count=8, spot=None)
        assert {(o.region, o.spot) for o in got} == {
            ("FIN-03", True), ("FIN-01", False)}


class TestTokenIsCached:
    """A token per call doubled every request and made a 47-SKU sweep time
    out. Tokens last ~10 minutes; minting one each time is pure waste."""

    def _stub_oauth(self, monkeypatch, expires_in=600):
        import evsys_sdk.compute.providers_verda as pv
        calls = []

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"access_token": "tok",
                                   "expires_in": expires_in}).encode()

        def fake_open(req, timeout=None):
            calls.append(req.full_url)
            return FakeResp()

        monkeypatch.setattr(pv.urllib.request, "urlopen", fake_open)
        monkeypatch.setattr(pv.json, "load", lambda f: json.loads(f.read()))
        monkeypatch.setattr(pv.pathlib.Path, "read_text",
                            lambda self: json.dumps({"client_id": "i",
                                                     "client_secret": "s"}))
        return calls

    def test_repeated_calls_mint_one_token(self, monkeypatch):
        calls = self._stub_oauth(monkeypatch)
        p = VerdaProvider()
        for _ in range(5):
            p._token()
        assert len(calls) == 1

    def test_an_expired_token_is_refetched(self, monkeypatch):
        calls = self._stub_oauth(monkeypatch)
        p = VerdaProvider()
        p._token()
        p._tok_exp = 0          # pretend it aged out
        p._token()
        assert len(calls) == 2

    def test_expiry_keeps_a_safety_margin(self, monkeypatch):
        """A token that expires mid-flight fails the call it was minted for."""
        self._stub_oauth(monkeypatch, expires_in=600)
        p = VerdaProvider()
        p._token()
        assert p._tok_exp < time.time() + 600


class TestStockListingIsComplete:
    """Verified against the live API: all 47 GPU SKUs x 3 locations x both
    modes probed per-SKU, and the aggregate listing omitted none of them. So
    one call per mode is correct, and 282 are waste."""

    def test_region_narrows_the_listing_query(self, api):
        api["responses"]["instance-types"] = []
        VerdaProvider()._stock(True, region="FIN-03")
        path = api["calls"][-1][0]
        assert "is_spot=true" in path and "location_code=FIN-03" in path

    def test_no_region_asks_about_all_of_them(self, api):
        api["responses"]["instance-types"] = []
        VerdaProvider()._stock(True)
        assert "location_code" not in api["calls"][-1][0]


class TestListingsAreCached:
    """A scan across GPU families refetched the 47-SKU catalogue once per
    probe: 132s for one sweep, against a scheduler that polls every 60s."""

    def test_the_catalogue_is_fetched_once_per_sweep(self, api):
        api["responses"]["instance-types"] = []
        api["responses"]["instance-availability"] = []
        p = VerdaProvider()
        for _ in range(5):
            p.offers(gpu="H200", count=8, spot=True)
        assert sum(1 for c in api["calls"] if c[0] == "instance-types") == 1

    def test_a_stale_cache_is_refetched(self, api):
        api["responses"]["instance-types"] = []
        api["responses"]["instance-availability"] = []
        p = VerdaProvider()
        p.offers(gpu="H200", count=8, spot=True)
        p._cache = {k: (0.0, v) for k, (_, v) in p._cache.items()}
        p.offers(gpu="H200", count=8, spot=True)
        assert sum(1 for c in api["calls"] if c[0] == "instance-types") == 2

    def test_availability_is_never_cached(self, api):
        """Stock changes minute to minute — a cached yes is how you launch
        into a machine that is already gone."""
        api["responses"]["instance-availability/1H200"] = True
        p = VerdaProvider()
        for _ in range(3):
            p.available("1H200", "FIN-02", spot=True)
        n = sum(1 for c in api["calls"] if c[0].startswith("instance-availability/"))
        assert n == 3

    def test_cache_cannot_outlive_an_availability_answer(self):
        """A listing older than the Capacity built from it would let a stale
        answer look fresh."""
        from evsys_sdk.compute import availability as av
        assert VerdaProvider.CACHE_TTL_S < av.DEFAULT_TTL_S


class TestLaunchRecordsThePriceActuallyPaid:
    """An on-demand machine was recorded at its spot price — 2.9x low. That
    figure feeds the reliability ledger and every cost comparison after it."""

    TYPES = [{"instance_type": "1RTXPRO6000.30V", "model": "RTX PRO 6000",
              "gpu": {"number_of_gpus": 1}, "spot_price": "0.6615",
              "price_per_hour": "1.89"}]

    def _ready(self, api):
        api["responses"]["instance-types"] = self.TYPES
        api["responses"]["sshkeys"] = [{"id": "k1"}]
        api["responses"]["instances"] = "inst-1"

    def test_on_demand_launch_records_the_on_demand_price(self, api):
        self._ready(api)
        m = VerdaProvider().launch("1RTXPRO6000.30V", "FIN-03", spot=False)
        assert m.usd_hr == 1.89

    def test_spot_launch_records_the_spot_price(self, api):
        self._ready(api)
        m = VerdaProvider().launch("1RTXPRO6000.30V", "FIN-03", spot=True)
        assert m.usd_hr == 0.6615

    def test_on_demand_sends_the_right_contract(self, api):
        self._ready(api)
        VerdaProvider().launch("1RTXPRO6000.30V", "FIN-03", spot=False)
        body = next(b for p, b, _ in api["calls"] if p == "instances" and b)
        assert body["is_spot"] is False and body["contract"] == "PAY_AS_YOU_GO"

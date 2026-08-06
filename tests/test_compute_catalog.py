"""Refreshing SkyPilot's own catalog rather than working around it.

The catalog is the single input to every price comparison and failover
ordering SkyPilot makes. Rewriting it in place keeps `sky launch`, the
optimizer and `sky show-gpus` working exactly as designed, on fresh numbers.
"""
from __future__ import annotations

import csv

import pytest

from evsys_sdk.compute import catalog

HEADER = ["InstanceType", "vCPUs", "AcceleratorName", "AcceleratorCount",
          "Region", "Price", "SpotPrice"]
ROWS = [
    ["1H100.80S", "30", "H100", "1.0", "FIN-01", "3.25", "1.138"],
    ["2H100.80S", "60", "H100", "2.0", "FIN-02", "6.50", "2.275"],
    ["1EXOTIC",   "8",  "MYSTERY", "1.0", "FIN-01", "9.99", "4.44"],
]


@pytest.fixture
def cat(tmp_path):
    p = tmp_path / "verda" / "vms.csv"
    p.parent.mkdir(parents=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(ROWS)
    return p


def read(p):
    with p.open() as f:
        return list(csv.DictReader(f))


class TestRefresh:
    def test_rewrites_prices_from_the_live_index(self, cat):
        def idx(c, g, n):
            return {("H100", 1, "FIN-01"): (4.00, 1.50)}
        got = catalog.refresh("verda", path=cat, index=idx)
        rows = read(cat)
        assert rows[0]["Price"] == "4" and rows[0]["SpotPrice"] == "1.5"
        assert got["updated"] == 1

    def test_unpriced_rows_keep_their_catalogued_price(self, cat):
        """A missing row means SkyPilot will not consider that machine at all,
        which is worse than an out-of-date number."""
        def idx(c, g, n):
            return {("H100", 1, "FIN-01"): (4.00, 1.50)}
        catalog.refresh("verda", path=cat, index=idx)
        rows = read(cat)
        assert len(rows) == 3
        exotic = next(r for r in rows if r["AcceleratorName"] == "MYSTERY")
        assert exotic["Price"] == "9.99" and exotic["SpotPrice"] == "4.44"

    def test_row_count_and_columns_are_preserved(self, cat):
        def idx(c, g, n):
            return {("H100", 2, "FIN-02"): (7.00, 2.00)}
        catalog.refresh("verda", path=cat, index=idx)
        rows = read(cat)
        assert len(rows) == 3
        assert list(rows[0].keys()) == HEADER

    def test_matches_on_gpu_count_and_region_together(self, cat):
        """Same GPU at a different count or region is a different product at a
        different price; matching on the name alone would corrupt both."""
        def idx(c, g, n):
            return {("H100", 2, "FIN-02"): (7.00, 2.00)}
        catalog.refresh("verda", path=cat, index=idx)
        rows = read(cat)
        assert rows[0]["SpotPrice"] == "1.138"     # 1x FIN-01 untouched
        assert rows[1]["SpotPrice"] == "2"         # 2x FIN-02 updated

    def test_only_the_prices_that_exist_are_written(self, cat):
        """A provider offering spot but no on-demand must not blank the other."""
        def idx(c, g, n):
            return {("H100", 1, "FIN-01"): (None, 0.99)}
        catalog.refresh("verda", path=cat, index=idx)
        rows = read(cat)
        assert rows[0]["Price"] == "3.25" and rows[0]["SpotPrice"] == "0.99"


class TestFailsSafe:
    def test_an_unreachable_provider_leaves_the_catalog_untouched(self, cat):
        """Stale prices beat no catalog: an empty rewrite would strand every
        launch."""
        before = cat.read_text()
        got = catalog.refresh("verda", path=cat, index=lambda c, g, n: {})
        assert cat.read_text() == before
        assert got["updated"] == 0

    def test_a_missing_catalog_is_not_an_error(self, tmp_path):
        assert catalog.refresh("nosuch", path=None)["rows"] == 0

    def test_a_file_without_price_columns_is_refused(self, tmp_path):
        p = tmp_path / "vms.csv"
        p.write_text("InstanceType,Region\nx,FIN-01\n")
        got = catalog.refresh("verda", path=p,
                              index=lambda c, g, n: {("H100", 1, "FIN-01"): (1.0, 1.0)})
        assert got["updated"] == 0
        assert "InstanceType,Region" in p.read_text()

    def test_write_is_atomic_no_temp_files_left_behind(self, cat):
        def idx(c, g, n):
            return {("H100", 1, "FIN-01"): (4.00, 1.50)}
        catalog.refresh("verda", path=cat, index=idx)
        assert list(cat.parent.glob("*.csv")) == [cat]

    def test_only_queries_the_gpus_the_catalog_contains(self, cat):
        """Refreshing sixteen rows must not sweep a fifty-SKU provider."""
        seen = {}
        def idx(cloud, gpus, counts):
            seen["gpus"], seen["counts"] = set(gpus), set(counts)
            return {}
        catalog.refresh("verda", path=cat, index=idx)
        assert seen["gpus"] == {"H100", "MYSTERY"}
        assert seen["counts"] == {1, 2}


class TestDiscovery:
    def test_schema_version_is_discovered_not_hardcoded(self, monkeypatch, tmp_path):
        """Pinning it would silently target the wrong directory after an
        upgrade, and quietly refresh a catalog nobody reads."""
        (tmp_path / "v8" / "verda").mkdir(parents=True)
        (tmp_path / "v8" / "verda" / "vms.csv").write_text("InstanceType\nx\n")
        monkeypatch.setattr(catalog, "CATALOG_ROOT", str(tmp_path))
        assert catalog.catalog_dir().name == "v8"
        assert catalog.catalog_path("verda") is not None

    def test_absent_root_is_none_not_a_crash(self, monkeypatch, tmp_path):
        monkeypatch.setattr(catalog, "CATALOG_ROOT", str(tmp_path / "nope"))
        assert catalog.catalog_dir() is None
        assert catalog.catalog_path("verda") is None


class TestStalenessGate:
    def test_a_fresh_catalog_is_not_refetched(self, cat, monkeypatch):
        """Called before every launch, so it must be nearly free when fresh."""
        monkeypatch.setattr(catalog, "catalog_path", lambda c: cat)
        calls = []
        monkeypatch.setattr(catalog, "refresh", lambda c, **k: calls.append(c))
        assert catalog.refresh_if_stale("verda", max_age_s=3600) is None
        assert calls == []

    def test_a_stale_catalog_is_refreshed(self, cat, monkeypatch):
        import os
        import time
        old = time.time() - 7200
        os.utime(cat, (old, old))
        monkeypatch.setattr(catalog, "catalog_path", lambda c: cat)
        calls = []
        monkeypatch.setattr(catalog, "refresh", lambda c, **k: calls.append(c) or {})
        catalog.refresh_if_stale("verda", max_age_s=900)
        assert calls == ["verda"]

    def test_default_age_is_minutes_not_seconds(self):
        """Prices move over hours; availability moves over minutes and is not
        in this file. Per-minute refreshes would be thousands of wasted calls."""
        assert 300 <= catalog.DEFAULT_MAX_AGE_S <= 3600

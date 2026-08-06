"""The reliability ledger.

Where to rent turns on two things marketing never states: whether a launch
succeeds, and how long the machine survives. Both are measurable only by
keeping score, and the snapshot cadence formula takes MTBF as a direct input —
so an unmeasured MTBF means the formula is applied to a guess.
"""
from __future__ import annotations

import json

import pytest

from evsys_sdk.compute import reliability as rel


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    p = tmp_path / "rel.jsonl"
    monkeypatch.setenv(rel.LEDGER_ENV, str(p))
    return p


class TestRecording:
    def test_appends_one_line_per_event(self, ledger):
        rel.record(rel.LAUNCH_OK, provider="verda", gpu="H200", region="FIN-03")
        rel.record(rel.PREEMPTED, provider="verda", gpu="H200", uptime_s=2400)
        assert len(ledger.read_text().strip().splitlines()) == 2

    def test_creates_the_directory(self, monkeypatch, tmp_path):
        monkeypatch.setenv(rel.LEDGER_ENV, str(tmp_path / "deep" / "nested" / "r.jsonl"))
        rel.record(rel.LAUNCH_OK, provider="p", gpu="g")
        assert rel.ledger_path().exists()

    def test_never_raises_when_unwritable(self, monkeypatch):
        """Telemetry that can break a training run is worse than no telemetry."""
        monkeypatch.setenv(rel.LEDGER_ENV, "/proc/cannot/write/here.jsonl")
        rel.record(rel.LAUNCH_OK, provider="p", gpu="g")   # must not raise

    def test_extra_fields_are_kept(self, ledger):
        rel.record(rel.LAUNCH_FAIL, provider="primeintellect", gpu="H100",
                   reason="no_funds", detail="wallet $0.00")
        row = json.loads(ledger.read_text().splitlines()[0])
        assert row["reason"] == "no_funds" and row["detail"] == "wallet $0.00"


class TestReadingIsRobust:
    def test_a_torn_final_line_does_not_lose_the_history(self, ledger):
        """A process can die mid-append; the earlier facts are still facts."""
        rel.record(rel.LAUNCH_OK, provider="verda", gpu="H200")
        with ledger.open("a") as f:
            f.write('{"ts": 1, "event": "pree')      # truncated
        assert len(rel.read()) == 1

    def test_missing_ledger_is_empty_not_an_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv(rel.LEDGER_ENV, str(tmp_path / "absent.jsonl"))
        assert rel.read() == [] and rel.summary() == {}


class TestSummary:
    def test_launch_rate_counts_only_launch_attempts(self, ledger):
        for _ in range(3):
            rel.record(rel.LAUNCH_OK, provider="verda", gpu="H200")
        rel.record(rel.LAUNCH_FAIL, provider="verda", gpu="H200", reason="no_capacity")
        rel.record(rel.PREEMPTED, provider="verda", gpu="H200", uptime_s=60)
        r = rel.summary()[("verda", "H200", 1)]
        assert r["launch_rate"] == pytest.approx(0.75)

    def test_failure_reasons_are_kept_apart(self, ledger):
        """'No capacity' says something about the provider; 'no funds' says
        nothing at all. Averaging them together destroys the signal."""
        rel.record(rel.LAUNCH_FAIL, provider="p", gpu="H100", reason="no_capacity")
        rel.record(rel.LAUNCH_FAIL, provider="p", gpu="H100", reason="no_funds")
        rel.record(rel.LAUNCH_FAIL, provider="p", gpu="H100", reason="no_funds")
        assert rel.summary()[("p", "H100", 1)]["fail_reasons"] == \
            {"no_capacity": 1, "no_funds": 2}

    def test_mtbf_uses_preemptions_only(self, ledger):
        """A deliberate teardown is not a survival — counting it as uptime
        would inflate MTBF and slacken the snapshot cadence."""
        rel.record(rel.PREEMPTED, provider="v", gpu="A100", uptime_s=1200)
        rel.record(rel.PREEMPTED, provider="v", gpu="A100", uptime_s=2400)
        rel.record(rel.TORN_DOWN, provider="v", gpu="A100", uptime_s=99999)
        r = rel.summary()[("v", "A100", 1)]
        assert r["mtbf_s"] == pytest.approx(1800)
        assert r["mtbf_samples"] == 2 and r["torn_down"] == 1

    def test_gpu_count_separates_rows(self, ledger):
        """1xA100 and 2xA100 are different products at different prices."""
        rel.record(rel.LAUNCH_OK, provider="v", gpu="A100", count=1, usd_hr=0.63)
        rel.record(rel.LAUNCH_OK, provider="v", gpu="A100", count=2, usd_hr=1.25)
        assert set(rel.summary()) == {("v", "A100", 1), ("v", "A100", 2)}

    def test_absent_data_is_none_not_zero(self, ledger):
        """A zero MTBF would read as 'preempted instantly' rather than
        'never observed', and would drive the cadence to nonsense."""
        rel.record(rel.LAUNCH_OK, provider="v", gpu="H200")
        r = rel.summary()[("v", "H200", 1)]
        assert r["mtbf_s"] is None and r["mean_usd_hr"] is None

    def test_report_renders_without_data(self, ledger):
        assert "no reliability data" in rel.report()


class TestCadenceFromMeasurement:
    def test_uses_measured_mtbf(self, ledger):
        rel.record(rel.PREEMPTED, provider="v", gpu="A100", uptime_s=7200)
        got = rel.suggested_snapshot_interval_s("v", "A100", snapshot_cost_s=5)
        assert got == pytest.approx((2 * 5 * 7200) ** 0.5)

    def test_falls_back_when_nothing_observed(self, ledger):
        got = rel.suggested_snapshot_interval_s("v", "H200", snapshot_cost_s=5,
                                                default_mtbf_s=3600)
        assert got == pytest.approx((2 * 5 * 3600) ** 0.5)


class TestDeadOnArrival:
    """A machine that provisions, accepts SSH, and cannot run CUDA is billed
    like a success and useless like a failure. Observed on a 2xA100 whose
    nvidia-fabricmanager had aborted on an NVSwitch fault while nvidia-smi
    still listed both GPUs — 20 minutes of rent before it was noticed."""

    def test_doa_counts_against_the_usable_launch_rate(self, ledger):
        rel.record(rel.LAUNCH_OK, provider="v", gpu="A100", count=2)
        rel.record(rel.DOA, provider="v", gpu="A100", count=2, reason="fabricmanager")
        r = rel.summary()[("v", "A100", 2)]
        assert r["doa"] == 1
        assert r["launch_rate"] == pytest.approx(0.5)

    def test_doa_is_not_a_capacity_failure(self, ledger):
        """Capacity shortage says the provider is busy; a broken host says the
        provider shipped faulty hardware. Different remedies."""
        rel.record(rel.DOA, provider="v", gpu="A100", count=2, reason="fabricmanager")
        r = rel.summary()[("v", "A100", 2)]
        assert r["launch_fail"] == 0 and r["fail_reasons"] == {}

    def test_doa_price_is_recorded_because_it_billed(self, ledger):
        rel.record(rel.DOA, provider="v", gpu="A100", count=2, usd_hr=1.253)
        assert rel.summary()[("v", "A100", 2)]["mean_usd_hr"] == pytest.approx(1.253)

    def test_report_flags_doa_visibly(self, ledger):
        rel.record(rel.DOA, provider="v", gpu="A100", count=2)
        assert "doa" in rel.report()

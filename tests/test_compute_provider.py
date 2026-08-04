"""The provider contract.

Every behaviour pinned here cost money to learn: a listing that disagreed with
reality, a machine that reported healthy GPUs and could not run a kernel, and
disks that kept billing after their instance died.
"""
from __future__ import annotations

import pytest

from evsys_sdk.compute import provider as pv
from evsys_sdk.compute import reliability as rel


class FakeProvider(pv.Provider):
    """Scriptable provider. Records calls so tests can assert on ordering."""

    name = "fake"

    def __init__(self, *, avail=True, launch_error=None, states=None,
                 usable=True):
        self.avail, self.launch_error = avail, launch_error
        self.states = list(states or [pv.RUNNING])
        self.usable = usable
        self.launched, self.terminated, self.orphans_released = [], [], []
        self._n = 0

    def offers(self, gpu=None, count=1, spot=None):
        return [pv.Offer("sku-a", "R1", "H200", 1, 1.40, True)]

    def available(self, sku, region, spot=True):
        return self.avail

    def launch(self, sku, region, *, spot=True, name="evsys"):
        if self.launch_error:
            raise pv.LaunchFailed("nope", self.launch_error)
        self.launched.append((sku, region))
        return pv.Machine(id=f"i-{len(self.launched)}", provider=self.name,
                          sku=sku, region=region, gpu="H200", count=1, usd_hr=1.4)

    def poll(self, machine):
        s = self.states[min(self._n, len(self.states) - 1)]
        self._n += 1
        machine.state = s
        if s == pv.RUNNING:
            machine.ip = "10.0.0.1"
        return machine

    def terminate(self, machine):
        self.terminated.append(machine.id)

    def orphans(self):
        return list(self.orphans_released)


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    monkeypatch.setenv(rel.LEDGER_ENV, str(tmp_path / "rel.jsonl"))
    return tmp_path


@pytest.fixture
def nosleep(monkeypatch):
    monkeypatch.setattr(pv.time, "sleep", lambda s: None)


class TestTheGate:
    def test_cuda_probe_is_cuinit_not_nvidia_smi(self):
        """Two hosts listed healthy GPUs through nvidia-smi while cuInit
        returned 802 on a dead NVSwitch fabric. nvidia-smi is not liveness."""
        assert "cuInit" in pv.CUDA_PROBE
        assert "nvidia-smi" not in pv.CUDA_PROBE

    def test_usable_when_probe_says_so(self, monkeypatch, nosleep):
        monkeypatch.setattr(pv, "ssh", lambda *a, **k: (0, "USABLE"))
        assert pv.wait_usable("1.2.3.4", "k") is True

    def test_broken_returns_immediately_not_after_the_timeout(self, monkeypatch, nosleep):
        """A definite BROKEN is an answer; waiting on it wastes rented time."""
        calls = []
        monkeypatch.setattr(pv, "ssh", lambda *a, **k: (calls.append(1), (0, "BROKEN"))[1])
        assert pv.wait_usable("1.2.3.4", "k") is False
        assert len(calls) == 1

    def test_unreachable_keeps_waiting_then_gives_up(self, monkeypatch, nosleep):
        monkeypatch.setattr(pv, "ssh", lambda *a, **k: (255, ""))
        assert pv.wait_usable("1.2.3.4", "k", timeout_s=0) is False


class TestAcquire:
    def test_returns_a_gated_machine(self, ledger, monkeypatch, nosleep):
        monkeypatch.setattr(pv, "ssh", lambda *a, **k: (0, "USABLE"))
        p = FakeProvider()
        m = pv.acquire(p, [("sku-a", "R1")], "k", attempts=1)
        assert m and m.ip == "10.0.0.1"
        assert not p.terminated

    def test_dead_on_arrival_is_released_and_the_search_continues(self, ledger, monkeypatch, nosleep):
        """DOA is common enough that stopping on it wastes the run."""
        monkeypatch.setattr(pv, "ssh", lambda *a, **k: (0, "BROKEN"))
        p = FakeProvider()
        assert pv.acquire(p, [("sku-a", "R1")], "k", attempts=2) is None
        assert len(p.terminated) == 2

    def test_doa_is_recorded_as_its_own_outcome(self, ledger, monkeypatch, nosleep):
        """It bills like a success and delivers like a failure."""
        monkeypatch.setattr(pv, "ssh", lambda *a, **k: (0, "BROKEN"))
        pv.acquire(FakeProvider(), [("sku-a", "R1")], "k", attempts=1)
        assert [e for e in rel.read() if e["event"] == rel.DOA]

    def test_unavailable_sku_is_never_launched(self, ledger, nosleep):
        p = FakeProvider(avail=False)
        pv.acquire(p, [("sku-a", "R1")], "k", attempts=2)
        assert p.launched == []

    def test_launch_failure_reason_is_preserved(self, ledger, nosleep):
        """no_capacity and no_funds are opposite signals; the provider
        flattens them into one error and telling them apart cost hours."""
        pv.acquire(FakeProvider(launch_error="no_funds"), [("sku-a", "R1")],
                   "k", attempts=1)
        fails = [e for e in rel.read() if e["event"] == rel.LAUNCH_FAIL]
        assert fails and fails[0]["reason"] == "no_funds"

    def test_preference_order_is_respected(self, ledger, monkeypatch, nosleep):
        monkeypatch.setattr(pv, "ssh", lambda *a, **k: (0, "USABLE"))
        p = FakeProvider()
        pv.acquire(p, [("cheap", "R1"), ("dear", "R2")], "k", attempts=1)
        assert p.launched[0] == ("cheap", "R1")


class TestSessionAlwaysReleases:
    def test_releases_on_clean_exit(self, ledger, monkeypatch):
        p = FakeProvider()
        m = pv.Machine("i-1", "fake", "sku", "R1", "H200", 1, 1.4, ip="10.0.0.1",
                       state=pv.RUNNING)
        with pv.Session(p, m, "k"):
            pass
        assert p.terminated == ["i-1"]

    def test_releases_even_when_the_body_raises(self, ledger):
        p = FakeProvider()
        m = pv.Machine("i-1", "fake", "sku", "R1", "H200", 1, 1.4, ip="10.0.0.1",
                       state=pv.RUNNING)
        with pytest.raises(ValueError):
            with pv.Session(p, m, "k"):
                raise ValueError("boom")
        assert p.terminated == ["i-1"]

    def test_a_vanished_machine_records_a_preemption_not_a_teardown(self, ledger):
        """Only real preemptions inform MTBF; counting a teardown as one
        inflates it and slackens the snapshot cadence."""
        p = FakeProvider(states=[pv.GONE])
        m = pv.Machine("i-1", "fake", "sku", "R1", "H200", 1, 1.4, ip="10.0.0.1")
        with pv.Session(p, m, "k"):
            pass
        assert [e for e in rel.read() if e["event"] == rel.PREEMPTED]
        assert p.terminated == []          # nothing left to terminate

    def test_teardown_is_recorded_when_we_ended_it(self, ledger):
        p = FakeProvider()
        m = pv.Machine("i-1", "fake", "sku", "R1", "H200", 1, 1.4, ip="10.0.0.1",
                       state=pv.RUNNING)
        with pv.Session(p, m, "k"):
            pass
        assert [e for e in rel.read() if e["event"] == rel.TORN_DOWN]


class TestWaitRunning:
    def test_returns_once_an_ip_appears(self, nosleep):
        p = FakeProvider(states=[pv.PENDING, pv.PENDING, pv.RUNNING])
        m = pv.Machine("i-1", "fake", "sku", "R1", "H200", 1, 1.4)
        assert pv.wait_running(p, m, timeout_s=100).ip == "10.0.0.1"

    def test_raises_if_it_vanishes_first(self, nosleep):
        p = FakeProvider(states=[pv.GONE])
        m = pv.Machine("i-1", "fake", "sku", "R1", "H200", 1, 1.4)
        with pytest.raises(pv.LaunchFailed) as e:
            pv.wait_running(p, m, timeout_s=100)
        assert e.value.reason == "gone"

    def test_raises_on_timeout(self, nosleep):
        p = FakeProvider(states=[pv.PENDING])
        m = pv.Machine("i-1", "fake", "sku", "R1", "H200", 1, 1.4)
        with pytest.raises(pv.LaunchFailed) as e:
            pv.wait_running(p, m, timeout_s=0)
        assert e.value.reason == "timeout"


class TestOffer:
    def test_price_is_comparable_per_gpu(self):
        """An 8-GPU box at $11.20 is cheaper per GPU than a single at $1.50."""
        big = pv.Offer("s", "R", "H200", 8, 11.20, True)
        one = pv.Offer("s", "R", "H200", 1, 1.50, True)
        assert big.usd_per_gpu_hr < one.usd_per_gpu_hr

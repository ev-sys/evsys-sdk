"""Three-state liveness.

Two boxes were preempted mid-benchmark and the watchers kept polling dead IPs
for the better part of an hour, reporting nothing, because an unreachable host
produced no output and no output looked like no news. These tests pin the
distinctions that would have caught it.
"""
from __future__ import annotations

import pytest

from evsys_sdk.compute import liveness as lv
from evsys_sdk.compute import reliability as rel


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    monkeypatch.setenv(rel.LEDGER_ENV, str(tmp_path / "rel.jsonl"))
    return tmp_path / "rel.jsonl"


def make(probe, exists, **kw):
    kw.setdefault("provider", "verda")
    kw.setdefault("gpu", "H200")
    return lv.Liveness(probe=probe, exists=exists, **kw)


class TestThreeStates:
    def test_working_is_alive(self, ledger):
        assert make(lambda: True, lambda: True).check() == lv.ALIVE

    def test_reachable_but_not_working_is_idle_not_gone(self, ledger):
        """A box whose benchmark died still answers. Calling that healthy is
        how a queue starves behind a corpse."""
        assert make(lambda: False, lambda: True).check() == lv.IDLE

    def test_idle_is_never_recorded_as_a_preemption(self, ledger):
        """It would inflate MTBF and slacken the snapshot cadence."""
        lz = make(lambda: False, lambda: True)
        for _ in range(5):
            lz.check()
        assert not [e for e in rel.read() if e["event"] == rel.PREEMPTED]


class TestUnreachableIsNotGone:
    def test_a_single_failed_probe_is_unknown(self, ledger):
        """One dropped SSH is noise. Declaring a preemption on it would tear
        down live work."""
        lz = make(lambda: None, lambda: True)
        assert lz.check() == lv.UNKNOWN

    def test_strikes_are_needed_before_asking_the_provider(self, ledger):
        asked = []
        lz = make(lambda: None, lambda: (asked.append(1), True)[1], strikes=3)
        lz.check(); lz.check()
        assert asked == []
        lz.check()
        assert len(asked) == 1

    def test_provider_says_still_there_stays_unknown(self, ledger):
        """Network trouble, not preemption."""
        lz = make(lambda: None, lambda: True, strikes=1)
        assert lz.check() == lv.UNKNOWN
        assert not [e for e in rel.read() if e["event"] == rel.PREEMPTED]

    def test_provider_unreachable_holds_at_unknown(self, ledger):
        """If we cannot ask, we do not know — and a guess here is expensive
        in both directions."""
        def boom():
            raise OSError("provider api down")
        lz = make(lambda: None, boom, strikes=1)
        assert lz.check() == lv.UNKNOWN

    def test_probe_recovery_resets_the_strike_count(self, ledger):
        seq = iter([None, None, True, None])
        lz = make(lambda: next(seq), lambda: False, strikes=3)
        lz.check(); lz.check()
        assert lz.check() == lv.ALIVE
        assert lz.check() == lv.UNKNOWN      # counter restarted, not at 3


class TestGone:
    def test_unreachable_plus_provider_agrees_is_gone(self, ledger):
        lz = make(lambda: None, lambda: False, strikes=1)
        assert lz.check() == lv.GONE

    def test_gone_records_a_preemption_with_uptime(self, ledger):
        lz = make(lambda: None, lambda: False, strikes=1,
                  region="FIN-02", usd_hr=1.40)
        lz.started_at -= 3600
        lz.check()
        ev = [e for e in rel.read() if e["event"] == rel.PREEMPTED]
        assert len(ev) == 1
        assert ev[0]["region"] == "FIN-02"
        assert 3500 < ev[0]["uptime_s"] < 3700

    def test_preemption_is_recorded_once_not_every_poll(self, ledger):
        """Otherwise one dead host manufactures a dozen MTBF samples."""
        lz = make(lambda: None, lambda: False, strikes=1)
        for _ in range(4):
            lz.check()
        assert len([e for e in rel.read() if e["event"] == rel.PREEMPTED]) == 1


class TestWatch:
    def test_returns_on_gone_and_fires_the_callback(self, ledger, monkeypatch):
        monkeypatch.setattr(lv.time, "sleep", lambda s: None)
        fired = []
        lz = make(lambda: None, lambda: False, strikes=1)
        assert lv.watch(lz, on_gone=lambda: fired.append("gone"), period=0) == lv.GONE
        assert fired == ["gone"]

    def test_returns_on_idle(self, ledger, monkeypatch):
        monkeypatch.setattr(lv.time, "sleep", lambda s: None)
        lz = make(lambda: False, lambda: True)
        assert lv.watch(lz, period=0) == lv.IDLE

    def test_keeps_waiting_while_merely_unknown(self, ledger, monkeypatch):
        """Waiting is the correct response to uncertainty; returning would
        hand the caller a decision it cannot make."""
        monkeypatch.setattr(lv.time, "sleep", lambda s: None)
        lz = make(lambda: None, lambda: True, strikes=99)
        assert lv.watch(lz, period=0, deadline_s=0) == lv.UNKNOWN

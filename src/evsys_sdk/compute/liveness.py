"""Telling "still working" apart from "gone".

Spot instances vanish without warning, and the failure mode that actually costs
you is not the preemption — it is not noticing. A watcher that polls a remote
host and treats an unreachable host as "no news" will sit quietly forever while
the work it was watching no longer exists. Silence reads as progress.

That happened here: two boxes were preempted mid-matrix and the monitors kept
polling a dead IP for the better part of an hour, reporting nothing, because
their probe was ``ssh ... 2>/dev/null || true``.

So liveness is modelled as **three** states, never two:

    ALIVE    the host answered and the work is running
    IDLE     the host answered and the work is NOT running (finished, or hung)
    GONE     the host did not answer, and the provider agrees it is gone

The distinction between IDLE and GONE matters because they call for opposite
responses — IDLE means collect results and queue more, GONE means relaunch. And
the distinction between "probe failed" and GONE matters just as much: a dropped
SSH connection is not a preemption, and treating it as one would tear down a
working run. Only the provider's own instance list is authoritative.

Transitions into GONE are recorded to the reliability ledger automatically,
because a preemption you did not write down is an MTBF sample you cannot use
([[reliability]] feeds the snapshot cadence).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..logger import get_logger
from . import reliability as rel

log = get_logger(__name__)

ALIVE = "alive"
IDLE = "idle"
GONE = "gone"
UNKNOWN = "unknown"
"""The probe itself failed. NOT the same as GONE — a dropped connection is not
a preemption, and acting on one as though it were tears down live work."""

#: Consecutive unreachable probes before the provider is consulted. A single
#: failed SSH is noise; several in a row is a question worth asking.
DEFAULT_STRIKES = 3


@dataclass
class Liveness:
    """Three-state health of one rented host.

    ``probe`` returns True when the *work* is running on the host (not merely
    that the host is up — a box that answers SSH while its benchmark died is
    IDLE, and treating that as healthy is how a queue starves).

    ``exists`` asks the *provider* whether the instance is still allocated. It
    is the only authority on GONE; ``probe`` failing just means we could not
    reach it.
    """

    probe: Callable[[], bool]
    exists: Callable[[], bool]
    provider: str
    gpu: str
    count: int = 1
    region: str | None = None
    usd_hr: float | None = None
    strikes: int = DEFAULT_STRIKES
    started_at: float = field(default_factory=time.time)

    _misses: int = 0
    _recorded_gone: bool = False

    def check(self) -> str:
        """One observation. Cheap enough to call on every poll."""
        try:
            working = self.probe()
        except Exception as e:
            log.debug("[liveness] probe raised: %s", e)
            working = None

        if working:
            self._misses = 0
            return ALIVE

        if working is False:
            # Reachable but not working. Finished or hung — the caller decides,
            # but either way it is NOT a preemption and must not be reported as
            # one, or the MTBF estimate silently inflates.
            self._misses = 0
            return IDLE

        # Unreachable. Do not guess: strike out first, then ask the provider.
        self._misses += 1
        if self._misses < self.strikes:
            return UNKNOWN
        try:
            still_there = self.exists()
        except Exception as e:
            log.info("[liveness] cannot reach the provider either (%s) — "
                     "holding at unknown rather than declaring a preemption", e)
            return UNKNOWN
        if still_there:
            return UNKNOWN
        self._record_gone()
        return GONE

    def _record_gone(self) -> None:
        if self._recorded_gone:
            return
        self._recorded_gone = True
        uptime = time.time() - self.started_at
        log.warning("[liveness] %s/%s x%d in %s is GONE after %.2f h — recording "
                    "the preemption", self.provider, self.gpu, self.count,
                    self.region or "?", uptime / 3600)
        rel.record(rel.PREEMPTED, provider=self.provider, gpu=self.gpu,
                   count=self.count, region=self.region, uptime_s=uptime,
                   usd_hr=self.usd_hr, detail="detected by liveness watcher")

    def uptime_s(self) -> float:
        return time.time() - self.started_at


def watch(liveness: Liveness, on_gone: Callable[[], Any] | None = None,
          on_idle: Callable[[], Any] | None = None,
          period: float = 60.0, deadline_s: float | None = None) -> str:
    """Poll until the host is GONE or IDLE, then hand back control.

    Returns the terminal state so the caller can branch: IDLE usually means
    collect and queue more, GONE means relaunch elsewhere. Never returns while
    the state is merely UNKNOWN — that is the case where waiting is correct.
    """
    # `if deadline_s` would treat 0 as "no deadline" and loop forever — a
    # zero deadline means check once and return, which is a legitimate ask.
    end = (time.time() + deadline_s) if deadline_s is not None else None
    while True:
        state = liveness.check()
        if state == GONE:
            if on_gone:
                on_gone()
            return GONE
        if state == IDLE:
            if on_idle:
                on_idle()
            return IDLE
        if end is not None and time.time() >= end:
            log.info("[liveness] deadline reached with state=%s", state)
            return state
        time.sleep(period)


__all__ = ["ALIVE", "GONE", "IDLE", "Liveness", "UNKNOWN", "watch"]

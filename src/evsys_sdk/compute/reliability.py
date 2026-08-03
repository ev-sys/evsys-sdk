"""A ledger of what actually happened when we rented GPUs.

Provider marketing says nothing useful about the two things that decide where
to run: whether a launch succeeds at all, and how long the machine survives.
Both are measurable, both vary by provider *and* by GPU type *and* by region,
and neither is knowable except by keeping score.

This matters beyond curiosity because the snapshot cadence is
``T* = sqrt(2 * C * MTBF)`` — it takes mean-time-between-preemptions as an
input. Without measured MTBF that formula is applied to a guess. One observed
preemption is a weak estimate; twenty is a policy.

Events are appended as JSONL, one line per fact, never rewritten. Each is
independently meaningful, so a partially-written file is still usable, and
concurrent writers appending single lines do not corrupt each other.

    from evsys_sdk.compute import reliability as rel

    rel.record("launch_ok", provider="verda", gpu="H200", count=1,
               region="FIN-03", usd_hr=1.40, wait_s=96)
    ...
    rel.record("preempted", provider="verda", gpu="H200", count=1,
               region="FIN-03", uptime_s=8130)

    rel.summary()          # per (provider, gpu) success rate, MTBF, $/hr seen
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any, Iterable

from ..logger import get_logger

log = get_logger(__name__)

#: Override for tests or to keep a per-project ledger.
LEDGER_ENV = "EVSYS_RELIABILITY_LEDGER"
DEFAULT_LEDGER = "~/.evsys/reliability.jsonl"

#: The vocabulary. Kept small on purpose — an event type nobody records is
#: worse than useless, because it makes the ledger look more complete than it
#: is.
LAUNCH_OK = "launch_ok"
"""Instance reached a usable state. Record ``wait_s`` from request to ready."""
LAUNCH_FAIL = "launch_fail"
"""Provisioning was refused. Record ``reason`` — 'no_capacity', 'no_funds',
'unsupported', 'timeout'. These are NOT interchangeable: capacity shortage says
something about the provider, an empty wallet says nothing at all."""
PREEMPTED = "preempted"
"""The machine went away without being asked. Record ``uptime_s`` — this is the
only source of MTBF."""
TORN_DOWN = "torn_down"
"""We ended it deliberately. Record ``uptime_s`` so it can censor the MTBF
estimate correctly rather than being mistaken for a survival."""


def ledger_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get(LEDGER_ENV, DEFAULT_LEDGER)).expanduser()


def record(event: str, *, provider: str, gpu: str | None = None,
           count: int = 1, region: str | None = None, **fields: Any) -> dict:
    """Append one observation. Never raises — telemetry must not break a run."""
    row = {"ts": time.time(), "event": event, "provider": provider,
           "gpu": gpu, "count": count, "region": region, **fields}
    try:
        p = ledger_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:  # pragma: no cover - the point is that it is silent
        log.debug("[reliability] could not record %s: %s", event, e)
    return row


def read(path: pathlib.Path | None = None) -> list[dict]:
    """All events, skipping any line that is not parseable.

    A truncated final line is expected — a process can die mid-append — and is
    not a reason to discard the rest of the history.
    """
    p = path or ledger_path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _key(e: dict) -> tuple:
    return (e.get("provider") or "?", e.get("gpu") or "?", e.get("count") or 1)


def summary(events: Iterable[dict] | None = None) -> dict[tuple, dict]:
    """Per ``(provider, gpu, count)``: launch success, MTBF, observed price.

    MTBF here is *mean uptime before preemption*, using only preempted
    instances. Deliberate teardowns are counted separately as censored
    observations: including them would inflate the estimate (we stopped it, it
    did not fail), and ignoring them entirely hides how much of the evidence is
    censored. Both numbers are reported so the caller can judge.
    """
    rows: dict[tuple, dict] = {}
    for e in events if events is not None else read():
        k = _key(e)
        r = rows.setdefault(k, {
            "provider": k[0], "gpu": k[1], "count": k[2],
            "launch_ok": 0, "launch_fail": 0, "fail_reasons": {},
            "preempted": 0, "torn_down": 0,
            "preempted_uptime_s": [], "censored_uptime_s": [],
            "wait_s": [], "usd_hr": [], "regions": set(),
        })
        if e.get("region"):
            r["regions"].add(e["region"])
        ev = e.get("event")
        if ev == LAUNCH_OK:
            r["launch_ok"] += 1
            if isinstance(e.get("wait_s"), (int, float)):
                r["wait_s"].append(e["wait_s"])
            if isinstance(e.get("usd_hr"), (int, float)):
                r["usd_hr"].append(e["usd_hr"])
        elif ev == LAUNCH_FAIL:
            r["launch_fail"] += 1
            why = e.get("reason") or "unknown"
            r["fail_reasons"][why] = r["fail_reasons"].get(why, 0) + 1
        elif ev == PREEMPTED:
            r["preempted"] += 1
            if isinstance(e.get("uptime_s"), (int, float)):
                r["preempted_uptime_s"].append(e["uptime_s"])
        elif ev == TORN_DOWN:
            r["torn_down"] += 1
            if isinstance(e.get("uptime_s"), (int, float)):
                r["censored_uptime_s"].append(e["uptime_s"])

    for r in rows.values():
        attempts = r["launch_ok"] + r["launch_fail"]
        r["launch_rate"] = (r["launch_ok"] / attempts) if attempts else None
        up = r["preempted_uptime_s"]
        r["mtbf_s"] = (sum(up) / len(up)) if up else None
        r["mtbf_samples"] = len(up)
        r["mean_wait_s"] = (sum(r["wait_s"]) / len(r["wait_s"])) if r["wait_s"] else None
        r["mean_usd_hr"] = (sum(r["usd_hr"]) / len(r["usd_hr"])) if r["usd_hr"] else None
        r["regions"] = sorted(r["regions"])
    return rows


def report(events: Iterable[dict] | None = None) -> str:
    """Human-readable table. Absent data reads as '-', never as zero."""
    rows = summary(events)
    if not rows:
        return "no reliability data recorded yet"
    out = [f"{'provider':<16} {'gpu':<12} {'n':>2} {'launch':>8} "
           f"{'MTBF':>10} {'wait':>7} {'$/hr':>7}  regions"]
    for k in sorted(rows):
        r = rows[k]
        lr = f"{r['launch_rate']*100:.0f}%" if r["launch_rate"] is not None else "-"
        lr += f" ({r['launch_ok']}/{r['launch_ok']+r['launch_fail']})"
        mt = (f"{r['mtbf_s']/3600:.1f}h/{r['mtbf_samples']}" if r["mtbf_s"]
              else (f"0/{r['torn_down']}cens" if r["torn_down"] else "-"))
        w = f"{r['mean_wait_s']:.0f}s" if r["mean_wait_s"] is not None else "-"
        pr = f"{r['mean_usd_hr']:.3f}" if r["mean_usd_hr"] is not None else "-"
        out.append(f"{r['provider']:<16} {r['gpu']:<12} {r['count']:>2} {lr:>8} "
                   f"{mt:>10} {w:>7} {pr:>7}  {','.join(r['regions']) or '-'}")
    return "\n".join(out)


def suggested_snapshot_interval_s(provider: str, gpu: str, count: int = 1,
                                  snapshot_cost_s: float = 5.0,
                                  default_mtbf_s: float = 3600.0) -> float:
    """Young/Daly optimum for this exact (provider, gpu) from measured MTBF.

    Falls back to ``default_mtbf_s`` when nothing has been observed yet, and
    says so in the log — an interval derived from one preemption is a guess
    wearing a formula's clothes, and the caller deserves to know which it is.
    """
    r = summary().get((provider, gpu, count))
    mtbf = (r or {}).get("mtbf_s")
    n = (r or {}).get("mtbf_samples") or 0
    if not mtbf:
        log.info("[reliability] no preemptions recorded for %s/%s x%d — using "
                 "default MTBF %.0fs", provider, gpu, count, default_mtbf_s)
        mtbf = default_mtbf_s
    elif n < 5:
        log.info("[reliability] MTBF for %s/%s x%d rests on %d observation(s)",
                 provider, gpu, count, n)
    return (2.0 * snapshot_cost_s * mtbf) ** 0.5


__all__ = ["LAUNCH_FAIL", "LAUNCH_OK", "PREEMPTED", "TORN_DOWN", "ledger_path",
           "read", "record", "report", "suggested_snapshot_interval_s", "summary"]

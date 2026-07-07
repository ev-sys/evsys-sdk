"""Unified ingestion daemon — pull **traces and context together** from one
``system.yaml``.

``evsys pull --watch system.yaml`` (and :func:`watch_all`) start one watcher per
configured source — every ``traces.trace_sources`` AND every
``context.context_sources`` — each on its own thread at its own ``pull_every``.
So the same service keeps agent traces and the surrounding context (a user's
emails, tickets, docs) flowing into the local ``.evsys/`` cache in parallel, for
the autoresearch agent to draw on.

This also lifts the single-source limit of ``trace_sources.run_pull`` (which could
only ``watch`` one source): here N sources of both kinds run concurrently.
"""

from __future__ import annotations

import threading
from typing import Any

from .context_sources.runtime import build_context_sources
from .logger import get_logger
from .trace_sources.base import TraceHook
from .trace_sources.runtime import _parse_duration, build_trace_sources

log = get_logger(__name__)


def _watchers(cfg: Any, *, trace_hook: TraceHook | None) -> list[tuple[str, Any, float]]:
    """(label, source, interval) for every trace + context source in the config."""
    out: list[tuple[str, Any, float]] = []
    traces = getattr(cfg, "traces", None)
    for spec, src in build_trace_sources(getattr(traces, "trace_sources", None) or [], hook=trace_hook):
        out.append((f"trace:{src.name}", src, _parse_duration(getattr(spec, "pull_every", "60s"))))
    context = getattr(cfg, "context", None)
    for spec, src in build_context_sources(getattr(context, "context_sources", None) or []):
        out.append((f"context:{src.name}", src, _parse_duration(getattr(spec, "pull_every", "60s"))))
    return out


def run_all_once(cfg: Any, *, trace_hook: TraceHook | None = None) -> dict[str, int]:
    """One pass over every source (no watch). Returns {label: new-count}."""
    return {label: src.run_once() for label, src, _ in _watchers(cfg, trace_hook=trace_hook)}


def watch_all(cfg: Any, *, trace_hook: TraceHook | None = None, stop: Any = None) -> None:
    """Watch every trace + context source concurrently until interrupted.

    Each source runs its own ``watch(interval, stop)`` on a daemon thread. Blocks
    until ``stop`` is set (or forever); a per-source failure is isolated inside
    that source's own loop and never stops the others."""
    watchers = _watchers(cfg, trace_hook=trace_hook)
    if not watchers:
        log.warning("no trace or context sources configured — nothing to watch")
        return
    stop = stop or threading.Event()
    threads = []
    for label, src, interval in watchers:
        log.info("[%s] watching every %ss", label, interval)
        t = threading.Thread(target=src.watch, args=(interval,), kwargs={"stop": stop},
                              name=label, daemon=True)
        t.start()
        threads.append(t)
    try:
        while not stop.is_set() and any(t.is_alive() for t in threads):
            stop.wait(1.0)
    except KeyboardInterrupt:
        stop.set()


__all__ = ["run_all_once", "watch_all"]

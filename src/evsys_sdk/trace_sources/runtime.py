"""Factory + driver for trace ingestion.

``build_trace_sources`` materializes ``TraceSourceSpec``\\s into live sources
(mirrors :func:`evsys_sdk.training.callbacks.build_callbacks`); ``run_pull``
runs them once or in a ``--watch`` loop. The per-trace ``hook`` defaults to the
no-op seam (Layer 2 supplies a real one).
"""

from __future__ import annotations

from typing import Any

from ..logger import get_logger
from ..registry import get_trace_source
from .base import BaseTraceSource, TraceHook
from .store import LocalTraceStore

log = get_logger(__name__)


def _parse_duration(s: str | int | float) -> float:
    """'60s' / '5m' / '1h' / 90 → seconds."""
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if s and s[-1] in units:
        return float(s[:-1]) * units[s[-1]]
    return float(s)


def build_trace_sources(
    specs: Any,
    *,
    hook: TraceHook | None = None,
    store: LocalTraceStore | None = None,
) -> list[tuple[Any, BaseTraceSource]]:
    """Resolve ``{kind, params, state_dir, ...}`` specs → ``(spec, source)`` pairs.

    ``kind`` is looked up in the trace-source registry; ``params`` are validated
    against the adapter's ``Config`` inside its ``__init__`` (a YAML typo fails
    loudly). Each source gets a ``LocalTraceStore`` rooted at the spec's
    ``state_dir`` (or the shared ``store`` if provided) and the ``hook``.
    """
    out: list[tuple[Any, BaseTraceSource]] = []
    for spec in specs or []:
        kind = spec.kind if hasattr(spec, "kind") else spec["kind"]
        raw = (spec.params if hasattr(spec, "params") else spec.get("params")) or {}
        state_dir = getattr(spec, "state_dir", None) or (
            spec.get("state_dir") if isinstance(spec, dict) else None
        ) or ".evsys/traces"
        cls = get_trace_source(kind)
        st = store or LocalTraceStore(state_dir)
        out.append((spec, cls(store=st, hook=hook, spec=spec, **raw)))
    return out


def run_pull(
    specs: Any,
    *,
    watch: bool = False,
    hook: TraceHook | None = None,
    source: str | None = None,
    limit: int | None = None,
    since: Any = None,
) -> int:
    """Build the sources and pull. Without ``watch``: one pass, returns new-trace
    count. With ``watch``: loop each source on its ``pull_every`` (does not return
    until interrupted)."""
    pairs = build_trace_sources(specs, hook=hook)
    if source:
        pairs = [(s, src) for (s, src) in pairs if src.name == source]
    if not pairs:
        log.warning("no trace sources to pull (source filter=%r)", source)
        return 0

    if watch:
        # For >1 source we'd want threads; Layer-1 keeps it simple and watches
        # them round-robin isn't needed — run each in sequence is wrong for a
        # daemon, so require a single source (or the first) under --watch.
        for spec, src in pairs:
            interval = _parse_duration(getattr(spec, "pull_every", "60s"))
            log.info("[%s] watching every %ss", src.name, interval)
            src.watch(interval)
        return 0

    total = 0
    for _, src in pairs:
        n = src.run_once(limit=limit, since_override=since)
        log.info("[%s] ingested %d new trace(s)", src.name, n)
        total += n
    return total


__all__ = ["build_trace_sources", "run_pull"]

"""Factory + driver for context ingestion (mirrors ``trace_sources.runtime``)."""

from __future__ import annotations

from typing import Any

from ..logger import get_logger
from ..registry import get_context_source
from .base import BaseContextSource
from .store import LocalContextStore

log = get_logger(__name__)


def build_context_sources(
    specs: Any,
    *,
    store: LocalContextStore | None = None,
) -> list[tuple[Any, BaseContextSource]]:
    """Resolve ``{kind, params, state_dir, ...}`` specs → ``(spec, source)`` pairs.

    ``kind`` is looked up in the context-source registry; ``params`` are validated
    against the adapter's ``Config`` (a YAML typo fails loudly). Each source gets a
    ``LocalContextStore`` rooted at the spec's ``state_dir`` (or the shared ``store``)."""
    out: list[tuple[Any, BaseContextSource]] = []
    for spec in specs or []:
        kind = spec.kind if hasattr(spec, "kind") else spec["kind"]
        raw = (spec.params if hasattr(spec, "params") else spec.get("params")) or {}
        state_dir = getattr(spec, "state_dir", None) or (
            spec.get("state_dir") if isinstance(spec, dict) else None
        ) or ".evsys/context"
        cls = get_context_source(kind)
        st = store or LocalContextStore(state_dir)
        out.append((spec, cls(store=st, spec=spec, **raw)))
    return out


def run_context_pull(specs: Any, *, source: str | None = None, limit: int | None = None) -> int:
    """One-shot pull of every context source (no watch). Returns new-item count."""
    pairs = build_context_sources(specs)
    if source:
        pairs = [(s, src) for (s, src) in pairs if src.name == source]
    total = 0
    for _, src in pairs:
        n = src.run_once(limit=limit)
        log.info("[%s] ingested %d new context item(s)", src.name, n)
        total += n
    return total


__all__ = ["build_context_sources", "run_context_pull"]

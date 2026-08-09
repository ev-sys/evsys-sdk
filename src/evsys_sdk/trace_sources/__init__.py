"""Trace ingestion — pull agent traces from hosted observability platforms into
a local, canonical :class:`~evsys_sdk.trace_types.Trace` form.

Importing this package registers the built-in adapters (side-effect import
below), so ``@register_trace_source`` fires on ``import evsys_sdk``.
"""

from __future__ import annotations

# Side-effect import: registers @register_trace_source("langgraph").
from . import langgraph as _langgraph  # noqa: F401
from .base import BaseTraceSource, TraceHook
from .runtime import build_trace_sources, run_pull
from .store import LocalTraceStore

__all__ = [
    "BaseTraceSource",
    "LocalTraceStore",
    "TraceHook",
    "build_trace_sources",
    "run_pull",
]

"""Context ingestion — pull external text context that helps autoresearch write a
better prompt, into a local, canonical
:class:`~evsys_sdk.context_types.ContextItem` form.

The sibling of ``trace_sources``: traces are what the agent did, context is
everything else about the same user/account. Importing this package registers the
built-in adapters (side-effect import below), so ``@register_context_source``
fires on ``import evsys_sdk``.
"""

from __future__ import annotations

# Side-effect import: registers @register_context_source("directory").
from . import directory as _directory  # noqa: F401
from .base import BaseContextSource
from .runtime import build_context_sources, run_context_pull
from .store import LocalContextStore

__all__ = [
    "BaseContextSource",
    "LocalContextStore",
    "build_context_sources",
    "run_context_pull",
]

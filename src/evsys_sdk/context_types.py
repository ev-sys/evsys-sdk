"""Ingested **context** data model.

A :class:`ContextItem` is one piece of external context — any plain text that
might help explain a failure or shape a better prompt — pulled from some source
into a local, provider-agnostic form. It is the sibling of
:class:`~evsys_sdk.trace_types.Trace`: traces are *what the agent did*, context is
*everything else about the same user/account*.

Deliberately minimal and format-agnostic: ``content`` IS the item (plain text),
plus an ``entity`` it's about (so context can be joined to the traces of the same
user/account) and a small ``metadata`` bag. Consumers — the autoresearch agent —
read ``ContextItem`` directly when improving a prompt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContextItem:
    """One ingested piece of context.

    ``content`` — the plain text (the thing autoresearch actually reads).
    ``entity`` — who/what this is about: a user id / account — the join key back to
    the agent traces of the same subject (``None`` if global / uncategorized).
    ``metadata`` — freeform: ``source``, ``timestamp``, ``path``, tags…
    """

    item_id: str
    source: str
    content: str
    entity: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["ContextItem"]

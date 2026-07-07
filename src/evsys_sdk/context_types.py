"""Ingested **context** data model.

A :class:`ContextItem` is one piece of external context — an email, a support
ticket, a doc, a CRM note — pulled from some system into a local,
provider-agnostic form. It is the sibling of
:class:`~evsys_sdk.trace_types.Trace`: traces are *what the agent did*, context is
*everything else that might explain a failure or shape a better prompt* (the
user's recent emails, the account's plan, the ticket that prompted the request).

Deliberately minimal and text-centric: ``content`` IS the item (the body text),
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

    ``kind`` — what it is: ``email`` / ``ticket`` / ``doc`` / ``note`` / …
    ``content`` — the text body (the thing autoresearch actually reads).
    ``entity`` — who/what this is about: a user id, email address, or account —
    the join key back to the agent traces of the same subject (``None`` if global).
    ``metadata`` — freeform: ``source``, ``subject``, ``timestamp``, ``url``, tags…
    """

    item_id: str
    source: str
    kind: str
    content: str
    entity: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["ContextItem"]

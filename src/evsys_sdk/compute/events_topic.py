"""The agent -> router event channel, over any ntfy-style HTTP topic.

The router's ``events`` input is just a callable returning dicts; this module
is the production implementation. Nodes ``post_event`` JSON bodies to a topic
URL; the router polls the same topic and replays every event it has not seen.
The channel carries only two event kinds (``checkpoint`` and ``done``) and
both are idempotent at the consumer — the map keeps latest-per-id and DONE is
DONE — so at-least-once delivery is enough, which is exactly what a dumb HTTP
topic provides.

Transport is an injected callable so tests never open a socket, the same seam
every effectful module in this package uses.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Callable

from ..logger import get_logger

log = get_logger(__name__)

Transport = Callable[..., bytes]


def _http(url: str, data: bytes | None = None) -> bytes:
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def post_event(url: str, event: dict, transport: Transport | None = None) -> bool:
    """Agent side: publish one event. Returns False on failure rather than
    raising — a node must never die because telemetry hiccupped; the next
    checkpoint event supersedes a lost one anyway."""
    try:
        (transport or _http)(url, json.dumps(event).encode())
        return True
    except Exception as e:                                  # noqa: BLE001
        log.warning("[events] post failed: %s", e)
        return False


class TopicEvents:
    """Router side: poll a topic, yield each event exactly once per process.

    ntfy's poll endpoint replays history; the ``since`` cursor (the last seen
    message timestamp) keeps re-polls incremental. Bodies that are not JSON
    objects are skipped — the topic may carry human chatter too.
    """

    def __init__(self, url: str, transport: Transport | None = None):
        self.url = url.rstrip("/")
        self._transport = transport or _http
        self._since = 0

    def __call__(self) -> list[dict]:
        try:
            raw = self._transport(
                f"{self.url}/json?poll=1&since={self._since or 'all'}")
        except Exception as e:                              # noqa: BLE001
            log.warning("[events] poll failed: %s", e)
            return []
        out: list[dict] = []
        for line in raw.decode(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("event") != "message":
                continue
            self._since = max(self._since, int(msg.get("time", 0)))
            try:
                body = json.loads(msg.get("message", ""))
            except json.JSONDecodeError:
                continue
            if isinstance(body, dict) and "kind" in body:
                out.append(body)
        return out


__all__ = ["TopicEvents", "post_event"]

"""Hosted platform trace source — pull via stored LangSmith credential.

Users connect LangSmith in the dashboard (or ``POST /api/integrations/langsmith/``).
The SDK then pulls through our backend so ``LANGSMITH_API_KEY`` never lives on
the researcher's laptop.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..platform_client import PlatformClient, PlatformClientError
from ..registry import register_trace_source
from ..trace_types import Trace
from .base import BaseTraceSource


class PlatformLangSmithConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str | None = None
    """LangSmith project; falls back to the name stored at connect time."""
    limit: int = 100
    filter: str | None = None


@register_trace_source("platform_langsmith")
class PlatformLangSmithTraceSource(BaseTraceSource):
    name = "platform_langsmith"
    Config = PlatformLangSmithConfig

    def pull_raw(self, since: datetime | None) -> Iterable[Any]:
        cfg = self.cfg
        assert cfg is not None
        client = PlatformClient()
        status = client.langsmith_status()
        if not status.get("connected"):
            raise PlatformClientError(
                "LangSmith is not connected for this project. "
                "Connect in the dashboard or POST /api/integrations/langsmith/."
            )
        since_iso = since.isoformat() if since is not None else None
        data = client.langsmith_pull(
            project_name=cfg.project_name or status.get("project_name"),
            since=since_iso,
            limit=cfg.limit,
            filter_expr=cfg.filter,
        )
        for row in data.get("traces") or []:
            yield row

    def to_trace(self, raw: Any) -> Trace:
        if isinstance(raw, Trace):
            return raw
        return Trace(
            trace_id=str(raw["trace_id"]),
            messages=list(raw.get("messages") or []),
            feedback=list(raw.get("feedback") or []),
            metadata=dict(raw.get("metadata") or {}),
        )


__all__ = ["PlatformLangSmithConfig", "PlatformLangSmithTraceSource"]

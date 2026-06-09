"""IdentityTransform — pass-through, useful for testing."""

from __future__ import annotations

from typing import Any, ClassVar, Iterable

from pydantic import BaseModel, ConfigDict

from ..registry import register_transform


class IdentityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


@register_transform("identity")
class IdentityTransform:
    name: ClassVar[str] = "identity"
    Config: ClassVar[type] = IdentityConfig

    def __call__(self, rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        return iter(rows)

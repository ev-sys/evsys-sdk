"""MockBackend — deterministic, no I/O. For tests.

prepare() returns a stub dict; teardown() is a no-op.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import register_backend


class MockBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fail_on_prepare: bool = False
    """Force prepare() to raise; used in error-path tests."""


@register_backend("mock")
class MockBackend:
    name: ClassVar[str] = "mock"
    Config: ClassVar[type] = MockBackendConfig

    def __init__(self, *, fail_on_prepare: bool = False) -> None:
        self.fail_on_prepare = fail_on_prepare

    def prepare(self, *, model: dict[str, Any], run_dir: str) -> dict[str, Any]:
        if self.fail_on_prepare:
            raise RuntimeError("MockBackend asked to fail")
        return {
            "backend": "mock",
            "model": dict(model),
            "run_dir": run_dir,
        }

    def teardown(self, handles: dict[str, Any]) -> None:
        return None

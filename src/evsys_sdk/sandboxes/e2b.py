"""E2B sandbox provider — the built-in hosted option.

Requires ``E2B_API_KEY`` and the ``remote`` extra (``pip install
evsys-sdk[remote]``). The E2B SDK is imported inside :meth:`E2BSandbox.start`
so neither the import of this module nor ``evsys_sdk`` itself pulls in the
vendor package — the same lazy-import discipline harbor's environment factory
uses for daytona/modal/e2b.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..registry import register_sandbox
from .base import BaseSandbox, OnLine

_TEARDOWN_GRACE_S = 120
"""Head-room added to the sandbox's own TTL so E2B does not reap it out from
under a command that is still inside its ``timeout_s``."""


@register_sandbox("e2b")
class E2BSandbox(BaseSandbox):
    """A fresh E2B sandbox per agent run."""

    class Config(BaseModel):
        model_config = {"extra": "forbid"}

        template: str | None = None
        """E2B template id. None → the provider default image (pair it with the
        remote block's ``setup_cmd`` to install what the agent needs); a
        prebuilt template makes spawns much faster."""
        metadata: dict[str, str] = Field(default_factory=dict)
        """Optional E2B sandbox metadata (shows up in the E2B dashboard)."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self._sbx: Any = None

    def start(self) -> None:
        from e2b import Sandbox  # lazy: the `remote` extra

        kwargs: dict[str, Any] = {
            "envs": self.envs,
            "timeout": int(self.timeout_s) + _TEARDOWN_GRACE_S,
        }
        if self.cfg.metadata:
            kwargs["metadata"] = self.cfg.metadata
        self._sbx = Sandbox(self.cfg.template, **kwargs) if self.cfg.template else Sandbox(**kwargs)

    def write(self, path: str, content: str) -> None:
        self._sbx.files.write(path, content)

    def read(self, path: str) -> str | None:
        try:
            return self._sbx.files.read(path)
        except Exception:
            return None

    def exec(self, cmd: str, *, timeout_s: float, cwd: str | None = None,
             on_line: OnLine | None = None) -> tuple[int, str]:
        cb = (lambda data: on_line(str(data))) if on_line else None
        result = self._sbx.commands.run(
            cmd, envs=self.envs, timeout=int(timeout_s), cwd=cwd,
            on_stdout=cb, on_stderr=cb,
        )
        out = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
        return int(result.exit_code or 0), out

    def kill(self) -> None:
        if self._sbx is None:
            return
        try:
            self._sbx.kill()
        except Exception:  # pragma: no cover - best-effort teardown
            pass
        self._sbx = None


__all__ = ["E2BSandbox"]

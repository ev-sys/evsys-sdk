"""Platform sandbox provider — hosted E2B via our APIs.

Uses the platform ``E2B_API_KEY``; the SDK user only needs ``EVSYS_API_KEY``.
Requires ``pip install evsys-sdk[remote]`` only if falling back to direct E2B —
this provider talks HTTP to ``/api/sandboxes/...`` and never imports ``e2b``.
"""

from __future__ import annotations

import base64
import shlex
import uuid
from typing import Any

from pydantic import BaseModel, Field

from ..platform_client import PlatformClient, PlatformClientError
from ..registry import register_sandbox
from .base import BaseSandbox, OnLine


@register_sandbox("platform")
class PlatformSandbox(BaseSandbox):
    """Ephemeral sandbox on the hosted platform (E2B behind our API)."""

    class Config(BaseModel):
        model_config = {"extra": "forbid"}

        template_id: str = "base"
        """Product template id (``base``, ``gmail-mock``, ``slack-mock``)."""
        name_prefix: str = "agent"
        """Sandbox name prefix; a uuid suffix is appended per run."""
        metadata: dict[str, str] = Field(default_factory=dict)
        """Ignored for MVP — reserved for future platform metadata."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self._client = PlatformClient()
        self._sandbox_id: str | None = None

    def start(self) -> None:
        name = f"{self.cfg.name_prefix}-{uuid.uuid4().hex[:10]}"
        sb = self._client.create_sandbox(
            name=name,
            template_id=self.cfg.template_id,
            auto_start=True,
            timeout_sec=max(int(self.timeout_s) + 120, 600),
        )
        self._sandbox_id = sb.get("id")
        if not self._sandbox_id:
            raise PlatformClientError(f"platform create returned no id: {sb!r}")

    def write(self, path: str, content: str) -> None:
        b64 = base64.b64encode(content.encode()).decode()
        parent = str(__import__("pathlib").Path(path).parent)
        cmd = (
            f"mkdir -p {shlex.quote(parent)} && "
            f"echo {shlex.quote(b64)} | base64 -d > {shlex.quote(path)}"
        )
        self._exec_checked(cmd, timeout_s=30)

    def read(self, path: str) -> str | None:
        cmd = f"test -f {shlex.quote(path)} && base64 -w0 {shlex.quote(path)} || true"
        code, out = self._exec(cmd, timeout_s=30)
        if code != 0 or not out.strip():
            return None
        try:
            return base64.b64decode(out.strip()).decode()
        except Exception:
            return None

    def exec(self, cmd: str, *, timeout_s: float, cwd: str | None = None,
             on_line: OnLine | None = None) -> tuple[int, str]:
        code, out = self._exec(cmd, timeout_s=timeout_s, cwd=cwd)
        if on_line and out:
            for line in out.splitlines(keepends=True):
                on_line(line)
        return code, out

    def kill(self) -> None:
        if not self._sandbox_id:
            return
        try:
            self._client.stop_sandbox(self._sandbox_id, kill=True)
        except Exception:
            pass
        self._sandbox_id = None

    def _exec(self, cmd: str, *, timeout_s: float, cwd: str | None = None) -> tuple[int, str]:
        if not self._sandbox_id:
            raise PlatformClientError("platform sandbox not started")
        result = self._client.exec_sandbox(
            self._sandbox_id,
            cmd,
            cwd=cwd or self.workdir,
            timeout_sec=timeout_s,
        )
        stdout = result.get("stdout") or ""
        stderr = result.get("stderr") or ""
        out = stdout + (("\n" + stderr) if stderr else "")
        return int(result.get("exit_code") or 0), out

    def _exec_checked(self, cmd: str, *, timeout_s: float) -> None:
        code, out = self._exec(cmd, timeout_s=timeout_s)
        if code != 0:
            raise PlatformClientError(f"platform exec failed ({code}): {out[:200]}")


__all__ = ["PlatformSandbox"]

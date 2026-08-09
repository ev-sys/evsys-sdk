"""Local subprocess "sandbox" — the zero-dependency provider.

Runs the agent on this host, but inside a scratch workdir holding nothing but
the staged snapshot. **This is not isolation**: the process has the host's
filesystem, network and credentials. What it does give you is the same
copy-in / copy-out *contract* as a hosted provider — the agent sees only the
staged corpus at relative paths, and only the declared artifacts are copied
back — which makes it the right choice for developing the loop without an
``E2B_API_KEY``, and for tests that exercise the real code path instead of a
fake. Reach for :class:`~evsys_sdk.sandboxes.e2b.E2BSandbox` (or your own
provider) when the point is to contain what the agent can touch.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..logger import get_logger
from ..registry import register_sandbox
from .base import BaseSandbox, OnLine

log = get_logger(__name__)


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the command's whole process group; fall back to the shell alone."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:  # pragma: no cover - already gone
            pass


@register_sandbox("local")
class LocalSandbox(BaseSandbox):
    """A scratch directory + ``subprocess`` on this host."""

    class Config(BaseModel):
        model_config = {"extra": "forbid"}

        workdir: str | None = None
        """Where to stage. None → a fresh ``mkdtemp``, removed on ``kill()``."""
        keep: bool = False
        """Keep the scratch dir after the run (for debugging what the agent saw)."""
        inherit_env: bool = True
        """Start from the host environment and layer ``envs`` on top. False →
        the passthrough vars only, which is closer to how a hosted sandbox
        starts (and a decent smoke test that ``env_passthrough`` is complete)."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self._owns_dir = False

    def start(self) -> None:
        if self.cfg.workdir:
            root = Path(self.cfg.workdir).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
        else:
            root = Path(tempfile.mkdtemp(prefix="evsys-sandbox-"))
            self._owns_dir = True
        self.workdir = str(root)
        log.info("[sandbox:local] workdir %s", self.workdir)

    def _env(self) -> dict[str, str]:
        base = dict(os.environ) if self.cfg.inherit_env else {}
        base.update(self.envs)
        return base

    def write(self, path: str, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def read(self, path: str) -> str | None:
        try:
            return Path(path).read_text()
        except OSError:
            return None

    def exec(self, cmd: str, *, timeout_s: float, cwd: str | None = None,
             on_line: OnLine | None = None) -> tuple[int, str]:
        # Own process group, so a timeout kills the agent's children too (a
        # `claude -p` that spawned a build would otherwise outlive the run).
        proc = subprocess.Popen(
            cmd, shell=True, cwd=cwd or self.workdir, env=self._env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,
        )
        # The deadline has to be a watchdog, not `wait(timeout=)`: streaming the
        # output blocks until EOF, which a hung command never reaches.
        timed_out = threading.Event()

        def _reap() -> None:
            timed_out.set()
            _kill_group(proc)

        watchdog = threading.Timer(timeout_s, _reap)
        watchdog.start()
        lines: list[str] = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line)
                if on_line:
                    on_line(line.rstrip("\n"))
            code = proc.wait()
        finally:
            watchdog.cancel()
        if timed_out.is_set():
            lines.append(f"\n[timed out after {timeout_s}s]")
            return 124, "".join(lines)
        return int(code or 0), "".join(lines)

    def kill(self) -> None:
        if self._owns_dir and not self.cfg.keep:
            shutil.rmtree(self.workdir, ignore_errors=True)
        self._owns_dir = False


__all__ = ["LocalSandbox"]

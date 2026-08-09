"""Modal sandbox provider.

The second hosted option, and the one that shares an account with the
*rollout* side: when `algorithm.params.environment.type` is ``modal``, harbor
runs the student's tool calls in Modal sandboxes, so pointing the escalation
agents at the same place keeps one vendor, one bill, one dashboard.

Auth is Modal's own — a token in ``~/.modal.toml`` (``modal token set``) or
``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET`` — so unlike E2B there is no API key
for this SDK to pass through. The ``modal`` package is imported lazily at
:meth:`ModalSandbox.start` (the ``remote-modal`` extra).

Sandboxes are created under a named Modal App so a stuck run is findable in
the dashboard and killable with ``modal app stop``, and they hold ``sleep
infinity`` open while the agent's commands ``exec`` into them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..logger import get_logger
from ..registry import register_sandbox
from .base import BaseSandbox, OnLine

log = get_logger(__name__)

_TEARDOWN_GRACE_S = 120
"""Head-room over the run's own timeout so Modal does not reap the sandbox
while a command is still inside its ``timeout_s``."""


@register_sandbox("modal")
class ModalSandbox(BaseSandbox):
    """A fresh Modal sandbox per agent run."""

    workdir = "/root/evsys"

    class Config(BaseModel):
        model_config = {"extra": "forbid"}

        app_name: str = "evsys-agents"
        """Modal App the sandboxes are created under (looked up, created if
        missing) — the handle for finding or stopping them in the dashboard."""
        image: str | None = None
        """Docker image to boot, e.g. ``python:3.12-slim``. None → Modal's
        ``debian_slim``. A prebuilt image with the agent's deps already in it
        makes ``setup_cmd`` unnecessary and spawns much faster."""
        image_pip: list[str] = Field(default_factory=list)
        """Packages layered onto the image at build time. Cached by Modal, so
        this is cheaper than reinstalling via ``setup_cmd`` on every spawn."""
        cpu: float | None = None
        """Cores to request. None → Modal's default."""
        memory: int | None = None
        """MiB of RAM to request. None → Modal's default."""
        gpu: str | None = None
        """GPU spec, e.g. ``"A10G"``. Agents rarely need one — the model calls
        go out to a hosted API, not to this box."""
        block_network: bool = False
        """Cut outbound network. The agents talk to the Anthropic API, so this
        must stay False unless the mission is fully offline."""
        region: str | None = None
        """Pin a region (e.g. ``us-east``); None → Modal chooses."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self._sbx: Any = None
        self._app: Any = None

    def _build_image(self, modal: Any) -> Any:
        image = (modal.Image.from_registry(self.cfg.image) if self.cfg.image
                 else modal.Image.debian_slim())
        return image.pip_install(*self.cfg.image_pip) if self.cfg.image_pip else image

    def start(self) -> None:
        import modal  # lazy: the `remote-modal` extra

        self._app = modal.App.lookup(self.cfg.app_name, create_if_missing=True)
        kwargs: dict[str, Any] = {
            "app": self._app,
            "image": self._build_image(modal),
            "timeout": int(self.timeout_s) + _TEARDOWN_GRACE_S,
            "workdir": self.workdir,
            "block_network": self.cfg.block_network,
        }
        # `envs` are the passthrough keys (ANTHROPIC_API_KEY & co). Modal wants
        # them at create time; every exec inherits them.
        if self.envs:
            kwargs["env"] = dict(self.envs)
        for field in ("cpu", "memory", "gpu", "region"):
            value = getattr(self.cfg, field)
            if value is not None:
                kwargs[field] = value
        # `sleep infinity` holds the container open; the agent's work arrives
        # as exec calls, exactly like the E2B provider.
        self._sbx = modal.Sandbox.create("sleep", "infinity", **kwargs)
        self._sbx.filesystem.make_directory(self.workdir, create_parents=True)
        log.info("[sandbox:modal] %s (app=%s)", self._sbx.object_id, self.cfg.app_name)

    def write(self, path: str, content: str) -> None:
        parent = path.rsplit("/", 1)[0]
        if parent:
            self._sbx.filesystem.make_directory(parent, create_parents=True)
        self._sbx.filesystem.write_text(content, path)

    def read(self, path: str) -> str | None:
        try:
            return self._sbx.filesystem.read_text(path)
        except Exception:  # FileNotFoundError, and any transport hiccup
            return None

    def exec(self, cmd: str, *, timeout_s: float, cwd: str | None = None,
             on_line: OnLine | None = None) -> tuple[int, str]:
        proc = self._sbx.exec(
            "bash", "-lc", cmd,
            workdir=cwd or self.workdir, timeout=int(timeout_s), text=True,
        )
        chunks: list[str] = []
        # Modal yields buffered chunks, not lines — split so `on_line` gets one
        # line at a time and the agent-runs log stays readable live.
        for chunk in proc.stdout:
            chunks.append(chunk)
            if on_line:
                for line in chunk.splitlines():
                    on_line(line)
        err = proc.stderr.read() or ""
        if err and on_line:
            for line in err.splitlines():
                on_line(line)
        code = proc.wait()
        out = "".join(chunks) + (("\n" + err) if err else "")
        return int(code or 0), out

    def kill(self) -> None:
        if self._sbx is None:
            return
        try:
            self._sbx.terminate()
        except Exception:  # pragma: no cover - best-effort teardown
            pass
        self._sbx = None


__all__ = ["ModalSandbox"]

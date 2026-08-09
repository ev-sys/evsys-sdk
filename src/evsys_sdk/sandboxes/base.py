"""Base sandbox — the provider contract plus the generic copy-in / copy-out.

A *sandbox* is somewhere other than this host to run an agent. The SDK needs
very little from one, so the contract adapters implement is five methods:

  * ``start()``          — create/boot it (a no-op for providers that boot lazily)
  * ``write(path, text)``— put a file in
  * ``read(path)``       — get a file out (``None`` when it does not exist)
  * ``exec(cmd, ...)``   — run a shell command, streaming output line by line
  * ``kill()``           — tear it down

Everything above that — staging a manifest, running the setup commands,
copying back only the artifacts that actually changed, teardown on the way out
of a ``with`` block — is generic and lives here, exactly once. That split is
the same one :class:`~evsys_sdk.trace_sources.base.BaseTraceSource` makes
(adapters implement ``pull_raw``/``to_trace``, the loop is shared), and the
same one harbor makes for environments (``BaseEnvironment`` declares
``start``/``stop``/``upload_file``/``download_file``/``exec`` and its docker,
modal, daytona, e2b, … subclasses fill them in).

Adapters follow the repo-wide extension convention — two ClassVars (``name``,
``Config``) and a ``@register_sandbox`` decorator — so a provider the SDK has
never heard of is selected from YAML by name::

    # in the user's project
    from evsys_sdk import BaseSandbox, register_sandbox

    @register_sandbox("my_cloud")
    class MyCloudSandbox(BaseSandbox):
        class Config(BaseModel):
            model_config = {"extra": "forbid"}
            region: str = "us-east-1"

        def start(self): ...
        def write(self, path, content): ...
        def read(self, path): ...
        def exec(self, cmd, *, timeout_s, cwd=None, on_line=None): ...
        def kill(self): ...

    # in system.yaml
    trigger:
      agent:
        remote:
          enabled: true
          sandbox: {kind: my_cloud, params: {region: eu-west-1}}
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from ..logger import get_logger

log = get_logger(__name__)

DEFAULT_WORKDIR = "/home/user/evsys"
"""Where the staged snapshot lands inside the sandbox. Providers may override
(``LocalSandbox`` uses a scratch dir on the host instead)."""

OnLine = Callable[[str], None]


class SandboxSetupError(RuntimeError):
    """A required setup command failed — the sandbox is unusable."""


class BaseSandbox:
    """One throwaway execution environment.

    Construct via :func:`evsys_sdk.sandboxes.build_sandbox` (which resolves the
    ``kind`` through the registry and validates ``params`` against this class's
    ``Config``), or directly for tests.

    ``envs`` are injected into every command; ``timeout_s`` is the default wall
    clock for a long-running agent command.
    """

    name: ClassVar[str] = ""
    Config: ClassVar[type | None] = None

    #: Sandbox-side root for the staged snapshot; adapters may override.
    workdir: str = DEFAULT_WORKDIR

    def __init__(self, *, envs: dict[str, str] | None = None,
                 timeout_s: float = 1800.0, **params: Any) -> None:
        self.envs = dict(envs or {})
        self.timeout_s = float(timeout_s)
        self._started = False
        # Validate adapter params against the adapter's Config (loud on a typo).
        self.cfg: Any = self.Config(**params) if self.Config is not None else None

    # -- adapter responsibilities -----------------------------------------

    def start(self) -> None:
        """Create the sandbox. Default: nothing (providers that boot lazily)."""

    def write(self, path: str, content: str) -> None:
        raise NotImplementedError

    def read(self, path: str) -> str | None:
        """File contents, or ``None`` when it does not exist."""
        raise NotImplementedError

    def exec(self, cmd: str, *, timeout_s: float, cwd: str | None = None,
             on_line: OnLine | None = None) -> tuple[int, str]:
        """Run ``cmd``; return ``(exit_code, combined stdout+stderr)``.
        ``on_line`` (when given) receives output incrementally, for live logs."""
        raise NotImplementedError

    def kill(self) -> None:
        """Tear down. Must be best-effort — never raise."""

    # -- generic orchestration --------------------------------------------

    def ensure_started(self) -> None:
        """Start exactly once. Every entry point goes through here rather than
        calling ``start()``: a provider's ``start()`` allocates real, billed
        infrastructure, and calling it twice orphans the first sandbox (the
        handle is overwritten, so nothing ever tears it down)."""
        if self._started:
            return
        self.start()
        self._started = True

    def __enter__(self) -> BaseSandbox:
        self.ensure_started()
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            self.kill()
        except Exception:  # pragma: no cover - best-effort teardown
            log.debug("[sandbox:%s] teardown failed", self.name, exc_info=True)
        finally:
            self._started = False

    def path(self, rel: str) -> str:
        """Sandbox-absolute path for a workdir-relative one."""
        return f"{self.workdir}/{rel}"

    def stage(self, manifest: dict[str, str]) -> None:
        """Copy-IN: write every ``{workdir-relative path: content}`` entry."""
        for rel, content in manifest.items():
            self.write(self.path(rel), content)

    def setup(self, cmd: str | None, *, required: bool = True,
              timeout_s: float = 600.0, on_line: OnLine | None = None,
              label: str = "setup_cmd") -> None:
        """Run a provisioning command in the workdir.

        ``required`` → a non-zero exit raises :class:`SandboxSetupError`;
        otherwise it is only reported through ``on_line`` (best-effort installs
        should not kill a run).
        """
        if not cmd:
            return
        code, out = self.exec(cmd, timeout_s=timeout_s, cwd=self.workdir)
        if code == 0:
            return
        if required:
            raise SandboxSetupError(f"{label} failed ({code}): {out[-500:]}")
        if on_line:
            on_line(f"[{label} failed ({code})] {out[-300:]}")

    def collect(self, pairs: list[tuple[str, Path]],
                baseline: dict[str, str]) -> list[str]:
        """Copy-OUT: land the allowed artifacts on the host, and return the
        workdir-relative paths that landed.

        Only files the agent actually CHANGED versus what was staged come back
        — an untouched file must not round-trip, because host mtimes drive UI
        signals like the prompt's "rewritten" flag.
        """
        landed: list[str] = []
        for rel, dest in pairs:
            content = self.read(self.path(rel))
            if content is None or content == baseline.get(rel):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
            landed.append(rel)
        return landed

    # -- copy-out of a whole tree ------------------------------------------

    def list_tree(self, rel_dir: str, *, max_files: int = 2000) -> list[str]:
        """Workdir-relative paths of every file under ``rel_dir`` in the sandbox.

        Implemented with ``find`` through :meth:`exec` rather than a per-vendor
        directory API, so it works on every provider — including one a user
        wrote — with no extra method on the contract.
        """
        root = self.path(rel_dir)
        code, out = self.exec(
            f"find {shlex.quote(root)} -type f 2>/dev/null | head -n {int(max_files)}",
            timeout_s=120,
        )
        if code != 0 and not out.strip():
            return []
        prefix = f"{self.workdir}/"
        found = []
        for line in out.splitlines():
            line = line.strip()
            if line.startswith(prefix):
                found.append(line[len(prefix):])
        return sorted(found)

    def collect_tree(self, rel_dir: str, dest_root: Path, *,
                     baseline: dict[str, str] | None = None,
                     max_files: int = 2000,
                     max_bytes: int = 8_000_000) -> list[str]:
        """Copy a whole DIRECTORY back to the host — the results an agent
        produced in there, whose filenames we cannot know in advance.

        The fixed-artifact :meth:`collect` is the right shape for the four
        files the agent may rewrite. It is the wrong shape for "everything the
        agent's experiments wrote": those are dozens of files with generated
        ids. Without this, an agent that runs a training run inside a sandbox
        loses every experiment, metric, eval and rollout when the box dies.

        Bounded on purpose (``max_files`` / ``max_bytes``): a sandbox is not a
        trusted peer, and an agent that fills a disk must not fill ours. What
        is skipped is logged, never silently dropped.
        """
        baseline = baseline or {}
        landed: list[str] = []
        budget = int(max_bytes)
        skipped = 0
        for rel in self.list_tree(rel_dir, max_files=max_files):
            content = self.read(self.path(rel))
            if content is None or content == baseline.get(rel):
                continue
            size = len(content.encode("utf-8", "ignore"))
            if size > budget:
                skipped += 1
                continue
            budget -= size
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
            landed.append(rel)
        if skipped:
            log.warning("[sandbox:%s] %s file(s) under %s skipped: %d-byte budget exhausted",
                        self.name, skipped, rel_dir, max_bytes)
        return landed


__all__ = ["DEFAULT_WORKDIR", "BaseSandbox", "OnLine", "SandboxSetupError"]

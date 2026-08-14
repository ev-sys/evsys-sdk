"""Headless trigger-agent invocation — spawn Claude Code (``claude -p``) on an
escalation event.

When the cheap deterministic gate escalates, this is the seam that hands the
batch to the (heavier) **trigger agent**: a one-shot, headless Claude Code run
pointed at the escalation event. The agent assesses the batch, writes a verdict,
may retune ``policy.json`` (the self-improving gate), and — on a YES — launches
the autoresearch agent (``training-decider``) to actually run an experiment.

The driver spawns this **detached** (fire-and-forget) so a per-trace ingestion
hook is never blocked on a full agent run; the demo / CLI can spawn it in the
foreground (``detach=False``) to watch the run. The actual process launch goes
through :data:`_LAUNCH`, a module-level seam tests monkeypatch so the suite never
shells out to a real ``claude``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

# The mission templates and clause constants moved to the agent extension
# point (`evsys_sdk.agents.base`) with everything else about mission/argv
# construction; re-exported here so existing imports keep working.
from ..agents.base import (
    AUTORESEARCH_OFF,  # noqa: F401  (re-export)
    AUTORESEARCH_ON,  # noqa: F401  (re-export)
    DEFAULT_PROMPT,  # noqa: F401  (re-export)
    DISTILL_PROMPT,  # noqa: F401  (re-export)
    TriggerAgent,
)
from ..logger import get_logger
from ..provenance import AGENT_TRIGGER, trigger_env

log = get_logger(__name__)


def build_command(escalation_path: str | Path, *, agent_cfg: Any, root: str | Path,
                  verdict_path: str | Path) -> list[str]:
    """Build the ``claude -p ...`` argv for one escalation. Pure — no side effects.

    Thin wrapper: mission + argv construction live on
    :class:`~evsys_sdk.agents.base.TriggerAgent`; this keeps the historical
    signature every caller (CLI, driver, remote runner) already uses.
    """
    agent = TriggerAgent.from_config(agent_cfg)
    return agent.build_command(escalation_path, root=root, verdict_path=verdict_path)


def _launch(cmd: list[str], *, cwd: str | Path, log_file: Path, detach: bool,
            env: dict[str, str] | None = None) -> Any:
    """The one real side effect (monkeypatched in tests). Detached → fire-and-forget
    ``Popen`` in its own session; foreground → ``run`` and return the completed process.

    ``env`` carries the trigger provenance, so every experiment the agent
    launches — however deep in whatever script it writes — is stamped with the
    escalation that caused it."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, **(env or {})} if env else None
    if detach:
        f = log_file.open("w")
        return subprocess.Popen(
            # stdin from /dev/null: a headless claude waits ~3s for input that
            # never comes and opens the transcript with a warning. The remote
            # path already closes it; the local one did not.
            cmd, cwd=str(cwd), stdin=subprocess.DEVNULL,
            stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
            env=full_env,
        )
    proc = subprocess.run(
        cmd, cwd=str(cwd), stdin=subprocess.DEVNULL,
        capture_output=True, text=True, check=False, env=full_env,
    )
    log_file.write_text((proc.stdout or "") + (proc.stderr or ""))
    return proc


_LAUNCH = _launch  # seam


def _snapshot_prompt(escalation_path: Path, *, agent_cfg: Any, root: Path, cwd: Path) -> None:
    """Copy the live prompt file to ``<root>/prompt-snapshots/<escalation>.txt`` so
    the UI can diff an autoresearch rewrite against what the agent started from.
    Best-effort: a missing prompt file just means no snapshot."""
    prompt = cwd / (getattr(agent_cfg, "prompt_file", None) or "prompt.txt")
    try:
        text = prompt.read_text()
    except OSError:
        return
    snap = root / "prompt-snapshots" / f"{escalation_path.stem}.txt"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text(text)


def spawn(escalation_path: str | Path, *, agent_cfg: Any, root: str | Path,
          cwd: str | Path | None = None, detach: bool = True) -> Any:
    """Spawn the trigger agent on one escalation event. Returns the Popen (detached)
    or the CompletedProcess (foreground). ``root`` is the trigger state dir."""
    escalation_path = Path(escalation_path)
    root = Path(root)
    verdict_path = root / "verdicts" / f"{escalation_path.stem}.json"
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = root / "agent-runs" / f"{escalation_path.stem}.log"
    _snapshot_prompt(escalation_path, agent_cfg=agent_cfg, root=root,
                     cwd=Path(cwd or root.parent.parent))
    remote_cfg = getattr(agent_cfg, "remote", None)
    if remote_cfg is not None and getattr(remote_cfg, "enabled", False):
        from .remote import spawn_remote

        log.info("[trigger] spawning REMOTE agent on %s (detach=%s)", escalation_path.name, detach)
        return spawn_remote(escalation_path, agent_cfg=agent_cfg, root=root,
                            cwd=Path(cwd or root.parent.parent),
                            verdict_path=verdict_path, log_file=log_file, detach=detach)
    cmd = build_command(escalation_path, agent_cfg=agent_cfg, root=root, verdict_path=verdict_path)
    log.info("[trigger] spawning agent on %s (detach=%s)", escalation_path.name, detach)
    return _LAUNCH(cmd, cwd=(cwd or root.parent.parent), log_file=log_file, detach=detach,
                   env=trigger_env(escalation_path, agent=AGENT_TRIGGER,
                                   agent_run=escalation_path.stem))


__all__ = ["build_command", "spawn"]

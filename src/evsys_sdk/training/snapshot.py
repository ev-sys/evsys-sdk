"""Codebase snapshots for same-environment rollouts.

"Same environment" means the student's tool calls execute against the repo
state the ingested traces came from. ``capture_codebase_snapshot`` tars the
repo (committed state via ``git archive`` — .gitignore honored by
construction; plain tar fallback for non-git dirs), and
``write_environment_dir`` turns a harbor task dir into a buildable context:
``environment/Dockerfile`` + the snapshot tarball — the standard harbor task
layout, so it works with ANY harbor environment that builds from a task
Dockerfile. Verified live with both:

  * ``environment: {type: docker}`` — local daemon, no account needed
  * ``environment: {type: modal}``  — hosted (Modal content-hashes identical
    contexts to ONE image build no matter how many tasks share them)
"""

from __future__ import annotations

import os
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

from ..logger import get_logger

log = get_logger(__name__)

_TAR_NAME = "snapshot.tar.gz"
_EXCLUDES = {".git", ".venv", "node_modules", "__pycache__", ".evsys", "outputs"}


@dataclass(frozen=True)
class SnapshotInfo:
    tarball: Path
    repo_dir: str
    commit: str | None
    branch: str | None


def _git(repo_dir: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def capture_codebase_snapshot(
    repo_dir: str | Path,
    out_dir: str | Path,
    *,
    ref: str | None = None,
) -> SnapshotInfo:
    """Snapshot ``repo_dir`` into ``out_dir/snapshot.tar.gz``.

    Git repo → ``git archive <ref or HEAD>`` (committed state only). Anything
    else → a filtered tar of the working dir (skips vcs/venv/cache dirs).
    """
    repo_dir = Path(repo_dir).expanduser().resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tarball = out_dir / _TAR_NAME

    commit = _git(repo_dir, "rev-parse", "HEAD")
    if commit is not None:
        target = ref or "HEAD"
        subprocess.run(
            ["git", "-C", str(repo_dir), "archive", "--format=tar.gz",
             "-o", str(tarball), target],
            check=True, capture_output=True,
        )
        branch = ref or _git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD")
        resolved = _git(repo_dir, "rev-parse", target) or commit
        return SnapshotInfo(tarball=tarball, repo_dir=str(repo_dir), commit=resolved, branch=branch)

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        return None if any(p in _EXCLUDES for p in parts) else info

    with tarfile.open(tarball, "w:gz") as tf:
        for entry in sorted(repo_dir.iterdir()):
            tf.add(entry, arcname=entry.name, filter=_filter)
    log.info("[snapshot] non-git dir %s tarred (dirty working state)", repo_dir)
    return SnapshotInfo(tarball=tarball, repo_dir=str(repo_dir), commit=None, branch=None)


def write_environment_dir(
    task_dir: str | Path,
    snapshot: SnapshotInfo,
    *,
    base_image: str = "python:3.12-slim",
    setup_cmd: str | None = None,
    workdir: str = "/workspace",
) -> Path:
    """Write ``<task_dir>/environment/{Dockerfile,snapshot.tar.gz}`` so harbor's
    Modal (or docker) environment builds the repo image for this task."""
    env_dir = Path(task_dir) / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    dest = env_dir / _TAR_NAME
    if not dest.exists():
        try:
            os.link(snapshot.tarball, dest)  # hardlink: no copy per task
        except OSError:
            dest.write_bytes(snapshot.tarball.read_bytes())
    # COPY + explicit tar, not `ADD tar` — Modal's image builder rejects ADD
    # for local archives (http URLs only); COPY+RUN works on docker AND modal.
    lines = [
        f"FROM {base_image}",
        f"COPY {_TAR_NAME} /tmp/{_TAR_NAME}",
        f"RUN mkdir -p {workdir} && tar -xzf /tmp/{_TAR_NAME} -C {workdir} && rm /tmp/{_TAR_NAME}",
        f"WORKDIR {workdir}",
    ]
    if setup_cmd:
        lines.append(f"RUN {setup_cmd}")
    (env_dir / "Dockerfile").write_text("\n".join(lines) + "\n")
    return env_dir


def make_env_writer(snapshot_cfg: dict | None, staging_dir: str | Path):
    """``snapshot:`` config block → an ``env_writer(task_dir)`` for
    ``run_harbor_rollouts`` (or None). Captures the snapshot ONCE at build time;
    every task dir then just gets the Dockerfile + a hardlink."""
    if not snapshot_cfg:
        return None
    snap = capture_codebase_snapshot(
        snapshot_cfg["repo_dir"], staging_dir, ref=snapshot_cfg.get("ref"),
    )

    def writer(task_dir: Path) -> None:
        write_environment_dir(
            task_dir, snap,
            base_image=snapshot_cfg.get("base_image", "python:3.12-slim"),
            setup_cmd=snapshot_cfg.get("setup_cmd"),
            workdir=snapshot_cfg.get("workdir", "/workspace"),
        )

    return writer


__all__ = ["SnapshotInfo", "capture_codebase_snapshot", "write_environment_dir", "make_env_writer"]

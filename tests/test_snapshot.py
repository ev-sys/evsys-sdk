"""Codebase snapshots for same-environment rollouts (training/snapshot.py)."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

from evsys_sdk.training.snapshot import (
    capture_codebase_snapshot,
    make_env_writer,
    write_environment_dir,
)


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("print('v1')\n")
    (repo / "ignored.log").write_text("noise")
    (repo / ".gitignore").write_text("*.log\n")
    for cmd in (["init", "-q"], ["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t",
                                               "commit", "-qm", "v1"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, capture_output=True)
    return repo


def _tar_names(tarball: Path) -> set[str]:
    with tarfile.open(tarball) as tf:
        return {m.name.rstrip("/") for m in tf.getmembers() if m.name not in ("", ".")}


class TestCapture:
    def test_git_repo_committed_state_only(self, tmp_path: Path):
        repo = _git_repo(tmp_path)
        (repo / "dirty.py").write_text("uncommitted")  # not in HEAD
        snap = capture_codebase_snapshot(repo, tmp_path / "stage")
        assert snap.commit and snap.branch
        names = _tar_names(snap.tarball)
        assert "app.py" in names
        assert "dirty.py" not in names       # committed state only
        assert "ignored.log" not in names    # .gitignore honored via git archive

    def test_non_git_dir_filtered_tar(self, tmp_path: Path):
        d = tmp_path / "plain"
        (d / "node_modules").mkdir(parents=True)
        (d / "node_modules" / "big.js").write_text("x")
        (d / "main.py").write_text("y")
        snap = capture_codebase_snapshot(d, tmp_path / "stage")
        assert snap.commit is None
        names = _tar_names(snap.tarball)
        assert "main.py" in names and not any("node_modules" in n for n in names)


class TestEnvironmentDir:
    def test_dockerfile_and_hardlinked_tarball(self, tmp_path: Path):
        repo = _git_repo(tmp_path)
        snap = capture_codebase_snapshot(repo, tmp_path / "stage")
        env = write_environment_dir(tmp_path / "task0", snap,
                                    base_image="python:3.12-slim", setup_cmd="pip install -e .")
        df = (env / "Dockerfile").read_text().splitlines()
        assert df[0] == "FROM python:3.12-slim"
        # COPY + tar, never `ADD <archive>` — Modal's builder rejects local-ADD
        assert df[1] == "COPY snapshot.tar.gz /tmp/snapshot.tar.gz"
        assert "tar -xzf /tmp/snapshot.tar.gz -C /workspace" in df[2]
        assert df[3] == "WORKDIR /workspace"
        assert df[4] == "RUN pip install -e ."
        # hardlink, not a copy (same inode) — cheap per-task
        assert (env / "snapshot.tar.gz").stat().st_ino == snap.tarball.stat().st_ino

    def test_make_env_writer_snapshot_once(self, tmp_path: Path):
        repo = _git_repo(tmp_path)
        writer = make_env_writer(
            {"repo_dir": str(repo), "base_image": "python:3.11"}, tmp_path / "stage",
        )
        assert writer is not None
        writer(tmp_path / "t1")
        writer(tmp_path / "t2")
        for t in ("t1", "t2"):
            assert (tmp_path / t / "environment" / "Dockerfile").read_text().startswith(
                "FROM python:3.11")
        assert make_env_writer(None, tmp_path) is None

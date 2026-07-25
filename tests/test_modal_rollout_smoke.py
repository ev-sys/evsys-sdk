"""REAL Modal same-environment rollout smoke — snapshot a git repo, let harbor
build its image on Modal and run a sandbox, and score with SandboxTestVerifier
executing INSIDE that environment. No LLM involved (EchoAgent): this isolates
exactly the codebase-upload → image-build → sandbox-exec → in-env-verify path.

Gated: needs the ``modal`` package and Modal credentials (~/.modal.toml or
MODAL_TOKEN_ID). Skipped everywhere else, like the tinker real-smokes.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("harbor")
pytest.importorskip("tinker")
pytest.importorskip("modal")

if not (Path.home() / ".modal.toml").exists() and not os.environ.get("MODAL_TOKEN_ID"):
    pytest.skip("no Modal credentials", allow_module_level=True)

from evsys_sdk.data_types import HarborTask, InProcessVerifier
from evsys_sdk.training.harbor_engine import run_harbor_rollouts
from evsys_sdk.training.snapshot import make_env_writer

_ECHO = "tests.harbor_echo_agent:EchoAgent"


@pytest.mark.slow
def test_modal_same_env_rollout(tmp_path: Path):
    # A tiny "codebase" whose presence in the sandbox is the assertion.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hello.py").write_text("print('same env')\n")
    for cmd in (["init", "-q"], ["add", "."],
                ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "v1"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, capture_output=True)

    task = HarborTask(
        task_id="modal-smoke", instruction="noop",
        verifier=InProcessVerifier(fn_name="contains", expected="ECHO"),
        metadata={},
    )
    workspace = tmp_path / "ws"
    dockerfile_writer = make_env_writer({"repo_dir": str(repo)}, tmp_path / "stage")

    def env_writer(task_dir: Path) -> None:
        dockerfile_writer(task_dir)
        # the reward IS the in-env check: the snapshot file must exist in the sandbox
        (Path(task_dir) / "sandbox_verifier.json").write_text(
            json.dumps({"test_command": "test -f /workspace/hello.py"}))

    groups = asyncio.run(run_harbor_rollouts(
        [task], outcome_reward=True, model_name="none", model_path=None,
        workspace_dir=workspace, model_client="litellm",
        agent_import_path=_ECHO,
        environment={"type": "modal", "kwargs": {"sandbox_timeout_secs": 600}},
        env_writer=env_writer,
        verifier_import_path="evsys_sdk.training.harbor_coding_agent:SandboxTestVerifier",
        n_concurrent=1, max_retries=0,
    ))
    (group,) = groups
    assert group.trajectories, "no trajectory harvested from the Modal trial"
    assert group.trajectories[0].reward == 1.0

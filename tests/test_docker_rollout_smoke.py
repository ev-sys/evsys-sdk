"""REAL local-Docker same-environment rollout smoke — the no-cloud-account
sibling of test_modal_rollout_smoke.py. Snapshot a git repo, let harbor's
DOCKER environment build its image locally and run the container, and score
with SandboxTestVerifier executing INSIDE that environment. No LLM (EchoAgent).

Gated: needs a reachable local docker daemon. Skipped everywhere else.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("harbor")
pytest.importorskip("tinker")  # harbor_agents imports TinkerLLM at module top

_docker = shutil.which("docker")
if not _docker or subprocess.run(
    [_docker, "info"], capture_output=True,
).returncode != 0:
    pytest.skip("no reachable docker daemon", allow_module_level=True)

from evsys_sdk.data_types import HarborTask, InProcessVerifier
from evsys_sdk.training.harbor_engine import run_harbor_rollouts
from evsys_sdk.training.snapshot import make_env_writer

_ECHO = "tests.harbor_echo_agent:EchoAgent"


@pytest.mark.slow
def test_docker_same_env_rollout(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hello.py").write_text("print('same env')\n")
    for cmd in (["init", "-q"], ["add", "."],
                ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "v1"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, capture_output=True)

    task = HarborTask(
        task_id="docker-smoke", instruction="noop",
        verifier=InProcessVerifier(fn_name="contains", expected="ECHO"),
        metadata={},
    )
    dockerfile_writer = make_env_writer({"repo_dir": str(repo)}, tmp_path / "stage")

    def env_writer(task_dir: Path) -> None:
        dockerfile_writer(task_dir)
        (Path(task_dir) / "sandbox_verifier.json").write_text(
            json.dumps({"test_command": "test -f /workspace/hello.py"}))

    groups = asyncio.run(run_harbor_rollouts(
        [task], outcome_reward=True, model_name="none", model_path=None,
        workspace_dir=tmp_path / "ws", model_client="litellm",
        agent_import_path=_ECHO,
        environment={"type": "docker"},           # ← the only delta vs Modal
        env_writer=env_writer,
        verifier_import_path="evsys_sdk.training.harbor_coding_agent:SandboxTestVerifier",
        n_concurrent=1, max_retries=0,
    ))
    (group,) = groups
    assert group.trajectories, "no trajectory harvested from the docker trial"
    assert group.trajectories[0].reward == 1.0

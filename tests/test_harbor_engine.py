"""Tests for the harbor rollout runner's harbor-free parts.

``materialize_task`` (minimal task dir) and ``_harvest`` (JobResult →
TrajectoryGroups) need no ``harbor`` install; the ``Job.run()`` path is mocked
in the RL composer test.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tinker")
pytest.importorskip("torch")

from evsys_sdk.data_types import E2BVerifier, HarborTask, InProcessVerifier
from evsys_sdk.training import harbor_engine as he


def _task(task_id="t0", instruction="solve it", expected="42"):
    return HarborTask(
        task_id=task_id,
        instruction=instruction,
        verifier=InProcessVerifier(fn_name="exact_match", expected=expected),
        metadata={"tags": ["x"]},
    )


# --- materialize_task ------------------------------------------------------


def test_materialize_writes_minimal_dir_no_dockerfile_no_testsh(tmp_path: Path):
    dest = he.materialize_task(_task(), tmp_path / "task")
    assert (dest / "instruction.md").read_text() == "solve it"
    toml = (dest / "task.toml").read_text()
    assert 'environment_mode = "separate"' in toml      # skips test.sh at load
    assert "harbor_agents:EvsysVerifier" in toml
    assert 'fn_name = "exact_match"' in toml
    # explicitly: no Dockerfile, no test.sh
    assert not (dest / "environment" / "Dockerfile").exists()
    assert not (dest / "tests").exists()


def test_materialize_rejects_non_in_process_verifier(tmp_path: Path):
    t = HarborTask(task_id="t1", instruction="i", verifier=E2BVerifier())
    with pytest.raises(RuntimeError, match="only 'in_process'"):
        he.materialize_task(t, tmp_path / "t1")


# --- harvest ---------------------------------------------------------------


def _trial(trial_name, *, tokens, logprobs, reward):
    rollout = {
        "prompt_token_ids": [[1, 2, 3]],
        "completion_token_ids": [list(tokens)],
        "logprobs": [list(logprobs)],
    }
    return SimpleNamespace(
        trial_name=trial_name,
        agent_result=SimpleNamespace(rollout_details=[rollout]),
        verifier_result=SimpleNamespace(rewards={"reward": reward}),
    )


def test_harvest_maps_trials_to_groups():
    tasks = [_task("t0"), _task("t1")]
    job_result = SimpleNamespace(trial_results=[
        _trial("t0__s0", tokens=[10, 11], logprobs=[-0.1, -0.2], reward=1.0),
        _trial("t1__s0", tokens=[20], logprobs=[-0.3], reward=0.0),
    ])
    groups = he._harvest(job_result, tasks, num_samples=1)
    assert len(groups) == 2
    assert groups[0].tags == ["x"]
    assert groups[0].trajectories[0].reward == 1.0
    assert groups[0].trajectories[0].completion_tokens == [10, 11]
    assert groups[1].trajectories[0].reward == 0.0


def test_harvest_groups_num_samples_per_task():
    tasks = [_task("t0")]
    job_result = SimpleNamespace(trial_results=[
        _trial("t0__s0", tokens=[1], logprobs=[-0.1], reward=1.0),
        _trial("t0__s1", tokens=[2], logprobs=[-0.2], reward=0.0),
    ])
    groups = he._harvest(job_result, tasks, num_samples=2)
    assert len(groups) == 1
    assert len(groups[0].trajectories) == 2          # both samples
    assert {t.reward for t in groups[0].trajectories} == {1.0, 0.0}


def test_harvest_drops_trials_with_no_rollout():
    tasks = [_task("t0")]
    empty = SimpleNamespace(trial_name="t0__s0", agent_result=None, verifier_result=None)
    groups = he._harvest(SimpleNamespace(trial_results=[empty]), tasks, num_samples=1)
    assert groups[0].trajectories == []

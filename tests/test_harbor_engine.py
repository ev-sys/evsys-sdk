"""Tests for the harbor rollout runner's harbor-free parts.

``materialize_task`` (minimal task dir) and ``_harvest`` (JobResult →
TrajectoryGroups) need no ``harbor`` install; the ``Job.run()`` path is mocked
in the RL composer test.
"""

from __future__ import annotations

from datetime import datetime, timedelta
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
    # No harbor verifier in the task — reward is scored in Python (harbor 0.13.2's
    # verifier is container-coupled; we disable it at the job level).
    assert "import_path" not in toml
    assert not (dest / "environment" / "Dockerfile").exists()
    assert not (dest / "tests").exists()


def test_materialize_rejects_non_in_process_verifier(tmp_path: Path):
    t = HarborTask(task_id="t1", instruction="i", verifier=E2BVerifier())
    with pytest.raises(RuntimeError, match="only 'in_process'"):
        he.materialize_task(t, tmp_path / "t1")


# --- harvest ---------------------------------------------------------------


def _trial(task_name, *, tokens, completion=""):
    # Harbor 0.13.2: trials are grouped by task_name (the dir basename), and the
    # completion is read off agent_result.metadata; reward is scored in Python.
    rollout = {
        "prompt_token_ids": [[1, 2, 3]],
        "completion_token_ids": [list(tokens)],
        "logprobs": [[-0.1] * len(tokens)],
    }
    return SimpleNamespace(
        trial_name=f"{task_name}__abc",
        task_name=task_name,
        agent_result=SimpleNamespace(rollout_details=[rollout], metadata={"completion": completion}),
    )


def test_harvest_maps_trials_to_groups_and_scores():
    # exact_match: completion == expected → 1.0, else 0.0
    tasks = [_task("t0", expected="42"), _task("t1", expected="7")]
    job_result = SimpleNamespace(trial_results=[
        _trial("t0", tokens=[10, 11], completion="42"),
        _trial("t1", tokens=[20], completion="nope"),
    ])
    groups = he._harvest(job_result, tasks, score=True)
    assert len(groups) == 2
    assert groups[0].tags == ["x"]
    assert groups[0].trajectories[0].reward == 1.0       # scored in Python
    assert groups[0].trajectories[0].turns[0].completion_tokens == [10, 11]
    assert groups[1].trajectories[0].reward == 0.0


def test_harvest_groups_n_attempts_per_task():
    tasks = [_task("t0", expected="42")]
    job_result = SimpleNamespace(trial_results=[
        _trial("t0", tokens=[1], completion="42"),
        _trial("t0", tokens=[2], completion="nope"),
    ])
    groups = he._harvest(job_result, tasks, score=True)
    assert len(groups) == 1
    assert len(groups[0].trajectories) == 2              # both attempts (samples)
    assert {t.reward for t in groups[0].trajectories} == {1.0, 0.0}


def test_harvest_drops_trials_with_no_rollout():
    tasks = [_task("t0")]
    empty = SimpleNamespace(trial_name="t0__abc", task_name="t0", agent_result=None)
    groups = he._harvest(SimpleNamespace(trial_results=[empty]), tasks, score=True)
    assert groups[0].trajectories == []


# --- usage (cost / tokens / timing) ----------------------------------------


def _phase(seconds):
    start = datetime(2026, 1, 1, 0, 0, 0)
    return SimpleNamespace(started_at=start, finished_at=start + timedelta(seconds=seconds))


def test_trial_usage_reads_harbor_native_cost_tokens_timing():
    # When harbor populates agent_result + agent_execution, surface those verbatim.
    tr = SimpleNamespace(
        trial_name="t0__s0",
        agent_result=SimpleNamespace(
            rollout_details=[{
                "prompt_token_ids": [[1, 2, 3]],
                "completion_token_ids": [[9, 9]],
                "logprobs": [[-0.1, -0.2]],
            }],
            cost_usd=0.0123, n_input_tokens=15, n_output_tokens=2, n_cache_tokens=4,
        ),
        verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
        agent_execution=_phase(2.5),
        started_at=None, finished_at=None,
    )
    traj = he._trial_to_trajectory(tr)
    assert traj is not None
    u = traj.metadata["usage"]
    assert u["cost_usd"] == pytest.approx(0.0123)
    assert u["prompt_tokens"] == 15          # harbor's count, not the token-id length
    assert u["completion_tokens"] == 2
    assert u["cached_tokens"] == 4
    assert u["latency_s"] == pytest.approx(2.5)


def test_trial_usage_backfills_tokens_and_falls_back_to_trial_timing():
    # No token counts on agent_result → count harvested ids; no agent_execution
    # → use the whole-trial span; no cost (on-policy tinker has no API price).
    start = datetime(2026, 1, 1)
    tr = SimpleNamespace(
        trial_name="t0__s0",
        agent_result=SimpleNamespace(
            rollout_details=[{
                "prompt_token_ids": [[1, 2, 3]],
                "completion_token_ids": [[7, 8]],
                "logprobs": [[-0.1, -0.2]],
            }],
            cost_usd=None, n_input_tokens=None, n_output_tokens=None, n_cache_tokens=None,
        ),
        verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
        agent_execution=None,
        started_at=start, finished_at=start + timedelta(seconds=4.0),
    )
    traj = he._trial_to_trajectory(tr)
    assert traj is not None
    u = traj.metadata["usage"]
    assert u["cost_usd"] is None
    assert u["prompt_tokens"] == 3           # counted from prompt_token_ids
    assert u["completion_tokens"] == 2       # counted from completion_token_ids
    assert u["latency_s"] == pytest.approx(4.0)


def test_phase_seconds_none_when_bounds_missing():
    assert he._phase_seconds(None) is None
    assert he._phase_seconds(SimpleNamespace(started_at=datetime(2026, 1, 1), finished_at=None)) is None

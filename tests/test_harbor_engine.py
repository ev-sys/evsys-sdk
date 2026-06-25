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

pytest.importorskip("harbor")  # harbor_engine imports harbor at module top
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


# --- adapters (our data formats → harbor task dir + TaskConfig) -------------


def test_harbor_task_adapter_writes_scored_task_dir(tmp_path: Path):
    import json
    cfgs = he.HarborTaskAdapter([_task(expected="42")]).to_harbor(tmp_path)
    assert len(cfgs) == 1
    dest = Path(cfgs[0].path)
    assert dest.name == "t0"                                  # dir basename == _safe(task_id)
    assert (dest / "instruction.md").read_text() == "solve it"
    # SHARED mode (no environment_mode); our EvsysVerifier is the job-level verifier.
    assert 'environment_mode' not in (dest / "task.toml").read_text()
    # dummy test.sh only satisfies harbor's load check (never executed)
    assert (dest / "tests" / "test.sh").exists()
    # per-task verifier spec the host-side EvsysVerifier reads
    spec = json.loads((dest / "evsys_verifier.json").read_text())
    assert spec == {"fn_name": "exact_match", "expected": "42", "params": {}}


def test_harbor_task_adapter_rejects_non_in_process_verifier(tmp_path: Path):
    adapter = he.HarborTaskAdapter(
        [HarborTask(task_id="t1", instruction="i", verifier=E2BVerifier())]
    )
    with pytest.raises(RuntimeError, match="only 'in_process'"):
        adapter.to_harbor(tmp_path)


def test_prompt_adapter_writes_generation_task_dir(tmp_path: Path):
    # generation-only: instruction + separate-mode task.toml, no verifier spec, no test.sh.
    cfgs = he.PromptAdapter(["write a poem"]).to_harbor(tmp_path)
    assert len(cfgs) == 1
    dest = Path(cfgs[0].path)
    assert dest.name == "gen_0"
    assert (dest / "instruction.md").read_text() == "write a poem"
    assert 'environment_mode = "separate"' in (dest / "task.toml").read_text()
    assert not (dest / "evsys_verifier.json").exists()
    assert not (dest / "tests" / "test.sh").exists()


# --- harvest ---------------------------------------------------------------


def _trial(task_name, *, tokens, reward):
    # Harbor 0.13.2: trials are grouped by task_name (the dir basename); the
    # reward comes from verifier_result (our host-side EvsysVerifier produced it).
    rollout = {
        "prompt_token_ids": [[1, 2, 3]],
        "completion_token_ids": [list(tokens)],
        "logprobs": [[-0.1] * len(tokens)],
    }
    return SimpleNamespace(
        trial_name=f"{task_name}__abc",
        task_name=task_name,
        agent_result=SimpleNamespace(rollout_details=[rollout], metadata={}),
        verifier_result=SimpleNamespace(rewards={"reward": reward}),
    )


def _tc(task_name):
    # Stand-in TaskConfig: _harvest matches trials by the dir basename of .path.
    return SimpleNamespace(path=Path("/tmp/tasks") / task_name)


def test_harvest_maps_trials_to_groups_with_verifier_reward():
    job_result = SimpleNamespace(trial_results=[
        _trial("t0", tokens=[10, 11], reward=1.0),
        _trial("t1", tokens=[20], reward=0.0),
    ])
    groups = he._harvest(job_result, [_tc("t0"), _tc("t1")])
    assert len(groups) == 2
    assert groups[0].trajectories[0].reward == 1.0       # from verifier_result
    assert groups[0].trajectories[0].turns[0].completion_tokens == [10, 11]
    assert groups[1].trajectories[0].reward == 0.0


def test_harvest_groups_n_attempts_per_task():
    job_result = SimpleNamespace(trial_results=[
        _trial("t0", tokens=[1], reward=1.0),
        _trial("t0", tokens=[2], reward=0.0),
    ])
    groups = he._harvest(job_result, [_tc("t0")])
    assert len(groups) == 1
    assert len(groups[0].trajectories) == 2              # both attempts (samples)
    assert {t.reward for t in groups[0].trajectories} == {1.0, 0.0}


def test_harvest_drops_trials_with_no_rollout():
    empty = SimpleNamespace(trial_name="t0__abc", task_name="t0", agent_result=None)
    groups = he._harvest(SimpleNamespace(trial_results=[empty]), [_tc("t0")])
    assert groups[0].trajectories == []


def test_trial_to_trajectory_keeps_api_model_trial_without_tokens():
    # Closed/API models (litellm) return no token ids, but an eval trial still
    # has a verifier reward — harvest must KEEP it (token-less) so it's scored,
    # carrying the reward + usage. (Previously it was dropped → n_tasks=0.)
    tr = SimpleNamespace(
        trial_name="t0__abc", task_name="t0",
        agent_result=SimpleNamespace(
            rollout_details=None, cost_usd=0.01, n_input_tokens=5,
            n_output_tokens=7, n_cache_tokens=0,
        ),
        verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
    )
    traj = he._trial_to_trajectory(tr)
    assert traj is not None
    assert traj.turns == []                              # no token-level turns
    assert traj.reward == 1.0                            # reward preserved
    assert traj.metadata["usage"]["cost_usd"] == 0.01    # usage preserved


def test_trial_to_trajectory_drops_errored_trial():
    tr = SimpleNamespace(
        trial_name="t0__abc", task_name="t0",
        exception_info={"exception_type": "BadRequestError"},
        agent_result=None, verifier_result=None,
    )
    assert he._trial_to_trajectory(tr) is None


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

"""Tests for `evsys_sdk.training.data_processing`.

Pure unit tests over the advantage + assemble functions. No backend, no
async, no tinker session.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tinker")  # optional dep; not installed in base CI
pytest.importorskip("torch")


from evsys_sdk.training.data_processing import (
    assemble_training_data,
    compute_advantages,
    compute_trajectory_metrics,
)
from evsys_sdk.training.trajectory import Trajectory, TrajectoryGroup, Turn


def _traj(prompt_ids, completion, *, reward, logprobs=None):
    """Single-turn trajectory helper."""
    return Trajectory(
        turns=[Turn(
            prompt_tokens=list(prompt_ids),
            completion_tokens=list(completion),
            logprobs=list(logprobs or [-0.1] * len(completion)),
        )],
        reward=float(reward),
    )


# ---------------------------------------------------------------------------
# compute_advantages
# ---------------------------------------------------------------------------


def test_compute_advantages_subtracts_group_mean():
    group = TrajectoryGroup(trajectories=[
        _traj([1], [2], reward=0.6),
        _traj([1], [3], reward=0.4),
    ])
    adv = compute_advantages([group])
    assert adv[0][0] == pytest.approx(0.1)
    assert adv[0][1] == pytest.approx(-0.1)


def test_compute_advantages_single_trajectory_returns_raw_reward():
    group = TrajectoryGroup(trajectories=[_traj([1], [2], reward=0.7)])
    adv = compute_advantages([group])
    assert adv == [[0.7]]


def test_compute_advantages_skip_normalize():
    group = TrajectoryGroup(trajectories=[
        _traj([1], [2], reward=0.6),
        _traj([1], [3], reward=0.4),
    ])
    adv = compute_advantages([group], normalize=False)
    assert adv == [[0.6, 0.4]]


def test_compute_advantages_empty_group_returns_empty_list():
    group = TrajectoryGroup(trajectories=[])
    assert compute_advantages([group]) == [[]]


# ---------------------------------------------------------------------------
# assemble_training_data
# ---------------------------------------------------------------------------


def test_assemble_training_data_flattens_groups():
    groups = [
        TrajectoryGroup(trajectories=[_traj([10, 11], [20, 21], reward=1.0)]),
        TrajectoryGroup(trajectories=[_traj([30, 31], [40, 41], reward=0.0)]),
    ]
    advantages = compute_advantages(groups)
    datums, meta = assemble_training_data(groups, advantages)
    assert len(datums) == 2
    assert len(meta) == 2
    assert meta[0].group_idx == 0
    assert meta[1].group_idx == 1


def test_assemble_training_data_skips_empty_trajectories():
    group = TrajectoryGroup(
        trajectories=[_traj([1, 2], [], reward=0.5)],
    )
    datums, meta = assemble_training_data([group], compute_advantages([group]))
    assert datums == []
    assert meta == []


def test_datum_carries_target_tokens_logprobs_advantages():
    """The Datum's loss_fn_inputs must populate exactly the IS keys tinker accepts."""
    group = TrajectoryGroup(trajectories=[
        _traj([10, 11], [20, 21, 22], reward=0.5, logprobs=[-0.1, -0.2, -0.3]),
    ])
    advantages = compute_advantages([group])
    datums, _ = assemble_training_data([group], advantages)
    assert len(datums) == 1
    d = datums[0]
    # tinker's importance_sampling accepts ONLY these keys — no mask/weights.
    assert set(d.loss_fn_inputs.keys()) == {"target_tokens", "logprobs", "advantages"}
    # full sequence: [10, 11, 20, 21, 22]; model_input drops last → 4 positions
    adv = d.loss_fn_inputs["advantages"].to_torch().tolist()
    assert len(adv) == 4
    # position 0 = prompt → advantage 0 (this is what masks the loss to completion);
    # positions 1-3 = completion → the single-traj advantage (= reward 0.5)
    assert adv[0] == 0.0
    assert adv[1:4] == [0.5, 0.5, 0.5]
    # logprobs follow the same alignment, one entry per completion position
    lp = d.loss_fn_inputs["logprobs"].to_torch().tolist()
    assert lp[1:4] == [pytest.approx(-0.1), pytest.approx(-0.2), pytest.approx(-0.3)]


# ---------------------------------------------------------------------------
# compute_trajectory_metrics
# ---------------------------------------------------------------------------


def test_trajectory_metrics_rolls_up_rewards():
    groups = [
        TrajectoryGroup(trajectories=[
            _traj([1], [2], reward=0.0), _traj([1], [3], reward=1.0)
        ]),
        TrajectoryGroup(trajectories=[
            _traj([1], [4], reward=0.5)
        ]),
    ]
    metrics = compute_trajectory_metrics(groups)
    assert metrics["reward/mean"] == pytest.approx(0.5)
    assert metrics["reward/n_trajectories"] == 3.0
    assert metrics["reward/n_groups"] == 2.0


def test_trajectory_metrics_empty_when_no_trajectories():
    assert compute_trajectory_metrics([TrajectoryGroup(trajectories=[])]) == {}

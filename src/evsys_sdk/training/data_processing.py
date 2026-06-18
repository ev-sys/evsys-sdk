r"""Turn :class:`~evsys_sdk.training.env.TrajectoryGroup`\s into
``list[tinker.Datum]`` ready for IS-loss training.

Port of ``tinker_cookbook.rl.data_processing.compute_advantages`` +
``assemble_training_data``. Both are pure functions — no I/O, no
tinker_cookbook dependency.

The two-step pipeline:

1. :func:`compute_advantages` — group-normalize rewards within each
   :class:`TrajectoryGroup` (subtract the group mean → reduces variance
   without bias).
2. :func:`assemble_training_data` — flatten to a ``list[tinker.Datum]``
   with ``loss_fn_inputs`` = ``target_tokens`` + per-position ``advantages``
   (0 off-completion → masks the loss) + ``logprobs`` from the sampler (the
   "old" logprobs for the IS ratio). These are the only keys tinker's
   ``importance_sampling`` loss accepts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import tinker
import torch

from .trajectory import TrajectoryGroup, Turn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# compute_advantages — group mean-baseline (pure function)
# ---------------------------------------------------------------------------


def compute_advantages(
    trajectory_groups: list[TrajectoryGroup],
    *,
    normalize: bool = True,
) -> list[list[float]]:
    """Return ``advantages[group][traj]`` — per-trajectory scalar advantages.

    ``normalize=True`` subtracts the group mean from each trajectory's reward
    (standard variance-reduction baseline). When a group has only one
    trajectory, advantage = reward.
    """
    out: list[list[float]] = []
    for g in trajectory_groups:
        rewards = g.rewards
        if not rewards:
            out.append([])
            continue
        if normalize and len(rewards) > 1:
            mean = sum(rewards) / len(rewards)
            out.append([r - mean for r in rewards])
        else:
            out.append(list(rewards))
    return out


# ---------------------------------------------------------------------------
# assemble_training_data — flatten to Datums
# ---------------------------------------------------------------------------


@dataclass
class DatumMetadata:
    """Per-Datum routing info kept on the side (not in the Datum itself —
    tinker rejects unknown keys in loss_fn_inputs)."""

    group_idx: int
    """Index into the original ``trajectory_groups`` list — used for SDFT
    teacher-prompt lookup."""


def assemble_training_data(
    trajectory_groups: list[TrajectoryGroup],
    advantages: list[list[float]],
) -> tuple[list[tinker.Datum], list[DatumMetadata]]:
    """Flatten ``(group, trajectory, turn)`` → ``(Datum, DatumMetadata)`` pairs.

    **One Datum per assistant turn** — a single-turn trajectory yields one
    Datum, a multi-turn one yields one per turn, all carrying the same
    trajectory-level advantage. Each Datum carries:

      * ``model_input``: turn prompt + completion[:-1] (left-shifted).
      * ``loss_fn_inputs["target_tokens"]``: the shifted next-token targets.
      * ``loss_fn_inputs["logprobs"]``: sampler's per-position logprobs (zero on
        prompt positions). The "old" logprobs for the IS ratio.
      * ``loss_fn_inputs["advantages"]``: the trajectory's group-normalized
        reward on completion positions; 0 on prompt positions — which is what
        masks the loss to the completion (tinker's ``importance_sampling`` takes
        no mask/weights key, only target_tokens + logprobs + advantages).

    Turns with no completion tokens are dropped silently — IS loss is a no-op.
    """
    datums: list[tinker.Datum] = []
    metas: list[DatumMetadata] = []
    for group_idx, (group, group_advantages) in enumerate(zip(trajectory_groups, advantages)):
        for traj, adv in zip(group.trajectories, group_advantages):
            for turn in traj.turns:
                datum = _turn_to_datum(turn, advantage=adv)
                if datum is None:
                    continue
                datums.append(datum)
                metas.append(DatumMetadata(group_idx=group_idx))
    return datums, metas


def _turn_to_datum(turn: Turn, *, advantage: float) -> tinker.Datum | None:
    if not turn.completion_tokens:
        return None
    prompt_tokens = list(turn.prompt_tokens)
    completion = list(turn.completion_tokens)
    logprobs = list(turn.logprobs)
    # Pad/truncate logprobs to match completion length defensively.
    if len(logprobs) < len(completion):
        logprobs = logprobs + [0.0] * (len(completion) - len(logprobs))
    elif len(logprobs) > len(completion):
        logprobs = logprobs[: len(completion)]

    full_ids = list(prompt_tokens) + completion
    n_positions = len(full_ids) - 1
    targets = full_ids[1:]

    lp_per_pos = [0.0] * n_positions
    adv_per_pos = [0.0] * n_positions
    start = len(prompt_tokens) - 1
    end = start + len(completion)
    j_lp = 0
    for j in range(max(0, start), min(n_positions, end)):
        if j_lp < len(logprobs):
            lp_per_pos[j] = float(logprobs[j_lp])
        adv_per_pos[j] = float(advantage)
        j_lp += 1

    # tinker's importance_sampling loss accepts ONLY target_tokens + logprobs +
    # advantages (no mask/weights). advantages are 0 on prompt positions, so the
    # loss is naturally masked to the completion (0 advantage → 0 gradient).
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full_ids[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(
                torch.tensor(targets, dtype=torch.long)
            ),
            "logprobs": tinker.TensorData.from_torch(
                torch.tensor(lp_per_pos, dtype=torch.float32)
            ),
            "advantages": tinker.TensorData.from_torch(
                torch.tensor(adv_per_pos, dtype=torch.float32)
            ),
        },
    )


# ---------------------------------------------------------------------------
# Group-level reward / advantage metrics (helpful for logging)
# ---------------------------------------------------------------------------


def compute_trajectory_metrics(groups: list[TrajectoryGroup]) -> dict[str, float]:
    """Roll up group-level reward stats into a flat metrics dict."""
    rewards: list[float] = []
    for g in groups:
        rewards.extend(g.rewards)
    if not rewards:
        return {}
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    metrics: dict[str, float] = {
        "reward/mean": float(mean),
        "reward/std": float(var ** 0.5),
        "reward/n_trajectories": float(len(rewards)),
        "reward/n_groups": float(len(groups)),
    }
    return metrics


__all__ = [
    "DatumMetadata",
    "assemble_training_data",
    "compute_advantages",
    "compute_trajectory_metrics",
]

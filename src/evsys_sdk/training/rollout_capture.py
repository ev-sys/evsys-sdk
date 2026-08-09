"""Capped rollout capture — keep the first N rollouts of each kind, locally.

A run can emit tens of thousands of rollouts; almost none of them are read.
What a person actually needs is *a few of each kind*, to answer "what does the
model literally produce here?" So the SDK keeps the **first N per kind, per
run** — N=20 by default — and drops the rest. First-N rather than a sample so
the set is deterministic and reproducible across reruns.

Three kinds, deliberately distinguished (they answer different questions):

  * ``train``      — on-policy rollouts the algorithm generated to learn from
                     (RL/SDFT set ``TrainingBatch.rollouts``). "What is the
                     model doing while it trains?"
  * ``validation`` — periodic in-training evals at a step. "Is it improving?"
  * ``eval``       — the final/benchmark scoring pass. "How good is it now?"

Everything lands in the run's ``predictions.jsonl`` in the local mirror, in
the one row shape ``harbor_eval.eval_predictions`` already produces, so the
local UI reads a single stream and filters on ``kind``.
"""

from __future__ import annotations

from typing import Any

from ..logger import get_logger

log = get_logger(__name__)

KIND_TRAIN = "train"
KIND_VALIDATION = "validation"
KIND_EVAL = "eval"
ROLLOUT_KINDS = (KIND_TRAIN, KIND_VALIDATION, KIND_EVAL)

DEFAULT_ROLLOUT_CAP = 20
"""Rollouts kept per kind per run. 0 disables capture; negative = unlimited."""


class RolloutCapture:
    """Per-run budget keeper: how many rollouts of each kind may still land.

    Callers ask :meth:`remaining` BEFORE building rows, so a 10k-rollout step
    never materialises 10k dicts to throw 9,980 away.
    """

    def __init__(self, cap: int = DEFAULT_ROLLOUT_CAP) -> None:
        self.cap = int(cap)
        self._used: dict[str, int] = dict.fromkeys(ROLLOUT_KINDS, 0)

    @property
    def enabled(self) -> bool:
        return self.cap != 0

    def remaining(self, kind: str) -> int:
        """How many more of ``kind`` may be kept. ``-1`` means unlimited."""
        if self.cap < 0:
            return -1
        return max(0, self.cap - self._used.get(kind, 0))

    def take(self, kind: str, rows: list[dict]) -> list[dict]:
        """Consume budget and return the slice of ``rows`` that may be kept."""
        if not rows or not self.enabled:
            return []
        left = self.remaining(kind)
        if left == 0:
            return []
        kept = rows if left < 0 else rows[:left]
        self._used[kind] = self._used.get(kind, 0) + len(kept)
        if left >= 0 and len(kept) < len(rows):
            log.info("[rollouts] %s cap reached (%d kept, %d dropped this batch)",
                     kind, self.cap, len(rows) - len(kept))
        return kept

    def counts(self) -> dict[str, int]:
        return dict(self._used)


def training_rollout_rows(
    rollouts: list[Any], *, step: int, limit: int = -1,
    items: list[Any] | None = None,
) -> list[dict]:
    """Flatten ``TrainingBatch.rollouts`` (harbor ``TrajectoryGroup``s) into
    prediction rows shaped like :func:`harbor_eval.eval_predictions` output.

    ``limit`` caps how many rows are built at all (``-1`` = no limit); the
    caller passes the capture's remaining budget so nothing extra is built.
    """
    rows: list[dict] = []
    items = list(items or [])
    for group_idx, group in enumerate(rollouts or []):
        # the task this group was sampled from, when the algorithm supplied it
        item = items[group_idx] if group_idx < len(items) else None
        verifier = getattr(item, "verifier", None)
        for sample_idx, traj in enumerate(getattr(group, "trajectories", []) or []):
            if limit >= 0 and len(rows) >= limit:
                return rows
            turns = getattr(traj, "turns", []) or []
            last = turns[-1] if turns else None
            # Turns carry token ids; `text` is populated when the engine decoded it.
            text = next((getattr(t, "text", "") for t in reversed(turns)
                         if getattr(t, "text", "")), "")
            usage = (getattr(traj, "metadata", None) or {}).get("usage") or {}
            rows.append({
                "kind": KIND_TRAIN,
                "eval_id": None,
                "task_id": (getattr(item, "task_id", None)
                            or getattr(group, "task_id", None) or f"group-{group_idx}"),
                "sample_idx": sample_idx,
                "step": step,
                "instruction": getattr(item, "instruction", None),
                "expected": getattr(verifier, "expected", None),
                "reward": getattr(traj, "reward", None),
                "completion": text,
                "completion_token_ids": getattr(last, "completion_tokens", []) if last else [],
                "metadata": {
                    **(getattr(traj, "metadata", None) or {}),
                    "group_idx": group_idx,
                    "latency_s": usage.get("latency_s"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                },
            })
    return rows


__all__ = [
    "DEFAULT_ROLLOUT_CAP",
    "KIND_EVAL",
    "KIND_TRAIN",
    "KIND_VALIDATION",
    "ROLLOUT_KINDS",
    "RolloutCapture",
    "training_rollout_rows",
]

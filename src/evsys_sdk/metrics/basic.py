"""Built-in benchmark metrics — reduce per-task rollout rewards to a scalar.

A benchmark scores each task by running its verifier on ``num_samples`` rollouts,
yielding a list of per-sample rewards per task. A **metric** reduces that
``list[list[float]]`` (one inner list per task, holding that task's sample
rewards) to a single number. Metrics are referenced by **string name** on a
benchmark's ``metrics:`` list and registered with ``@register_metric``; add your
own the same way in a project.

Built-ins:
  * ``mean_reward`` / ``avg`` — macro mean reward (mean over tasks of each task's
    mean sample reward).
  * ``pass_rate`` — micro pass rate (passing samples / total samples, pooled).
  * ``pass@k`` — a task is solved if **any** of its first ``k`` samples passes.
  * ``pass^k`` — a task is solved only if **all** of its first ``k`` samples pass
    (consistency / "pass-hat-k").

The interface is one method::

    def compute(self, task_rewards: Sequence[Sequence[float]]) -> float
"""

from __future__ import annotations

from typing import ClassVar, Sequence

from ..registry import register_metric

# A sample "passes" when its reward clears this threshold.
PASS_THRESHOLD = 1.0


def _passes(reward: float) -> bool:
    return reward >= PASS_THRESHOLD


def _nonempty(task_rewards: Sequence[Sequence[float]]) -> list[Sequence[float]]:
    return [rs for rs in task_rewards if rs]


@register_metric("mean_reward")
class MeanReward:
    """Macro mean reward: mean over tasks of each task's mean sample reward."""

    name: ClassVar[str] = "mean_reward"

    def compute(self, task_rewards: Sequence[Sequence[float]]) -> float:
        tasks = _nonempty(task_rewards)
        if not tasks:
            return 0.0
        return sum(sum(rs) / len(rs) for rs in tasks) / len(tasks)


@register_metric("avg")
class Avg(MeanReward):
    """Alias for ``mean_reward``."""

    name: ClassVar[str] = "avg"


@register_metric("pass_rate")
class PassRate:
    """Micro pass rate: passing samples / total samples across all tasks."""

    name: ClassVar[str] = "pass_rate"

    def compute(self, task_rewards: Sequence[Sequence[float]]) -> float:
        passes = sum(1 for rs in task_rewards for r in rs if _passes(r))
        total = sum(len(rs) for rs in task_rewards)
        return passes / total if total else 0.0


class _PassAtK:
    """pass@k: a task is solved if **any** of its first ``k`` samples passes."""

    k: ClassVar[int]

    def compute(self, task_rewards: Sequence[Sequence[float]]) -> float:
        tasks = _nonempty(task_rewards)
        if not tasks:
            return 0.0
        solved = sum(1 for rs in tasks if any(_passes(r) for r in rs[: self.k]))
        return solved / len(tasks)


class _PassHatK:
    """pass^k: a task is solved only if **all** of its first ``k`` samples pass."""

    k: ClassVar[int]

    def compute(self, task_rewards: Sequence[Sequence[float]]) -> float:
        tasks = _nonempty(task_rewards)
        if not tasks:
            return 0.0
        solved = sum(1 for rs in tasks if all(_passes(r) for r in rs[: self.k]))
        return solved / len(tasks)


@register_metric("pass@1")
class PassAt1(_PassAtK):
    name: ClassVar[str] = "pass@1"
    k: ClassVar[int] = 1


@register_metric("pass@3")
class PassAt3(_PassAtK):
    name: ClassVar[str] = "pass@3"
    k: ClassVar[int] = 3


@register_metric("pass^3")
class PassHat3(_PassHatK):
    name: ClassVar[str] = "pass^3"
    k: ClassVar[int] = 3


__all__ = [
    "MeanReward",
    "Avg",
    "PassRate",
    "PassAt1",
    "PassAt3",
    "PassHat3",
]

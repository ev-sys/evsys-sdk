"""Learning-rate schedules for the Tinker training loop.

Used by :class:`~evsys_sdk.algorithms.base.BaseAlgorithm` / ``TrainingLoop``
to recompute ``AdamParams.learning_rate`` each step. Schedules are pure
functions of ``(step, num_steps, base_lr, ...)`` — no global state.

Kinds (YAML ``algorithm.params.lr_schedule``):

* ``constant`` — flat ``base_lr`` (default; matches historical behavior)
* ``linear_warmup`` — ramp 0→``base_lr`` over ``lr_warmup_steps``, then hold
* ``linear_decay`` — (optional warmup) then linear decay to ``base_lr * lr_min_ratio``
* ``cosine`` — cosine anneal from ``base_lr`` to ``base_lr * lr_min_ratio``
* ``cosine_with_warmup`` — linear warmup, then cosine anneal

Example YAML::

    algorithm:
      kind: sft
      params:
        learning_rate: 1.0e-4
        lr_schedule: cosine_with_warmup
        lr_warmup_steps: 100
        lr_min_ratio: 0.1
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

LrScheduleKind = Literal[
    "constant",
    "linear_warmup",
    "linear_decay",
    "cosine",
    "cosine_with_warmup",
]

LR_SCHEDULE_KINDS: tuple[str, ...] = (
    "constant",
    "linear_warmup",
    "linear_decay",
    "cosine",
    "cosine_with_warmup",
)


def compute_lr(
    *,
    kind: str,
    step: int,
    num_steps: int,
    base_lr: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> float:
    """Return the learning rate for ``step`` (0-based) of ``num_steps``.

    ``min_lr_ratio`` is relative to ``base_lr`` (e.g. ``0.1`` → floor at
    ``0.1 * base_lr``). ``warmup_steps`` is clamped into ``[0, num_steps]``.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0 (got {num_steps})")
    if step < 0:
        raise ValueError(f"step must be >= 0 (got {step})")
    if base_lr < 0:
        raise ValueError(f"base_lr must be >= 0 (got {base_lr})")
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1] (got {min_lr_ratio})")

    warm = max(0, min(int(warmup_steps), num_steps))
    min_lr = base_lr * float(min_lr_ratio)
    kind_n = (kind or "constant").strip().lower()

    if kind_n == "constant":
        return float(base_lr)

    if kind_n == "linear_warmup":
        if warm <= 0:
            return float(base_lr)
        if step < warm:
            return float(base_lr) * float(step + 1) / float(warm)
        return float(base_lr)

    if kind_n == "linear_decay":
        if step < warm:
            if warm <= 0:
                return float(base_lr)
            return float(base_lr) * float(step + 1) / float(warm)
        # Reach min_lr exactly on the final step.
        denom = max(1, num_steps - warm - 1)
        t = float(step - warm) / float(denom)
        t = min(1.0, max(0.0, t))
        return float(base_lr) * (1.0 - t) + float(min_lr) * t

    if kind_n in ("cosine", "cosine_with_warmup"):
        need_warm = kind_n == "cosine_with_warmup"
        w = warm if need_warm else 0
        if need_warm and step < w:
            if w <= 0:
                return float(base_lr)
            return float(base_lr) * float(step + 1) / float(w)
        # Reach min_lr exactly on the final step.
        denom = max(1, num_steps - w - 1)
        t = float(step - w) / float(denom)
        t = min(1.0, max(0.0, t))
        return float(min_lr) + 0.5 * (float(base_lr) - float(min_lr)) * (
            1.0 + math.cos(math.pi * t)
        )

    raise ValueError(
        f"unknown lr_schedule {kind!r}; expected one of {list(LR_SCHEDULE_KINDS)}"
    )


def make_lr_fn(
    *,
    kind: str = "constant",
    base_lr: float,
    num_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> Callable[[int], float]:
    """Build ``lr_fn(step) -> float`` for :class:`TrainingLoop`."""

    def _fn(step: int) -> float:
        return compute_lr(
            kind=kind,
            step=step,
            num_steps=num_steps,
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )

    return _fn


__all__ = [
    "LR_SCHEDULE_KINDS",
    "LrScheduleKind",
    "compute_lr",
    "make_lr_fn",
]

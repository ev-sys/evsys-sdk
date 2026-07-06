"""Benchmark-metric + verifier behavior.

Metrics reduce per-task rollout rewards (``list[list[float]]`` — one inner list
of sample rewards per task) to a scalar. See ``evsys_sdk/metrics/basic.py``.
"""

from __future__ import annotations

import pytest

from evsys_sdk.metrics.basic import (
    Avg,
    MeanReward,
    PassAt1,
    PassAt3,
    PassHat3,
    PassRate,
)
from evsys_sdk.registry import get_metric, list_metrics
from evsys_sdk.verifiers.format_only import FormatOnlyVerifier

# 4 tasks x 3 samples: rewards chosen so each metric lands on a distinct value.
_TASK_REWARDS = [
    [1.0, 1.0, 0.0],  # 2/3 pass; any → solved; all → no
    [1.0, 0.0, 0.0],  # 1/3 pass; any → solved; all → no
    [0.0, 0.0, 0.0],  # none pass
    [1.0, 1.0, 1.0],  # all pass
]


def test_mean_reward_is_macro_mean():
    # per-task means = [2/3, 1/3, 0, 1] → 0.5
    assert MeanReward().compute(_TASK_REWARDS) == pytest.approx(0.5)


def test_avg_is_alias_of_mean_reward():
    assert Avg().compute(_TASK_REWARDS) == pytest.approx(0.5)
    assert Avg().name == "avg"


def test_pass_rate_is_micro():
    # passing samples = 6 of 12
    assert PassRate().compute(_TASK_REWARDS) == pytest.approx(0.5)


def test_pass_at_k_any_of_first_k():
    # tasks with ≥1 pass in first 3 = 3 of 4
    assert PassAt3().compute(_TASK_REWARDS) == pytest.approx(0.75)
    # first sample passes for tasks 0,1,3 = 3 of 4
    assert PassAt1().compute(_TASK_REWARDS) == pytest.approx(0.75)


def test_pass_hat_k_all_of_first_k():
    # only task 3 has all 3 pass = 1 of 4
    assert PassHat3().compute(_TASK_REWARDS) == pytest.approx(0.25)


def test_empty_is_zero():
    for m in (MeanReward(), PassRate(), PassAt3(), PassHat3()):
        assert m.compute([]) == 0.0


def test_fewer_samples_than_k_uses_available():
    # single sample: pass@3 and pass^3 both collapse to that one sample
    one = [[1.0], [0.0]]
    assert PassAt3().compute(one) == pytest.approx(0.5)
    assert PassHat3().compute(one) == pytest.approx(0.5)


def test_registered_by_string_name():
    assert {"avg", "mean_reward", "pass_rate", "pass@1", "pass@3", "pass^3"} <= set(
        list_metrics()
    )
    assert get_metric("pass@3")().compute(_TASK_REWARDS) == pytest.approx(0.75)
    assert get_metric("pass^3")().compute(_TASK_REWARDS) == pytest.approx(0.25)


def test_format_only_verifier():
    v = FormatOnlyVerifier()
    r1 = v.verify(prompt="", completion="<think>t</think>\n<answer>A</answer>", target={})
    assert r1.reward == pytest.approx(1.0)
    r2 = v.verify(prompt="", completion="bad", target={})
    assert r2.reward == 0.0

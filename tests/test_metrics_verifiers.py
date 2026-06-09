"""Metrics + verifiers behavior."""

from __future__ import annotations

import pytest

from evsys_sdk.metrics.basic import (
    ExactMatch,
    MeanReward,
    PassAtK,
    ToolkitMatch,
)
from evsys_sdk.verifiers.format_only import FormatOnlyVerifier


def test_exact_match_basic():
    m = ExactMatch()
    preds = [{"answer": "A"}, {"answer": "B"}, {"answer": "C"}]
    targets = [{"answer": "A"}, {"answer": "X"}, {"answer": "C"}]
    assert m.compute(predictions=preds, targets=targets) == pytest.approx(2 / 3)


def test_exact_match_empty():
    assert ExactMatch().compute(predictions=[], targets=[]) == 0.0


def test_exact_match_length_mismatch_raises():
    with pytest.raises(ValueError):
        ExactMatch().compute(predictions=[{"answer": "A"}], targets=[])


def test_toolkit_match():
    m = ToolkitMatch()
    preds = [{"answer": "SLACK_FOO"}, {"answer": "WRONG_BAR"}]
    targets = [{"toolkit": "SLACK"}, {"toolkit": "SLACK"}]
    assert m.compute(predictions=preds, targets=targets) == 0.5


def test_mean_reward():
    assert MeanReward().compute(
        predictions=[{"reward": 1.0}, {"reward": 0.0}, {"reward": 0.5}],
        targets=[{}, {}, {}],
    ) == pytest.approx(0.5)


def test_pass_at_k():
    m = PassAtK(k=2)
    preds = [
        {"samples": ["A", "B", "C"]},
        {"samples": ["X", "B"]},
        {"samples": ["A"]},
    ]
    targets = [{"answer": "B"}, {"answer": "B"}, {"answer": "A"}]
    # First: B is in first 2 -> pass
    # Second: B is in first 2 -> pass
    # Third: A is in first 2 -> pass
    assert m.compute(predictions=preds, targets=targets) == 1.0
    m1 = PassAtK(k=1)
    # First: A != B -> fail; Second: X != B -> fail; Third: A == A -> pass
    assert m1.compute(predictions=preds, targets=targets) == pytest.approx(1 / 3)








def test_format_only_verifier():
    v = FormatOnlyVerifier()
    r1 = v.verify(prompt="", completion="<think>t</think>\n<answer>A</answer>", target={})
    assert r1.reward == pytest.approx(1.0)
    r2 = v.verify(prompt="", completion="bad", target={})
    assert r2.reward == 0.0

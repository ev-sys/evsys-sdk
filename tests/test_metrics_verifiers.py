"""Metrics + verifiers behavior."""

from __future__ import annotations

import pytest

from trajectory_experiments.metrics.basic import (
    ExactMatch,
    MeanReward,
    PassAtK,
    ToolkitMatch,
)
from trajectory_experiments.verifiers.composio_tool_match import (
    ComposioToolMatchVerifier,
)
from trajectory_experiments.verifiers.format_only import FormatOnlyVerifier


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


def test_composio_verifier_exact_match():
    v = ComposioToolMatchVerifier()
    completion = "<think>x</think>\n<answer>SLACK_FOO</answer>"
    target = {"tool_slug": "SLACK_FOO", "toolkit": "SLACK"}
    r = v.verify(prompt="", completion=completion, target=target)
    # +1.0 exact + 0.05 + 0.05
    assert r.reward == pytest.approx(1.1)
    assert r.info["exact_match"] is True


def test_composio_verifier_toolkit_only():
    v = ComposioToolMatchVerifier()
    completion = "<think>x</think>\n<answer>SLACK_BAR</answer>"
    target = {"tool_slug": "SLACK_FOO", "toolkit": "SLACK"}
    r = v.verify(prompt="", completion=completion, target=target)
    # 0.3 toolkit + 0.05 + 0.05
    assert r.reward == pytest.approx(0.4)
    assert r.info["exact_match"] is False
    assert r.info["toolkit_match"] is True


def test_composio_verifier_no_answer_penalty():
    v = ComposioToolMatchVerifier()
    completion = "totally wrong text"
    target = {"tool_slug": "SLACK_FOO", "toolkit": "SLACK"}
    r = v.verify(prompt="", completion=completion, target=target)
    # No think, no answer => -0.5
    assert r.reward == pytest.approx(-0.5)


def test_format_only_verifier():
    v = FormatOnlyVerifier()
    r1 = v.verify(prompt="", completion="<think>t</think>\n<answer>A</answer>", target={})
    assert r1.reward == pytest.approx(1.0)
    r2 = v.verify(prompt="", completion="bad", target={})
    assert r2.reward == 0.0

"""Protocols are advisory but worth a smoke test."""

from __future__ import annotations

from evsys_sdk.algorithms.mock_sft import MockSFT
from evsys_sdk.backends.mock import MockBackend
from evsys_sdk.metrics.basic import ExactMatch, MeanReward
from evsys_sdk.protocols import (
    Algorithm,
    Backend,
    Metric,
    Verifier,
)
from evsys_sdk.verifiers.format_only import FormatOnlyVerifier


def test_mock_sft_satisfies_algorithm_protocol():
    inst = MockSFT()
    assert isinstance(inst, Algorithm)


def test_mock_backend_satisfies_backend_protocol():
    assert isinstance(MockBackend(), Backend)


def test_format_only_satisfies_verifier_protocol():
    assert isinstance(FormatOnlyVerifier(), Verifier)


def test_metrics_satisfy_metric_protocol():
    assert isinstance(ExactMatch(), Metric)
    assert isinstance(MeanReward(), Metric)

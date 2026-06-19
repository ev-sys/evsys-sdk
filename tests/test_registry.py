"""Registry decorator + lookup behavior."""

from __future__ import annotations

import pytest

from evsys_sdk import (
    get_algorithm,
    list_algorithms,
    list_backends,
    list_metrics,
    list_transforms,
    list_verifiers,
    register_algorithm,
)
from evsys_sdk.registry import _algorithms, schema_for


def test_builtin_algorithms_registered():
    names = list_algorithms()
    assert "mock_sft" in names
    assert "mock_rl" in names


def test_builtin_backends_include_mock():
    assert "mock" in list_backends()


def test_builtin_metrics():
    names = list_metrics()
    for k in ("mean_reward", "avg", "pass_rate", "pass@1", "pass@3", "pass^3"):
        assert k in names


def test_builtin_verifiers():
    names = list_verifiers()
    for k in ("format_only",):
        assert k in names


def test_builtin_transforms():
    names = list_transforms()
    for k in ("identity", "jsonl_to_chat"):
        assert k in names


def test_register_decorator_registers_by_key_without_mutating_class():
    from pydantic import BaseModel, ConfigDict

    class _C(BaseModel):
        model_config = ConfigDict(extra="forbid")

    @register_algorithm("test_dummy_alg")
    class Dummy:
        Config = _C
        def train(self, ctx):
            return None

    try:
        # The registry key (the decorator arg) is the single source of truth — it's
        # exactly the `kind:` used in config. The class itself is NOT mutated.
        assert get_algorithm("test_dummy_alg") is Dummy
        assert not hasattr(Dummy, "name")  # registry injects no `name` attribute
    finally:
        _algorithms.unregister("test_dummy_alg")


def test_duplicate_registration_raises():
    from pydantic import BaseModel, ConfigDict

    class _C(BaseModel):
        model_config = ConfigDict(extra="forbid")

    @register_algorithm("test_dup")
    class A:
        Config = _C
        def train(self, ctx):
            return None

    with pytest.raises(ValueError, match="already registered"):
        @register_algorithm("test_dup")
        class B:
            Config = _C
            def train(self, ctx):
                return None
    _algorithms.unregister("test_dup")


def test_unknown_lookup_raises_with_available():
    with pytest.raises(KeyError, match="Available"):
        get_algorithm("nope_not_here")


def test_schema_for_returns_json_schema():
    s = schema_for("algorithm", "mock_sft")
    assert "properties" in s
    assert "learning_rate" in s["properties"]


def test_schema_for_unknown_kind_raises():
    with pytest.raises(KeyError):
        schema_for("not_a_kind", "x")

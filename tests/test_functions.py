"""The function extension point — registration, the adapters over the existing
trigger / verifier-fn registries (parity with the wrapped fns), the environment
default, and the spec factory. The `trigger` and `verifier_fn` registries are
untouched by design; these tests also pin that."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from evsys_sdk.config import FunctionSpec
from evsys_sdk.functions import (
    EvsysFunction,
    TriggerFunction,
    VerifierFunction,
    build_function,
)
from evsys_sdk.protocols import TriggerDecision
from evsys_sdk.registry import (
    _functions,
    _triggers,
    get_function,
    list_functions,
    register_function,
    register_trigger,
)
from evsys_sdk.verifiers import fns

# 1. Registration -----------------------------------------------------------

def test_adapters_are_registered():
    assert get_function("trigger_fn") is TriggerFunction
    assert get_function("verifier_fn") is VerifierFunction
    assert {"trigger_fn", "verifier_fn"} <= set(list_functions())


def test_project_registered_function():
    @register_function("t_latency")
    class LatencyBudget(EvsysFunction):
        class Config(BaseModel):
            model_config = {"extra": "forbid"}
            max_ms: float = 500.0

        def run(self, ms):
            return ms <= self.cfg.max_ms

    try:
        fn = build_function({"kind": "t_latency", "params": {"max_ms": 100}})
        assert fn(80) is True and fn(200) is False   # __call__ delegates to run
    finally:
        _functions.unregister("t_latency")


def test_bad_function_params_fail_loudly():
    with pytest.raises(ValidationError):
        build_function(FunctionSpec(kind="verifier_fn", params={"fn_nmae": "contains"}))


def test_unknown_function_fails_loudly():
    with pytest.raises(KeyError, match="No function registered under 'nope'"):
        get_function("nope")


# 2. The verifier-fn adapter ------------------------------------------------

def test_verifier_function_matches_registry_fn():
    direct = fns.get("exact_match")
    wrapped = VerifierFunction(fn_name="exact_match")
    for out, exp in [("yes", "yes"), ("yes", "no"), (" YES ", "yes")]:
        assert wrapped.run(out, exp, {}) == direct(out, exp, {})


def test_verifier_function_default_params():
    wrapped = VerifierFunction(fn_name="contains", params={"ignore_case": True})
    assert wrapped.run("The ANSWER is 42", "answer") == 1.0     # constructor params
    assert wrapped.run("The ANSWER is 42", "answer", {}) == 0.0  # per-call override


def test_verifier_function_unknown_name_fails_loudly():
    with pytest.raises(ValueError, match="Unknown verifier fn"):
        VerifierFunction(fn_name="nope")


def test_verifier_fn_registry_untouched():
    """The adapter must not have re-registered anything under new names."""
    assert set(fns.list_fns()) >= {"contains", "exact_match", "regex_match",
                                   "tool_calls_match"}


# 3. The trigger-fn adapter -------------------------------------------------

def test_trigger_function_wraps_registered_gate():
    @register_trigger("t_fn_gate")
    class Gate:
        name = "t_fn_gate"

        class Config(BaseModel):
            model_config = {"extra": "forbid"}
            threshold: float = 0.5

        def __init__(self, **params):
            self.cfg = self.Config(**params)

        def evaluate(self, state):
            return TriggerDecision(True, f"th={self.cfg.threshold}", {}, [])

    try:
        fn = TriggerFunction(kind="t_fn_gate", params={"threshold": 0.9})
        decision = fn.run(state=None)
        assert decision.escalate is True and "th=0.9" in decision.reason
    finally:
        _triggers.unregister("t_fn_gate")


def test_trigger_function_unknown_kind_fails_loudly():
    with pytest.raises(KeyError, match="No trigger registered under"):
        TriggerFunction(kind="nope")


# 4. The environment field --------------------------------------------------

def test_environment_defaults_to_local():
    assert VerifierFunction(fn_name="contains").environment == "local"
    assert EvsysFunction.environment == "local"


def test_environment_override_at_construction():
    fn = VerifierFunction(fn_name="contains", environment="e2b")
    assert fn.environment == "e2b"
    # the class default is untouched
    assert VerifierFunction.environment == "local"


# 5. The spec factory -------------------------------------------------------

def test_build_function_from_spec():
    fn = build_function(FunctionSpec(kind="verifier_fn",
                                     params={"fn_name": "regex_match"}))
    assert fn.run("hello world", r"w.rld", {}) == 1.0


def test_build_function_requires_kind():
    with pytest.raises(ValueError, match="no `kind`"):
        build_function({})

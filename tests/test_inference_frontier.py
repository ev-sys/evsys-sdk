"""Smoke tests for the frontier inference clients.

These tests are intentionally light: they verify the registry wiring, config
construction, error surfaces when the vendor SDK isn't installed, and when
the API key env var is missing. The live-API path is gated by markers that
won't fire without the corresponding env vars.
"""

from __future__ import annotations

import os
import sys
from importlib import import_module
from unittest import mock

import pytest

from evsys_sdk.registry import get_inference, list_inferences

# ---------------------------------------------------------------------------
# Registry wiring — these should always be enumerable, regardless of whether
# the underlying vendor SDK is installed (lazy import inside __init__).
# ---------------------------------------------------------------------------


def test_three_clients_register_when_present():
    inferences = set(list_inferences())
    # At minimum the mock client is always present.
    assert "mock" in inferences
    # Frontier clients register only when their vendor SDK can be imported;
    # in CI we don't know which will be installed, so this is best-effort.
    for name in ("claude", "gemini", "openai"):
        if name in inferences:
            cls = get_inference(name)
            assert hasattr(cls, "generate")


# ---------------------------------------------------------------------------
# Config models — extra fields should be rejected (config has extra="forbid").
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module,config_cls", [
    ("evsys_sdk.inference.claude", "ClaudeInferenceConfig"),
    ("evsys_sdk.inference.gemini", "GeminiInferenceConfig"),
    ("evsys_sdk.inference.openai", "OpenAIInferenceConfig"),
])
def test_config_rejects_extra_fields(module, config_cls):
    try:
        m = import_module(module)
    except ImportError:
        pytest.skip(f"{module} cannot be imported (vendor SDK missing)")
    cfg_cls = getattr(m, config_cls)
    with pytest.raises(Exception):  # pydantic.ValidationError
        cfg_cls(model="m", nonsense_field=True)


# ---------------------------------------------------------------------------
# Missing API key surfaces a clear runtime error (not an obscure SDK error).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module,class_name,env_var,vendor_pkg", [
    ("evsys_sdk.inference.claude", "ClaudeInference", "ANTHROPIC_API_KEY", "anthropic"),
    ("evsys_sdk.inference.openai", "OpenAIInference", "OPENAI_API_KEY",    "openai"),
])
def test_missing_api_key_raises_clearly(module, class_name, env_var, vendor_pkg):
    # Skip when the vendor SDK isn't installed — construction would raise
    # ImportError first, before we ever get to the env-var check.
    try:
        __import__(vendor_pkg)
    except ImportError:
        pytest.skip(f"{vendor_pkg} package not installed")

    m = import_module(module)
    cls = getattr(m, class_name)
    # Clear ALL plausible env vars so neither primary nor fallback works.
    env_to_clear = {env_var: ""}
    with mock.patch.dict(os.environ, env_to_clear, clear=False):
        for k in env_to_clear:
            os.environ.pop(k, None)
        with pytest.raises(RuntimeError, match=env_var):
            cls()


# ---------------------------------------------------------------------------
# When the vendor SDK is genuinely missing, the *module* should import as a
# no-op (the symbol just isn't registered). Construct without the SDK
# installed → ImportError with a helpful message.
# ---------------------------------------------------------------------------


def test_missing_vendor_sdk_helpful_message(monkeypatch):
    """If the user calls a vendor-backed client but the SDK isn't installed,
    the ImportError should name the install command."""
    # Pretend `anthropic` is missing.
    monkeypatch.setitem(sys.modules, "anthropic", None)  # importlib treats None as missing
    from evsys_sdk.inference.claude import ClaudeInference  # registration is lazy

    with pytest.raises(ImportError, match="pip install anthropic"):
        ClaudeInference()

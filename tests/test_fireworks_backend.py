"""Fireworks backend wiring — registry + config + the thin training subclass.

The deep training path (forward/backward, sampling) is shared with TinkerBackend
(covered by its tests); these tests pin the Fireworks-specific seams: it's
registered, selectable from config, its training allocator only swaps the
service client, and it errors clearly without a key / without the optional dep.
"""

from __future__ import annotations

import asyncio

import pytest

from evsys_sdk.config import BackendConfig
from evsys_sdk.registry import get_backend, list_backends
from evsys_sdk.training.fireworks_backend import FireworksBackend
from evsys_sdk.training.tinker_backend import TinkerBackend


def test_fireworks_registered_and_selectable():
    assert "fireworks" in list_backends()
    assert get_backend("fireworks").name == "fireworks"
    assert BackendConfig(kind="fireworks").kind == "fireworks"


def test_training_subclass_only_swaps_the_service_client():
    # The whole point: FireworksBackend reuses TinkerBackend's allocator and
    # only overrides the service-client factory + the default key env.
    assert issubclass(FireworksBackend, TinkerBackend)
    assert FireworksBackend.DEFAULT_API_KEY_ENV == "FIREWORKS_API_KEY"
    assert TinkerBackend.DEFAULT_API_KEY_ENV == "TINKER_API_KEY"
    assert (
        FireworksBackend._make_service_client.__func__
        is not TinkerBackend._make_service_client.__func__
    )


def test_registry_prepare_requires_key(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY not set"):
        get_backend("fireworks")().prepare(model={"name": "m"}, run_dir="/tmp/x")


def test_registry_prepare_errors_without_dep(monkeypatch):
    # fireworks-ai is an optional extra; with a key set but the SDK absent, the
    # error should tell the user to install the extra (not an opaque ImportError).
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake")
    pytest.importorskip  # keep import order; no-op
    try:
        import fireworks  # noqa: F401
        pytest.skip("fireworks-ai is installed; the no-dep path can't be exercised")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="evsys-sdk\\[fireworks\\]"):
        get_backend("fireworks")().prepare(model={"name": "m"}, run_dir="/tmp/x")


def test_create_uses_overridden_service_client(monkeypatch):
    # FireworksBackend.create() must build the LoRA client off the Firetitan
    # service client (the override), not tinker's — verified with a fake client.
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake")

    class _FakeTraining:
        pass

    class _FakeService:
        def __init__(self):
            self.calls = []

        async def create_lora_training_client_async(self, model_name, **kw):
            self.calls.append((model_name, kw))
            return _FakeTraining()

    fake = _FakeService()
    monkeypatch.setattr(
        FireworksBackend, "_make_service_client",
        classmethod(lambda cls, *, api_key_env: fake),
    )
    monkeypatch.setattr(
        "evsys_sdk.training.tinker_backend.get_tokenizer", lambda name: object()
    )

    backend = asyncio.run(
        FireworksBackend.create(model_name="Qwen/Qwen3.5-4B", lora_rank=8)
    )
    assert isinstance(backend, FireworksBackend)
    assert backend._service is fake
    assert fake.calls and fake.calls[0][0] == "Qwen/Qwen3.5-4B"
    assert fake.calls[0][1].get("rank") == 8

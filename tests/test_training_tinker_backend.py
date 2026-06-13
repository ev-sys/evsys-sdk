"""Tests for ``evsys_sdk.training.tinker_backend.TinkerBackend``.

Stub out ``tinker.ServiceClient`` so the tests exercise the wrapper logic
end-to-end without spending a real tinker session: factory routing
(fresh-LoRA vs resume), method passthrough, sampler-snapshot path, error
when ``TINKER_API_KEY`` is unset.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("tinker")  # optional dep; not installed in base CI
pytest.importorskip("torch")
pytest.importorskip("tinker_cookbook")

import evsys_sdk.training.tinker_backend as tb_module
from evsys_sdk.training.tinker_backend import TinkerBackend, TinkerSamplingClient


# ---------------------------------------------------------------------------
# Doubles for tinker.ServiceClient + training_client + sampling_client
# ---------------------------------------------------------------------------


class _FakeFuture:
    def __init__(self, value: Any) -> None:
        self._value = value

    async def result_async(self) -> Any:
        return self._value

    def result(self) -> Any:
        return self._value


class _PathResult:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeTrainingClient:
    def __init__(self) -> None:
        self.fb_calls: list[dict] = []
        self.fb_custom_calls: list[dict] = []
        self.optim_calls: list[Any] = []
        self.save_sampler_calls: list[str] = []
        self.save_state_calls: list[str] = []

    def forward_backward_async(self, **kwargs):
        # Capture exactly what TinkerBackend passed; the wrapper omits
        # loss_fn_config when None, so kwargs reflects the real call shape.
        self.fb_calls.append(dict(kwargs))
        return _FakeFuture({"loss_fn_outputs": []})

    def forward_backward_custom_async(self, *, data, loss_fn):
        self.fb_custom_calls.append({"data": data, "loss_fn": loss_fn})
        return _FakeFuture({"loss_fn_outputs": []})

    def optim_step_async(self, adam):
        self.optim_calls.append(adam)
        return _FakeFuture({"metrics": {}})

    async def save_weights_for_sampler_async(self, name):
        self.save_sampler_calls.append(name)
        return _FakeFuture(_PathResult(f"tinker://sampler/{name}"))

    async def save_state_async(self, name):
        self.save_state_calls.append(name)
        return _FakeFuture(_PathResult(f"tinker://state/{name}"))


class _FakeServiceClient:
    """Replacement for tinker.ServiceClient. Records every call."""

    def __init__(self) -> None:
        self.lora_calls: list[dict] = []
        self.resume_calls: list[dict] = []
        self.sampling_calls: list[dict] = []
        self.tc = _FakeTrainingClient()

    async def create_lora_training_client_async(
        self, base_model, rank=32, **extra,
    ):
        self.lora_calls.append({"base_model": base_model, "rank": rank, **extra})
        return self.tc

    async def create_training_client_from_state_with_optimizer_async(
        self, state_path, **extra,
    ):
        self.resume_calls.append({"state_path": state_path, **extra})
        return self.tc

    def create_sampling_client(self, *, base_model=None, model_path=None,
                               **extra):
        self.sampling_calls.append({
            "base_model": base_model, "model_path": model_path, **extra,
        })
        return _FakeRawSamplingClient(model_path=model_path)


class _FakeRawSamplingClient:
    def __init__(self, *, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.sample_calls: list[dict] = []
        self.logprob_calls: list[Any] = []

    async def sample_async(self, **kwargs):
        self.sample_calls.append(kwargs)
        return {"sequences": []}

    async def compute_logprobs_async(self, prompt):
        self.logprob_calls.append(prompt)
        return [0.0] * prompt.length


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_tinker(monkeypatch):
    """Replace tinker.ServiceClient + get_tokenizer with deterministic doubles."""
    svc = _FakeServiceClient()
    monkeypatch.setattr(tb_module.tinker, "ServiceClient", lambda: svc)
    monkeypatch.setattr(tb_module, "get_tokenizer", lambda model: _FakeTok(model))
    monkeypatch.setenv("TINKER_API_KEY", "sk-test")
    return svc


class _FakeTok:
    def __init__(self, model: str) -> None:
        self.model = model


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------


def test_create_fresh_lora_when_no_resume_path(patched_tinker):
    backend = asyncio.run(TinkerBackend.create(
        model_name="Qwen/Qwen3-4B", lora_rank=16, renderer_name="qwen3_5",
    ))
    assert patched_tinker.lora_calls == [{
        "base_model": "Qwen/Qwen3-4B",
        "rank": 16,
        "user_metadata": {"renderer_name": "qwen3_5"},
    }]
    assert patched_tinker.resume_calls == []
    assert backend.get_tokenizer().model == "Qwen/Qwen3-4B"


def test_create_resumes_when_state_path_given(patched_tinker):
    asyncio.run(TinkerBackend.create(
        model_name="Qwen/Qwen3-4B",
        resume_state_path="tinker://state/step_500",
    ))
    assert patched_tinker.resume_calls == [{
        "state_path": "tinker://state/step_500",
        "user_metadata": None,
    }]
    assert patched_tinker.lora_calls == []


def test_create_requires_tinker_api_key(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TINKER_API_KEY"):
        asyncio.run(TinkerBackend.create(model_name="m"))


# ---------------------------------------------------------------------------
# Method passthrough
# ---------------------------------------------------------------------------


def test_forward_backward_async_passes_loss_fn_config_when_set(patched_tinker):
    backend = asyncio.run(TinkerBackend.create(model_name="m"))
    fut = backend.forward_backward_async(
        ["d1", "d2"], loss_fn="cross_entropy",
        loss_fn_config={"label_smoothing": 0.1},
    )
    assert patched_tinker.tc.fb_calls == [{
        "data": ["d1", "d2"],
        "loss_fn": "cross_entropy",
        "loss_fn_config": {"label_smoothing": 0.1},
    }]


def test_forward_backward_async_omits_empty_loss_fn_config(patched_tinker):
    """An empty/None config should NOT be forwarded — tinker rejects unknown."""
    backend = asyncio.run(TinkerBackend.create(model_name="m"))
    backend.forward_backward_async(["d"], loss_fn="cross_entropy")
    assert "loss_fn_config" not in patched_tinker.tc.fb_calls[0]


def test_forward_backward_custom_async_passes_callable(patched_tinker):
    backend = asyncio.run(TinkerBackend.create(model_name="m"))

    def _custom(model_out, batch_meta):
        return 0.5

    backend.forward_backward_custom_async(["d"], _custom)
    assert patched_tinker.tc.fb_custom_calls[0]["loss_fn"] is _custom


def test_optim_step_async_passes_adam(patched_tinker):
    import tinker
    backend = asyncio.run(TinkerBackend.create(model_name="m"))
    adam = tinker.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8)
    backend.optim_step_async(adam)
    assert patched_tinker.tc.optim_calls == [adam]


# ---------------------------------------------------------------------------
# Save / snapshot path
# ---------------------------------------------------------------------------


def test_save_for_sampler_unwraps_path(patched_tinker):
    backend = asyncio.run(TinkerBackend.create(model_name="m"))
    path = asyncio.run(backend.save_for_sampler("step_100"))
    assert path == "tinker://sampler/step_100"
    assert patched_tinker.tc.save_sampler_calls == ["step_100"]


def test_save_full_state_unwraps_path(patched_tinker):
    backend = asyncio.run(TinkerBackend.create(model_name="m"))
    path = asyncio.run(backend.save_full_state("step_100"))
    assert path == "tinker://state/step_100"
    assert patched_tinker.tc.save_state_calls == ["step_100"]


def test_snapshot_sampling_client_builds_wrapped_client(patched_tinker):
    backend = asyncio.run(TinkerBackend.create(model_name="Qwen/Qwen3-4B"))
    sc = asyncio.run(backend.snapshot_sampling_client(name="eval_50"))
    assert isinstance(sc, TinkerSamplingClient)
    assert sc.name == "eval_50"
    # the underlying create_sampling_client should be called with the URI
    # we just minted via save_weights_for_sampler.
    assert patched_tinker.sampling_calls == [{
        "base_model": "Qwen/Qwen3-4B",
        "model_path": "tinker://sampler/eval_50",
    }]


def test_snapshot_auto_increments_label_when_none(patched_tinker):
    backend = asyncio.run(TinkerBackend.create(model_name="m"))
    asyncio.run(backend.snapshot_sampling_client())
    asyncio.run(backend.snapshot_sampling_client())
    assert patched_tinker.tc.save_sampler_calls == ["snap_1", "snap_2"]


def test_raises_when_save_result_has_no_path(monkeypatch, patched_tinker):
    """If tinker ever changes its result shape, we surface the failure
    early rather than passing None into the manifest."""
    class _NoPathResult:
        pass
    async def _bad_save(name):
        return _FakeFuture(_NoPathResult())
    patched_tinker.tc.save_weights_for_sampler_async = _bad_save  # type: ignore[assignment]
    backend = asyncio.run(TinkerBackend.create(model_name="m"))
    with pytest.raises(RuntimeError, match=r"\.path attribute"):
        asyncio.run(backend.save_for_sampler("x"))

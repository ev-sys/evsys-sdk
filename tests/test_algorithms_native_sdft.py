"""End-to-end test of ``evsys_sdk.algorithms.native_sdft.NativeSDFT``.

Mirrors the SFT composer test (test_algorithms_native_sft.py) but for the
self-distillation path. Uses MockBackend + canned MockSamplingClient
responses so the full rollout → teacher score → CE train cycle runs
without a real tinker session.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("tinker")  # optional dep; not installed in base CI
pytest.importorskip("torch")

import evsys_sdk.algorithms.native_sdft as native_sdft_module
from evsys_sdk.algorithms.native_sdft import NativeSDFT, NativeSDFTConfig
from evsys_sdk.protocols import RunResult
from evsys_sdk.registry import get_algorithm
from evsys_sdk.training import MockBackend, MockSamplingClient


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _StubLogStore:
    def __init__(self) -> None:
        self.hyperparams: dict | None = None
        self.metric_rows: list[dict] = []
        self.artifacts: list[tuple[str, str, str]] = []

    def log_hyperparams(self, hp):
        self.hyperparams = dict(hp)

    def log_metrics(self, metrics, *, step, split="train"):
        self.metric_rows.append({"step": step, "split": split, "metrics": dict(metrics)})

    def log_artifact(self, key, value, *, kind):
        self.artifacts.append((key, value, kind))


class _StubTokenizer:
    def apply_chat_template(self, messages, *, tokenize=True,
                            add_generation_prompt=False, **extra):
        parts = [f"<{m['role'][0]}>{m['content']}" for m in messages]
        text = "$".join(parts)
        if add_generation_prompt:
            text += "?"
        return text

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class _SDFTMockBackend(MockBackend):
    """MockBackend that exposes a fake ``_service`` for the teacher-client
    construction NativeSDFT does. Returns a stub teacher SamplingClient."""

    def __init__(self, *, tokenizer):
        super().__init__(tokenizer=tokenizer)
        outer = self

        class _Svc:
            def create_sampling_client(self, *, base_model=None, **kw):
                return outer._sampler_factory(f"teacher_{base_model}").raw \
                    if hasattr(outer._sampler_factory(f"teacher_{base_model}"), "raw") \
                    else outer._sampler_factory(f"teacher_{base_model}")
        self._service = _Svc()


class _SDFTMockSampler:
    """MockSamplingClient that returns:

    - a SamplingResponse with one sequence of 4 tokens (the rollout)
    - a topk_prompt_logprobs list when topk_prompt_logprobs>0 (the teacher path)
    """

    def __init__(self, *, name="mock"):
        self.name = name
        self.calls: list[dict] = []

    async def sample_async(self, **kwargs):
        self.calls.append(kwargs)
        # Build a fake sequence with 4 tokens.
        class _Seq:
            tokens = [11, 12, 13, 14]
        topk = kwargs.get("topk_prompt_logprobs", 0)
        topk_resp = None
        if topk > 0:
            # length matches teacher_prompt + completion = prompt_len + 4.
            # We don't know prompt_len here; just emit a long list and let the
            # math truncate appropriately.
            entry = [(99, -0.0), (100, -0.5)]
            topk_resp = [entry] * 20

        class _Resp:
            pass

        r = _Resp()
        r.sequences = [_Seq()]
        r.topk_prompt_logprobs = topk_resp
        return r

    async def compute_logprobs_async(self, prompt):
        return [0.0] * prompt.length


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_tinker_backend(monkeypatch):
    """Replace TinkerBackend.create with a factory that returns _SDFTMockBackend.

    The backend's sampler factory returns _SDFTMockSampler so both student
    rollouts AND teacher topK scoring exercise the right code paths.
    """
    backend = _SDFTMockBackend(tokenizer=_StubTokenizer())
    backend._sampler_factory = lambda name: _SDFTMockSampler(name=name)  # type: ignore[assignment]

    async def _factory(**kwargs):
        backend._model_name = kwargs.get("model_name")  # type: ignore[attr-defined]
        return backend

    monkeypatch.setattr(native_sdft_module.TinkerBackend, "create", _factory)
    return backend


@pytest.fixture
def ctx(tmp_path: Path):
    rows = [
        {"question": f"Q{i}", "golden_answer": f"A{i}"}
        for i in range(20)
    ]

    class _Backend:
        name = "tinker"

    class _Ctx:
        def __init__(self):
            self.run_id = "run-sdft"
            self.output_dir = tmp_path
            self.log_store = _StubLogStore()
            self.backend = _Backend()
            self.extras = {
                "train_rows": rows,
                "backend_handles": {"model_name": "Qwen/Qwen3-4B"},
                "model_name": "Qwen/Qwen3-4B",
            }

    return _Ctx()


# ---------------------------------------------------------------------------
# Registry + Config
# ---------------------------------------------------------------------------


def test_registered_under_native_sdft():
    assert get_algorithm("native_sdft") is NativeSDFT


def test_config_defaults():
    cfg = NativeSDFTConfig()
    assert cfg.topk == 20
    assert cfg.batch_size == 4
    assert cfg.teacher_sync_every is None  # static teacher
    assert cfg.skip_first_n_tokens == 3


def test_config_rejects_unknown_kwarg():
    with pytest.raises(Exception):
        NativeSDFTConfig(bogus=True)


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------


def test_rejects_non_tinker_backend(ctx):
    class _M:
        name = "mock"
    ctx.backend = _M()
    with pytest.raises(RuntimeError, match="backend=tinker"):
        NativeSDFT(max_steps=2, batch_size=4).train(ctx)


def test_rejects_rows_missing_question_or_golden(ctx):
    ctx.extras["train_rows"] = [{"question": "q1"}, {"question": "q2"}]
    with pytest.raises(RuntimeError, match="question.*golden_answer"):
        NativeSDFT(max_steps=2, batch_size=2).train(ctx)


def test_rejects_missing_model_name(ctx):
    ctx.extras["backend_handles"] = {}
    ctx.extras.pop("model_name", None)
    with pytest.raises(RuntimeError, match="model_name"):
        NativeSDFT(max_steps=2, batch_size=4).train(ctx)


# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------


def test_train_runs_end_to_end(patched_tinker_backend, ctx):
    algo = NativeSDFT(
        max_steps=3, batch_size=4, save_at_fractions=[1.0],
        system_prompt="SYS", user_template="Query: {question}",
        skip_first_n_tokens=0,   # so every completion position contributes
    )
    result = algo.train(ctx)
    assert isinstance(result, RunResult)
    assert result.status == "completed"
    # 3 fb + 3 optim steps
    assert len(patched_tinker_backend.fb_calls) == 3
    assert len(patched_tinker_backend.optim_calls) == 3
    # final sampler URI surfaces in artifacts
    assert result.artifacts.get("checkpoint-final", "").startswith("mock://sampler/")


def test_train_logs_per_step_sdft_metrics(patched_tinker_backend, ctx):
    """SDFTStepBuilder.batch.metrics should land in each per-step row."""
    algo = NativeSDFT(max_steps=2, batch_size=4, skip_first_n_tokens=0)
    algo.train(ctx)
    train_rows = [r for r in ctx.log_store.metric_rows if r["split"] == "train"]
    assert len(train_rows) == 2
    # `sdft/*` keys should appear (from build_topk_targets metrics merged via
    # batch.metrics).
    for r in train_rows:
        assert "sdft/num_datums" in r["metrics"]
        assert "sdft/topk" in r["metrics"]
        # progress + optim keys still present
        assert "progress/step" in r["metrics"]


def test_train_logs_hyperparams_once(patched_tinker_backend, ctx):
    NativeSDFT(max_steps=2, batch_size=4).train(ctx)
    hp = ctx.log_store.hyperparams
    assert hp is not None
    assert hp["algorithm"] == "native_sdft"
    assert hp["model_name"] == "Qwen/Qwen3-4B"
    assert hp["total_steps"] == 2

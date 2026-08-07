"""End-to-end test of ``evsys_sdk.algorithms.sdft.SDFT``.

Mirrors the SFT composer test (test_algorithms_sft.py) but for the
self-distillation path. Uses MockBackend + canned MockSamplingClient
responses so the full rollout → teacher score → CE train cycle runs
without a real tinker session.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tinker")  # optional dep; not installed in base CI
pytest.importorskip("torch")

import evsys_sdk.algorithms.sdft as sdft_module
from evsys_sdk.algorithms.sdft import SDFT, SDFTConfig
from evsys_sdk.protocols import RunResult
from evsys_sdk.registry import get_algorithm
from evsys_sdk.training import MockBackend

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
    construction SDFT does. Returns a stub teacher SamplingClient."""

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
            __slots__ = ("tokens",)

            def __init__(self):
                self.tokens = [11, 12, 13, 14]
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


async def _fake_run_harbor_rollouts(tasks, **kwargs):
    """Stand-in for the harbor engine's student rollout (verify=False): one
    canned 4-token completion per task, as a one-trajectory TrajectoryGroup (the
    teacher topK path still runs against the mock teacher client)."""
    from evsys_sdk.training.trajectory import Trajectory, TrajectoryGroup, Turn

    return [
        TrajectoryGroup(trajectories=[Trajectory(turns=[Turn(
            prompt_tokens=[1, 2, 3], completion_tokens=[11, 12, 13, 14],
            logprobs=[-0.5, -0.5, -0.5, -0.5],
        )])])
        for _ in tasks
    ]


@pytest.fixture
def patched_tinker_backend(monkeypatch):
    """TinkerBackend.create → _SDFTMockBackend; student rollout → mocked harbor
    generations; teacher topK still runs against the mock teacher client
    (_SDFTMockSampler via backend._service)."""
    backend = _SDFTMockBackend(tokenizer=_StubTokenizer())
    backend._sampler_factory = lambda name: _SDFTMockSampler(name=name)  # type: ignore[assignment]

    async def _factory(**kwargs):
        backend._model_name = kwargs.get("model_name")  # type: ignore[attr-defined]
        return backend

    monkeypatch.setattr(sdft_module.TinkerBackend, "create", _factory)
    monkeypatch.setattr(
        "evsys_sdk.training.harbor_engine.run_harbor_rollouts",
        _fake_run_harbor_rollouts,
    )
    return backend


@pytest.fixture
def ctx(tmp_path: Path):
    # Standardized PromptExample shape: inputs['question'] + expected.
    rows = [
        {"inputs": {"question": f"Q{i}"}, "expected": f"A{i}"}
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


def test_registered_under_sdft():
    assert get_algorithm("sdft") is SDFT


def test_config_defaults():
    cfg = SDFTConfig()
    assert cfg.topk == 20
    assert cfg.batch_size == 4
    assert cfg.teacher_sync_every is None  # static teacher
    assert cfg.skip_first_n_tokens == 3


def test_config_rejects_unknown_kwarg():
    with pytest.raises(Exception):
        SDFTConfig(bogus=True)


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------


def test_rejects_non_tinker_backend(ctx):
    class _M:
        name = "mock"
    ctx.backend = _M()
    with pytest.raises(RuntimeError, match="backend=tinker"):
        SDFT(max_steps=2, batch_size=4).train(ctx)


def test_rejects_rows_missing_question(ctx):
    # PromptExample shape but no inputs['question'] → SimpleSDFTDataset rejects.
    ctx.extras["train_rows"] = [
        {"inputs": {}, "expected": "a1"},
        {"inputs": {}, "expected": "a2"},
    ]
    with pytest.raises(ValueError, match="question"):
        SDFT(max_steps=2, batch_size=2).train(ctx)


def test_rejects_non_prompt_dataset_rows(ctx):
    # Wrong format entirely → parse_rows rejects strictly.
    ctx.extras["train_rows"] = [{"question": "q1", "golden_answer": "a1"}]
    with pytest.raises(ValueError, match="expected 'prompt_dataset'"):
        SDFT(max_steps=2, batch_size=2).train(ctx)


def test_rejects_missing_model_name(ctx):
    ctx.extras["backend_handles"] = {}
    ctx.extras.pop("model_name", None)
    with pytest.raises(RuntimeError, match="model_name"):
        SDFT(max_steps=2, batch_size=4).train(ctx)


# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------


def test_train_runs_end_to_end(patched_tinker_backend, ctx):
    algo = SDFT(
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
    """SDFT.build_batch's batch.metrics should land in each per-step row."""
    algo = SDFT(max_steps=2, batch_size=4, skip_first_n_tokens=0)
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
    SDFT(max_steps=2, batch_size=4).train(ctx)
    hp = ctx.log_store.hyperparams
    assert hp is not None
    assert hp["algorithm"] == "sdft"
    assert hp["model_name"] == "Qwen/Qwen3-4B"
    assert hp["total_steps"] == 2


# ---------------------------------------------------------------------------
# step_metrics — weighted soft CE (matches the optimizer)
# ---------------------------------------------------------------------------


def _soft_datum(*, weights_flat: list[float]) -> "tinker.Datum":
    import tinker
    import torch
    n = len(weights_flat)
    # pretend 1 position × K candidates (or N×K flattened — metric flattens anyway)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(
                torch.arange(n, dtype=torch.long),
            ),
            "weights": tinker.TensorData.from_torch(
                torch.tensor(weights_flat, dtype=torch.float32),
            ),
        },
    )


def test_step_metrics_uses_teacher_weights_not_unweighted_mean():
    """Peaked teacher: unweighted mean of K logprobs is dominated by the
    low-prob tail; weighted CE must follow the peak token."""
    from evsys_sdk.training.loop import TrainingBatch

    # K=3: peak weight 0.9 on token with logprob -1; tail tokens at -10
    weights = [0.9, 0.05, 0.05]
    logprobs = [-1.0, -10.0, -10.0]
    batch = TrainingBatch(
        data=[_soft_datum(weights_flat=weights)],
        loss_fn="cross_entropy",
    )

    class _Result:
        loss_fn_outputs = [{"logprobs": logprobs}]

    metrics = SDFT().step_metrics(0, batch, _Result())
    # weighted: -(0.9*-1 + 0.05*-10 + 0.05*-10) / 1.0 = 1.9
    assert metrics["train/mean_loss"] == pytest.approx(1.9)
    assert metrics["train/mean_logprob"] == pytest.approx(-1.9)
    assert metrics["train/loss_n_tokens"] == pytest.approx(1.0)

    # Unweighted nonzero mean would be -( -1-10-10 )/3 = 7 — must NOT match.
    assert metrics["train/mean_loss"] != pytest.approx(7.0)


def test_step_metrics_ignores_zero_weight_slots():
    from evsys_sdk.training.loop import TrainingBatch

    batch = TrainingBatch(
        data=[_soft_datum(weights_flat=[0.0, 1.0, 0.0])],
        loss_fn="cross_entropy",
    )

    class _Result:
        loss_fn_outputs = [{"logprobs": [-99.0, -0.5, -99.0]}]

    metrics = SDFT().step_metrics(0, batch, _Result())
    assert metrics["train/mean_loss"] == pytest.approx(0.5)


def test_step_metrics_empty_without_outputs():
    from evsys_sdk.training.loop import TrainingBatch

    batch = TrainingBatch(data=[_soft_datum(weights_flat=[1.0])], loss_fn="cross_entropy")

    class _Result:
        loss_fn_outputs = None

    assert SDFT().step_metrics(0, batch, _Result()) == {}

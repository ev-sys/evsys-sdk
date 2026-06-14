"""End-to-end test of `evsys_sdk.algorithms.rl.RL`.

RL now hands rollouts to harbor's engine via
``evsys_sdk.training.harbor_engine.run_harbor_rollouts``. We mock that helper
(harbor isn't installed in CI) so the rest of the pipeline — HarborTask parsing,
advantage computation, IS-loss Datums, the training loop on MockBackend — runs
for real.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tinker")  # optional dep; not installed in base CI
pytest.importorskip("torch")

import tinker

import evsys_sdk.algorithms.rl as rl_module
from evsys_sdk.algorithms.rl import RL, RLConfig
from evsys_sdk.protocols import RunResult
from evsys_sdk.registry import get_algorithm
from evsys_sdk.training import MockBackend
from evsys_sdk.training.trajectory import Trajectory, TrajectoryGroup, Turn


class _StubLogStore:
    def __init__(self):
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
        return "$".join(f"<{m['role'][0]}>{m['content']}" for m in messages)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


async def _fake_run_harbor_rollouts(tasks, *, num_samples=1, **kwargs):
    """Stand-in for the harbor engine: one TrajectoryGroup per task, each with
    a canned single-turn trajectory + reward (no harbor / containers)."""
    groups = []
    for t in tasks:
        trajs = [
            Trajectory(
                turns=[Turn(
                    prompt_tokens=[1, 2, 3],
                    completion_tokens=[200, 201, 202],
                    logprobs=[-0.5, -0.5, -0.5],
                )],
                reward=1.0,
            )
            for _ in range(num_samples)
        ]
        groups.append(TrajectoryGroup(trajectories=trajs, tags=list(t.metadata.get("tags") or [])))
    return groups


@pytest.fixture
def patched_tinker_backend(monkeypatch):
    backend = MockBackend(tokenizer=_StubTokenizer())

    async def _factory(**kwargs):
        backend._model_name = kwargs.get("model_name")  # type: ignore[attr-defined]
        return backend

    monkeypatch.setattr(rl_module.TinkerBackend, "create", _factory)
    monkeypatch.setattr(
        "evsys_sdk.training.harbor_engine.run_harbor_rollouts",
        _fake_run_harbor_rollouts,
    )
    return backend


@pytest.fixture
def ctx(tmp_path: Path):
    rows = [
        {
            "task_id": f"t{i}",
            "instruction": f"P{i}",
            "verifier": {"kind": "in_process", "fn_name": "exact_match", "expected": "T200_201_202"},
            "metadata": {"tags": ["foo"]},
        }
        for i in range(20)
    ]

    class _Backend:
        name = "tinker"

    class _Ctx:
        def __init__(self):
            self.run_id = "run-rl"
            self.output_dir = tmp_path
            self.log_store = _StubLogStore()
            self.backend = _Backend()
            self.extras = {
                "train_rows": rows,
                "backend_handles": {"model_name": "Qwen/Qwen3-4B"},
                "model_name": "Qwen/Qwen3-4B",
            }

    return _Ctx()


# --- Registry + Config -----------------------------------------------------


def test_registered_under_rl():
    assert get_algorithm("rl") is RL


def test_config_defaults():
    cfg = RLConfig()
    assert cfg.batch_size == 4
    assert cfg.num_samples == 1
    assert cfg.drop_constant_reward is True
    assert cfg.learning_rate == 1.0e-5


def test_config_rejects_unknown_kwarg():
    with pytest.raises(Exception):
        RLConfig(bogus_field=True)


# --- Validation gates ------------------------------------------------------


def test_rejects_non_tinker_backend(ctx):
    class _M:
        name = "mock"
    ctx.backend = _M()
    with pytest.raises(RuntimeError, match="backend=tinker"):
        RL(max_steps=2, batch_size=4, verifier_name="exact_match").train(ctx)


def test_rejects_missing_train_rows(ctx, patched_tinker_backend):
    ctx.extras["train_rows"] = []
    with pytest.raises(RuntimeError, match="train_rows"):
        RL(max_steps=2, batch_size=4, verifier_name="exact_match").train(ctx)


def test_rejects_unknown_verifier(ctx, patched_tinker_backend):
    ctx.extras["train_rows"] = [{
        "task_id": "t0", "instruction": "P0",
        "verifier": {"kind": "in_process", "fn_name": "not_real_verifier", "expected": "x"},
    }]
    with pytest.raises(RuntimeError, match="unknown verifier_name"):
        RL(max_steps=2, batch_size=1).train(ctx)


def test_rejects_missing_verifier_fn(ctx, patched_tinker_backend):
    ctx.extras["train_rows"] = [{
        "task_id": "t0", "instruction": "P0",
        "verifier": {"kind": "in_process", "fn_name": "", "expected": "x"},
    }]
    with pytest.raises(RuntimeError, match="no verifier fn_name"):
        RL(max_steps=2, batch_size=1).train(ctx)


def test_rejects_non_harbor_task_rows(ctx, patched_tinker_backend):
    ctx.extras["train_rows"] = [{"prompt": "P0", "expected": "x"}]
    with pytest.raises(ValueError, match="expected 'harbor_task'"):
        RL(max_steps=2, batch_size=1, verifier_name="exact_match").train(ctx)


# --- End-to-end happy path -------------------------------------------------


def test_train_runs_end_to_end(patched_tinker_backend, ctx):
    algo = RL(max_steps=2, batch_size=4, num_samples=1,
              verifier_name="exact_match", drop_constant_reward=False)
    result = algo.train(ctx)
    assert isinstance(result, RunResult)
    assert result.status == "completed"
    assert len(patched_tinker_backend.fb_calls) == 2
    assert len(patched_tinker_backend.optim_calls) == 2
    assert patched_tinker_backend.fb_calls[0]["loss_fn"] == "importance_sampling"
    assert result.artifacts.get("checkpoint-final", "").startswith("mock://sampler/")


def test_train_logs_reward_metrics_per_step(patched_tinker_backend, ctx):
    algo = RL(max_steps=2, batch_size=4, num_samples=1,
              verifier_name="exact_match", drop_constant_reward=False)
    algo.train(ctx)
    train_rows = [r for r in ctx.log_store.metric_rows if r["split"] == "train"]
    assert len(train_rows) == 2
    for r in train_rows:
        assert "reward/mean" in r["metrics"]
        assert "reward/n_trajectories" in r["metrics"]
        assert "progress/step" in r["metrics"]


def test_train_logs_hyperparams(patched_tinker_backend, ctx):
    RL(max_steps=2, batch_size=4, num_samples=1,
       verifier_name="exact_match", drop_constant_reward=False).train(ctx)
    hp = ctx.log_store.hyperparams
    assert hp is not None
    assert hp["algorithm"] == "rl"
    assert hp["n_tasks"] == 20
    assert hp["total_steps"] == 2

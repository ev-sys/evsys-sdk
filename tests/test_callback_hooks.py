"""Foundation for callback-based logging: the experiment-scope hooks,
the shared LogContext, and the error-isolated dispatch fan-out."""

from __future__ import annotations

from pathlib import Path

import pytest

from evsys_sdk import (
    AlgorithmConfig,
    BackendConfig,
    DataConfig,
    Experiment,
    ExperimentConfig,
    ModelConfig,
    RunConfig,
)
from evsys_sdk.config import CallbackSpec
from evsys_sdk.protocols import RunResult
from evsys_sdk.registry import _callbacks as _CB_REGISTRY
from evsys_sdk.registry import register_callback
from evsys_sdk.training.callbacks import Callback, LogContext, dispatch

_EXPERIMENT_HOOKS = [
    "on_experiment_start",
    "on_group_start",
    "on_run_start",
    "on_benchmark_eval",
    "on_run_end",
    "on_experiment_end",
]


def test_callback_base_has_noop_experiment_hooks():
    cb = Callback()
    ctx = LogContext(output_dir=Path("."))
    # every new hook exists and is a no-op (returns None, doesn't raise)
    assert cb.on_experiment_start(ctx) is None
    assert cb.on_group_start(ctx, "g0") is None
    assert cb.on_run_start(ctx) is None
    assert cb.on_benchmark_eval(ctx, object(), [], step=None) is None
    assert cb.on_run_end(ctx, object(), object()) is None
    assert cb.on_experiment_end(ctx, object()) is None


def test_logcontext_ids_accumulate():
    ctx = LogContext(output_dir=Path("/tmp/x"), run_key="run_0")
    ctx.ids["experiment_id"] = "exp1"
    ctx.ids["group:base"] = "grp1"
    ctx.ids["run_id"] = "run1"
    assert ctx.ids == {"experiment_id": "exp1", "group:base": "grp1", "run_id": "run1"}
    ctx.extras["wandb_url"] = "http://w/and/b"
    assert ctx.extras["wandb_url"] == "http://w/and/b"


def test_dispatch_is_error_isolated():
    seen: list[str] = []

    class Boom(Callback):
        def on_run_start(self, ctx):
            raise RuntimeError("boom")

    class Good(Callback):
        def on_run_start(self, ctx):
            seen.append("good")

    ctx = LogContext(output_dir=Path("."))
    # the raising callback must NOT stop the next one, and dispatch must not raise
    dispatch([Boom(), Good()], "on_run_start", ctx)
    assert seen == ["good"]


def test_dispatch_passes_kwargs_and_skips_unknown_hook():
    seen: list[int | None] = []

    class C(Callback):
        def on_benchmark_eval(self, ctx, eval_result, predictions, *, step=None):
            seen.append(step)

    ctx = LogContext(output_dir=Path("."))
    dispatch([C()], "on_benchmark_eval", ctx, object(), [], step=42)
    assert seen == [42]
    # a hook name no callback implements is a clean no-op
    dispatch([C()], "on_nonexistent_hook", ctx)


# ---------------------------------------------------------------------------
# Integration: Experiment dispatches the experiment-scope hooks on the SAME
# instances it built from ExperimentConfig.callbacks.
# ---------------------------------------------------------------------------

_EVENTS: list[tuple] = []


@register_callback("_capture_logger")
class _CaptureLogger(Callback):
    name = "_capture_logger"

    def on_experiment_start(self, ctx):
        _EVENTS.append(("exp_start", ctx.ids.get("experiment_id")))

    def on_run_start(self, ctx):
        _EVENTS.append(("run_start", ctx.run_key, id(ctx)))

    def on_run_end(self, ctx, run_result, arm):
        _EVENTS.append(("run_end", arm.name, getattr(run_result, "status", None)))

    def on_experiment_end(self, ctx, result):
        _EVENTS.append(("exp_end", result.status, id(ctx)))


def _capture_train_fn(cfg):
    run = cfg.run
    return [RunResult(run_id=run.name, status="completed", metrics={"loss": 0.5}, artifacts={})]


def test_experiment_dispatches_lifecycle_hooks_on_shared_ctx(tmp_path):
    _EVENTS.clear()
    cfg = ExperimentConfig(
        name="cbexp",
        output_dir=str(tmp_path / "out"),
        callbacks=[CallbackSpec(kind="_capture_logger")],
        run=RunConfig(
            name="arm0",
            data=DataConfig(source_kind="in_memory", rows=[{"q": "Q"}]),
            model=ModelConfig(name="m"),
            algorithm=AlgorithmConfig(kind="mock_sft", params={"lora_rank": 0}),
            backend=BackendConfig(kind="mock"),
        ),
    )
    result = Experiment(cfg, train_fn=_capture_train_fn).run()

    kinds = [e[0] for e in _EVENTS]
    # lifecycle order: experiment start → run start → run end → experiment end
    assert kinds == ["exp_start", "run_start", "run_end", "exp_end"]
    # run hook saw the arm + completed status
    assert _EVENTS[2][1] == "arm0" and _EVENTS[2][2] == "completed"
    assert _EVENTS[3][1] == result.status
    # SAME shared LogContext instance across on_run_start and on_experiment_end
    assert _EVENTS[1][2] == _EVENTS[3][2]


def teardown_module(_mod):
    # keep the global callback registry clean for other tests
    _CB_REGISTRY.unregister("_capture_logger")

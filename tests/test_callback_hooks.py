"""Foundation for callback-based logging: the experiment-scope hooks,
the shared LogContext, and the error-isolated dispatch fan-out."""

from __future__ import annotations

from pathlib import Path

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

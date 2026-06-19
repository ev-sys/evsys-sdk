"""Logger callbacks (wandb / evsys / tensorboard / local) — unit tests that
drive the hooks directly with fakes (no real wandb/tb/store needed)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from evsys_sdk.registry import get_callback
from evsys_sdk.training.callbacks import LogContext, WandbLoggerCallback


# --- fake wandb -------------------------------------------------------------

class _FakeRun:
    def __init__(self):
        self.logs: list[tuple] = []
        self.url = "http://wandb/run/1"
        self.finished = False

    def log(self, data, step=None):
        self.logs.append((step, data))

    def finish(self):
        self.finished = True


class _FakeWandb:
    def __init__(self):
        self.run = _FakeRun()
        self.init_kwargs = None

    def init(self, **kw):
        self.init_kwargs = kw
        return self.run

    class Table:
        def __init__(self, columns):
            self.columns = columns
            self.rows: list[tuple] = []

        def add_data(self, *a):
            self.rows.append(a)


def _ctx():
    return LogContext(output_dir=Path("."), run_key="r0")


def test_wandb_logger_registered():
    assert get_callback("wandb_logger") is WandbLoggerCallback


def test_wandb_logger_full_lifecycle():
    cb = WandbLoggerCallback(project="proj")
    fake = _FakeWandb()
    cb._wandb = fake  # inject so _lazy_wandb skips the real import
    ctx = _ctx()

    cb.on_run_start(ctx)
    assert fake.init_kwargs["project"] == "proj"
    assert fake.init_kwargs["reinit"] is True
    assert ctx.extras["wandb_url"] == "http://wandb/run/1"   # surfaced for evsys_logger

    st = SimpleNamespace()
    cb.on_step_end(st, 0, None, {"loss": 1.5})
    assert fake.run.logs[-1] == (0, {"loss": 1.5})

    cb.on_eval(st, 5, "val", {"pass_rate": 0.5})
    assert fake.run.logs[-1] == (5, {"val/val/pass_rate": 0.5})

    eval_result = SimpleNamespace(name="full", metrics={"pass@3": 0.7})
    preds = [{"task_id": "t1", "expected": "X", "reward": 1.0}]
    cb.on_benchmark_eval(ctx, eval_result, preds, step=None)
    _, payload = fake.run.logs[-1]
    assert payload["eval/full/pass@3"] == 0.7
    assert payload["eval/full/predictions"].rows == [("t1", "X", 1.0)]

    cb.on_run_end(ctx, None, None)
    assert fake.run.finished and cb._run is None


def test_wandb_logger_log_every_skips():
    cb = WandbLoggerCallback(log_every=2)
    cb._wandb = _FakeWandb()
    ctx = _ctx()
    cb.on_run_start(ctx)
    st = SimpleNamespace()
    cb.on_step_end(st, 0, None, {"a": 1})   # (0+1)%2==1 → skip
    cb.on_step_end(st, 1, None, {"a": 2})   # (1+1)%2==0 → log
    assert [d for _, d in cb._run.logs] == [{"a": 2.0}]


def test_wandb_logger_disabled_when_absent_is_noop():
    cb = WandbLoggerCallback()
    cb._disabled = True   # simulate wandb not installed
    ctx = _ctx()
    cb.on_run_start(ctx)                       # no-op, no run opened
    cb.on_step_end(SimpleNamespace(), 0, None, {"loss": 1.0})  # no crash
    cb.on_run_end(ctx, None, None)
    assert cb._run is None and "wandb_url" not in ctx.extras


# --- LocalLoggerCallback ----------------------------------------------------

def test_local_logger_writes_metrics_and_predictions(tmp_path, capsys):
    from evsys_sdk.training.callbacks import LocalLoggerCallback
    import json

    cb = LocalLoggerCallback(print_every=1)
    ctx = LogContext(output_dir=tmp_path, run_key="arm0")
    cb.on_run_start(ctx)
    st = SimpleNamespace(num_steps=2)
    cb.on_step_end(st, 0, None, {"loss": 1.5})
    cb.on_step_end(st, 1, None, {"loss": 1.0})

    eval_result = SimpleNamespace(name="val/full", metrics={"pass@3": 0.5})
    preds = [{"task_id": "t1", "reward": 1.0}, {"task_id": "t2", "reward": 0.0}]
    cb.on_benchmark_eval(ctx, eval_result, preds, step=None)
    cb.on_run_end(ctx, SimpleNamespace(status="completed"), SimpleNamespace(name="arm0"))

    run_dir = tmp_path / "arm0"
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["split"] for r in rows] == ["train", "train"]
    assert rows[0]["metrics"]["loss"] == 1.5
    # predictions file written (name slashes sanitized)
    pred_lines = (run_dir / "predictions" / "val_full.jsonl").read_text().splitlines()
    assert len(pred_lines) == 2
    # summary.md mentions the eval
    summary = (run_dir / "summary.md").read_text()
    assert "val/full" in summary and "status: completed" in summary
    # printed a per-step line
    assert "[1/2]" in capsys.readouterr().out


def test_local_logger_registered():
    from evsys_sdk.training.callbacks import LocalLoggerCallback
    assert get_callback("local_logger") is LocalLoggerCallback


# --- TensorBoardLoggerCallback (no torch → disables cleanly) -----------------

def test_tensorboard_logger_disables_without_torch():
    from evsys_sdk.training.callbacks import TensorBoardLoggerCallback
    cb = TensorBoardLoggerCallback()
    cb._disabled = True   # simulate torch/tensorboard absent
    ctx = LogContext(output_dir=Path("."), run_key="r0")
    cb.on_run_start(ctx)                       # no writer opened
    cb.on_step_end(SimpleNamespace(), 0, None, {"loss": 1.0})  # no crash
    cb.on_run_end(ctx, None, None)
    assert cb._writer is None


def test_tensorboard_logger_registered():
    from evsys_sdk.training.callbacks import TensorBoardLoggerCallback
    assert get_callback("tensorboard_logger") is TensorBoardLoggerCallback

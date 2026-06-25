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
    cb.on_experiment_end(ctx, SimpleNamespace(hypothesis=None, conclusion="done"))

    run_dir = tmp_path / "arm0" / "logs"
    # train metrics live under training/, never a flat metrics.jsonl
    assert not (run_dir / "metrics.jsonl").exists()
    rows = [json.loads(l) for l in (run_dir / "training" / "metrics.jsonl").read_text().splitlines()]
    assert [r["split"] for r in rows] == ["train", "train"]
    assert rows[0]["metrics"]["loss"] == 1.5
    # benchmark (default split=test): aggregate scores + per-example predictions
    test_metrics = json.loads((run_dir / "test" / "metrics.jsonl").read_text().splitlines()[0])
    assert test_metrics["metrics"]["pass@3"] == 0.5
    pred_lines = (run_dir / "test" / "rollouts.jsonl").read_text().splitlines()
    assert len(pred_lines) == 2
    # conclusion.md mentions the eval + run status
    conclusion = (run_dir / "conclusion.md").read_text()
    assert "val/full" in conclusion and "status: completed" in conclusion
    # printed a per-step line
    assert "[1/2]" in capsys.readouterr().out


def test_local_logger_registered():
    from evsys_sdk.training.callbacks import LocalLoggerCallback
    assert get_callback("local_logger") is LocalLoggerCallback


def test_local_logger_writes_training_data_and_rollouts(tmp_path):
    """on_train_data persists the rendered training rows; on_rollout persists
    on-policy rollouts (text + reward + usage) under <run>/logs/."""
    import json

    from evsys_sdk.training.callbacks import LocalLoggerCallback
    from evsys_sdk.training.trajectory import Trajectory, TrajectoryGroup, Turn

    cb = LocalLoggerCallback(print_every=0)
    ctx = LogContext(output_dir=tmp_path, run_key="arm0")
    cb.on_run_start(ctx)

    cb.on_train_data(ctx, [{"messages": [{"role": "user", "content": "hi"}]}])
    td = (tmp_path / "arm0" / "logs" / "data" / "training_data.jsonl").read_text().splitlines()
    assert len(td) == 1
    assert json.loads(td[0])["messages"][0]["content"] == "hi"

    grp = TrajectoryGroup(trajectories=[Trajectory(
        turns=[Turn(text="hello", prompt_tokens=[1], completion_tokens=[2])],
        reward=1.0, metadata={"usage": {"cost_usd": 0.01}},
    )])
    cb.on_rollout(SimpleNamespace(num_steps=1), 0, [grp])
    roll = (tmp_path / "arm0" / "logs" / "training" / "rollouts.jsonl").read_text().splitlines()
    rec = json.loads(roll[0])
    assert rec["reward"] == 1.0
    assert rec["text"] == "hello"
    assert rec["usage"]["cost_usd"] == 0.01


def test_local_logger_persists_hypothesis_and_conclusion(tmp_path):
    """hypothesis (from config metadata at experiment start) and conclusion
    (from the ExperimentResult at experiment end) are written locally."""
    from evsys_sdk.training.callbacks import LocalLoggerCallback

    cb = LocalLoggerCallback(print_every=0)
    config = SimpleNamespace(name="exp42", metadata={"hypothesis": "more data helps"})
    ctx = LogContext(output_dir=tmp_path, config=config, run_key="arm0")

    cb.on_experiment_start(ctx)
    # experiment.md exists with the hypothesis right away
    exp_md = (tmp_path / "experiment.md").read_text()
    assert "hypothesis: more data helps" in exp_md

    # a run gets its own per-run hypothesis.md at run start
    cb.on_run_start(ctx)
    cb.on_run_end(ctx, SimpleNamespace(status="completed"), SimpleNamespace(name="arm0"))
    hyp = (tmp_path / "arm0" / "logs" / "hypothesis.md").read_text()
    assert "more data helps" in hyp

    # conclusion lands at experiment end — root experiment.md AND per-run conclusion.md
    result = SimpleNamespace(
        hypothesis="more data helps",
        conclusion="Best arm: arm0 at pass_rate=0.9000. 1/1 arms completed.",
    )
    cb.on_experiment_end(ctx, result)
    exp_md = (tmp_path / "experiment.md").read_text()
    assert "hypothesis: more data helps" in exp_md
    assert "conclusion: Best arm: arm0 at pass_rate=0.9000." in exp_md
    run_conclusion = (tmp_path / "arm0" / "logs" / "conclusion.md").read_text()
    assert "status: completed" in run_conclusion
    assert "conclusion: Best arm: arm0 at pass_rate=0.9000." in run_conclusion


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


# --- EvsysLoggerCallback (store-owning, mode B) -----------------------------

class _FakeStore:
    def __init__(self):
        self.calls: list[tuple] = []
        self._n = 0

    def _id(self, kind):
        self._n += 1
        return {"id": f"{kind}{self._n}"}

    def create_experiment(self, **kw):
        self.calls.append(("create_experiment", kw)); return self._id("exp")

    def create_group(self, experiment_id, name):
        self.calls.append(("create_group", experiment_id, name)); return self._id("grp")

    def create_run(self, **kw):
        self.calls.append(("create_run", kw)); return self._id("run")

    def log_metrics(self, **kw):
        self.calls.append(("log_metrics", kw)); return []

    def create_eval(self, **kw):
        self.calls.append(("create_eval", kw)); return self._id("eval")

    def log_predictions(self, run_id, preds):
        self.calls.append(("log_predictions", run_id, preds)); return {"ok": True}

    def update_run(self, run_id, **patch):
        self.calls.append(("update_run", run_id, patch)); return {"id": run_id}

    def add_checkpoint(self, *, run_id, uri, label=None, step=None, **kw):
        # keyword-only run_id, mirroring the real EvsysStore.add_checkpoint
        self.calls.append(("add_checkpoint", {"run_id": run_id, "uri": uri,
                                              "label": label, "step": step})); return {"ok": True}


def _run_config_stub():
    return SimpleNamespace(
        model_dump=lambda: {"name": "arm0"},
        algorithm=SimpleNamespace(kind="sft"),
        seed=42,
        name="arm0",
    )


def test_evsys_logger_owns_full_store_lifecycle():
    from evsys_sdk.training.callbacks import EvsysLoggerCallback

    cb = EvsysLoggerCallback(flush_every=1)
    fake = _FakeStore()
    cb._store = fake   # inject (skip env EvsysStore build)
    ctx = LogContext(
        output_dir=Path("."),
        config=SimpleNamespace(name="exp", metadata={"hypothesis": "h", "tags": ["t"]}),
    )

    cb.on_experiment_start(ctx)
    assert ctx.ids["experiment_id"] == "exp1"

    ctx.run_config = _run_config_stub()
    cb.on_run_start(ctx)
    assert ctx.ids["run_id"] == "run2"

    st = SimpleNamespace(ctx=ctx)
    cb.on_step_end(st, 0, None, {"loss": 1.5})   # flush_every=1 → immediate
    cb.on_eval(st, 1, "val", {"pass_rate": 0.5})

    eval_result = SimpleNamespace(name="full", benchmark_id="b1",
                                  metrics={"pass@3": 0.7}, breakdowns={})
    preds = [{"task_id": "t1", "reward": 1.0}]
    cb.on_benchmark_eval(ctx, eval_result, preds, step=None)
    cb.on_run_end(ctx, SimpleNamespace(status="completed", error=None),
                  SimpleNamespace(status="completed", error=None))

    kinds = [c[0] for c in fake.calls]
    assert kinds[:3] == ["create_experiment", "create_run", "log_metrics"]
    assert "create_eval" in kinds and "log_predictions" in kinds
    assert kinds[-1] == "update_run"
    # linkage: run parents to experiment; eval + metrics + update use run_id
    run_kw = next(c[1] for c in fake.calls if c[0] == "create_run")
    assert run_kw["experiment_id"] == "exp1" and run_kw["recipe_kind"] == "sft"
    eval_kw = next(c[1] for c in fake.calls if c[0] == "create_eval")
    assert eval_kw["run_id"] == "run2"
    # prediction row carries the eval_id
    _, _, pr = next(c for c in fake.calls if c[0] == "log_predictions")
    assert pr[0]["eval_id"] == "eval3"




def test_evsys_logger_on_checkpoint_uses_keyword_run_id():
    """Regression: EvsysStore.add_checkpoint is keyword-only on run_id, so
    on_checkpoint must pass run_id= (not positionally)."""
    from evsys_sdk.training.callbacks import EvsysLoggerCallback
    cb = EvsysLoggerCallback()
    fake = _FakeStore()
    cb._store = fake
    ctx = LogContext(output_dir=Path("."), ids={"run_id": "run9"})
    row = SimpleNamespace(name="final", batch=10, sampler_path="tinker://ckpt/s10", state_path=None)
    state = SimpleNamespace(ctx=ctx)
    cb.on_checkpoint(state, row)   # must NOT raise (the bug raised TypeError)
    call = next(c for c in fake.calls if c[0] == "add_checkpoint")
    assert call[1] == {"run_id": "run9", "uri": "tinker://ckpt/s10", "label": "final", "step": 10}

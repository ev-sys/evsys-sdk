"""Tests for RunLog — one human-readable log per run, organized into folders.

RunLog duck-types its rollout inputs, so these tests use lightweight stand-ins
instead of the real (tinker-importing) training.trajectory dataclasses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from evsys_sdk.run_log import RunLog, get_run_log, _CURRENT


@dataclass
class _Turn:
    completion_tokens: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    text: str = ""


@dataclass
class _Traj:
    turns: list = field(default_factory=list)
    reward: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class _Group:
    trajectories: list = field(default_factory=list)


@dataclass
class _Task:
    instruction: str = ""


def _group(rewards, texts=None):
    texts = texts or [f"answer {i}" for i in range(len(rewards))]
    return _Group([_Traj(turns=[_Turn(completion_tokens=[2, 3], logprobs=[-0.1, -0.2], text=t)],
                         reward=r) for r, t in zip(rewards, texts)])


def test_harbor_dir_is_referenced_not_a_numbered_folder(tmp_path):
    rl = RunLog(tmp_path / "run")
    d = rl.harbor_dir("train")
    assert d == tmp_path / "run" / "harbor" / "train"
    assert d.is_dir()


def test_log_data_after_transform(tmp_path):
    rl = RunLog(tmp_path / "run")
    rows = [{"q": f"row{i}"} for i in range(20)]
    rl.log_data(rows, dataset_meta={"name": "ds", "format": "chat_messages"}, n_preview=5)
    md = (tmp_path / "run" / "01_data" / "data.md").read_text()
    assert "rows: **20**" in md and "chat_messages" in md
    sample = (tmp_path / "run" / "01_data" / "after_transform.jsonl").read_text().splitlines()
    assert len(sample) == 5 and json.loads(sample[0])["q"] == "row0"


def test_log_chat_templates_text_not_ids(tmp_path):
    rl = RunLog(tmp_path / "run")
    rl.log_chat_templates(["<|im_start|>user\nhi<|im_end|>"], n=1)
    md = (tmp_path / "run" / "01_data" / "chat_template.md").read_text()
    assert "<|im_start|>" in md


def test_training_rollouts_reward_advantage_and_per_token(tmp_path):
    rl = RunLog(tmp_path / "run")
    groups = [_group([0.0, 1.0], texts=["bad", "great"])]
    advantages = [[-0.5, 0.5]]
    rl.log_training_rollouts(0, groups, advantages=advantages, tasks=[_Task("solve x")])
    md = (tmp_path / "run" / "02_training_rollouts" / "step_0.md").read_text()
    assert "reward 1.000" in md and "advantage 0.500" in md
    assert "great" in md and "solve x" in md
    assert "per-token logprobs" in md and "logprob" in md  # loss-per-token signal


def test_validation_rollouts(tmp_path):
    rl = RunLog(tmp_path / "run")
    rl.log_validation_rollouts("bench1", [_group([1.0])], tasks=[_Task("q")])
    md = (tmp_path / "run" / "03_validation_rollouts" / "bench1.md").read_text()
    assert "reward 1.000" in md


def test_render_metrics_splits_train_and_val(tmp_path):
    rl = RunLog(tmp_path / "run")
    logs = tmp_path / "run" / "logs"
    logs.mkdir(parents=True)
    with (logs / "metrics.jsonl").open("w") as f:
        f.write(json.dumps({"step": 0, "metrics": {"train/loss": 2.0, "optim/lr": 1e-4, "noise": 9}}) + "\n")
        f.write(json.dumps({"step": 1, "metrics": {"train/loss": 1.0, "val/bench/pass_rate": 0.5}}) + "\n")
    rl.render_metrics()
    train_csv = (tmp_path / "run" / "04_training_metrics" / "metrics.csv").read_text()
    assert "train/loss" in train_csv and "lr" in train_csv and "noise" not in train_csv
    val_csv = (tmp_path / "run" / "05_validation_metrics" / "metrics.csv").read_text()
    assert "val/bench/pass_rate" in val_csv


def test_validation_metrics_block_and_summary(tmp_path):
    rl = RunLog(tmp_path / "run", experiment_name="exp", run_name="armA", hypothesis="h")
    rl.log_validation_metrics("bench1", {"pass_rate": 0.75, "n_tasks": 4.0}, step=10)
    rl.write_summary(status="completed", conclusion="it worked")
    vm = (tmp_path / "run" / "05_validation_metrics" / "metrics.md").read_text()
    assert "bench1" in vm and "pass_rate" in vm and "step 10" in vm
    summary = (tmp_path / "run" / "summary.md").read_text()
    assert "h" in summary and "completed" in summary and "it worked" in summary and "bench1" in summary


def test_user_named_folder_api(tmp_path):
    rl = RunLog(tmp_path / "run")
    rl.note("my_transform", "dropped 3 rows missing tool_slug", title="cleanup")
    rl.record("my_transform", {"dropped": 3})
    d = rl.dir("scratch")
    (d / "x.txt").write_text("hi")
    notes = (tmp_path / "run" / "my_transform" / "notes.md").read_text()
    assert "dropped 3 rows" in notes and "cleanup" in notes
    rec = (tmp_path / "run" / "my_transform" / "records.jsonl").read_text()
    assert json.loads(rec)["dropped"] == 3
    assert (tmp_path / "run" / "scratch" / "x.txt").read_text() == "hi"


def test_get_run_log_contextvar(tmp_path):
    assert get_run_log() is None
    rl = RunLog(tmp_path / "run")
    token = _CURRENT.set(rl)
    try:
        assert get_run_log() is rl
    finally:
        _CURRENT.reset(token)
    assert get_run_log() is None


def test_decodes_via_tokenizer_when_no_text(tmp_path):
    class FakeTok:
        def decode(self, ids):
            return "DECODED:" + ",".join(str(i) for i in ids)

    rl = RunLog(tmp_path / "run")
    g = _Group([_Traj(turns=[_Turn(completion_tokens=[7, 8], logprobs=[-0.1, -0.2])], reward=1.0)])
    rl.log_training_rollouts(0, [g], tokenizer=FakeTok())
    md = (tmp_path / "run" / "02_training_rollouts" / "step_0.md").read_text()
    assert "DECODED:7,8" in md


def test_logging_never_raises_on_bad_input(tmp_path):
    rl = RunLog(tmp_path / "run")
    rl.log_training_rollouts(0, [])
    rl.log_validation_rollouts("x", [])
    rl.log_data([])
    rl.log_chat_templates([])
    rl.render_metrics()
    rl.log_validation_metrics("b", {})
    rl.write_summary()

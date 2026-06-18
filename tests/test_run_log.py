"""Tests for RunLog — the two-track (human/agent) per-run logger.

RunLog duck-types its rollout inputs, so these tests use lightweight stand-ins
instead of the real (tinker-importing) training.trajectory dataclasses — that
keeps the logger's tests free of the heavy training deps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from evsys_sdk.run_log import RunLog


@dataclass
class _Turn:
    completion_tokens: list = field(default_factory=list)
    text: str = ""


@dataclass
class _Traj:
    turns: list = field(default_factory=list)
    reward: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class _Group:
    trajectories: list = field(default_factory=list)


def _group(rewards, texts=None):
    texts = texts or [f"answer {i}" for i in range(len(rewards))]
    return _Group([_Traj(turns=[_Turn(completion_tokens=[2, 3], text=t)], reward=r)
                   for r, t in zip(rewards, texts)])


def test_creates_two_track_layout(tmp_path):
    RunLog(tmp_path / "run", experiment_name="exp", run_name="armA")
    assert (tmp_path / "run" / "human").is_dir()
    assert (tmp_path / "run" / "agent").is_dir()
    for sub in ("01_data", "02_rollouts", "03_target_tokens", "04_training", "05_benchmark"):
        assert (tmp_path / "run" / "human" / sub).is_dir()


def test_harbor_dir_lives_under_agent(tmp_path):
    rl = RunLog(tmp_path / "run")
    d = rl.harbor_dir("train")
    assert d == tmp_path / "run" / "agent" / "harbor" / "train"
    assert d.is_dir()


def test_log_data_writes_meta_and_capped_preview(tmp_path):
    rl = RunLog(tmp_path / "run")
    rows = [{"q": f"row{i}"} for i in range(20)]
    rl.log_data(rows, dataset_meta={"name": "ds", "version": 2, "format": "chat_messages"},
                n_preview=5)
    md = (tmp_path / "run" / "human" / "01_data" / "datasets.md").read_text()
    assert "rows: **20**" in md and "`ds`" in md and "chat_messages" in md
    sample = (tmp_path / "run" / "human" / "01_data" / "sample.jsonl").read_text().splitlines()
    assert len(sample) == 5
    assert json.loads(sample[0])["q"] == "row0"


def test_note_rollouts_curates_and_references_harbor_no_copy(tmp_path):
    rl = RunLog(tmp_path / "run")
    groups = [_group([0.0, 0.5, 1.0], texts=["bad", "ok", "great"])]
    rl.note_rollouts("train", groups, step=0)
    md = (tmp_path / "run" / "human" / "02_rollouts" / "rollouts.md").read_text()
    assert "reward mean" in md
    assert "great" in md and "bad" in md
    assert "agent/harbor/train/" in md
    # No rollout dump file written by us in the human track (harbor owns the dump).
    assert not (tmp_path / "run" / "human" / "02_rollouts" / "rollouts.jsonl").exists()


def test_note_rollouts_appends_across_steps(tmp_path):
    rl = RunLog(tmp_path / "run")
    rl.note_rollouts("train", [_group([1.0])], step=0)
    rl.note_rollouts("train", [_group([0.0])], step=1)
    md = (tmp_path / "run" / "human" / "02_rollouts" / "rollouts.md").read_text()
    assert "step 0" in md and "step 1" in md
    assert md.count("# Rollouts") == 1  # header written once


def test_render_training_whitelists_metrics(tmp_path):
    rl = RunLog(tmp_path / "run")
    logs = tmp_path / "run" / "logs"
    logs.mkdir(parents=True)
    with (logs / "metrics.jsonl").open("w") as f:
        f.write(json.dumps({"step": 0, "metrics": {"loss": 2.0, "internal_buf": 999, "optim/lr": 1e-4}}) + "\n")
        f.write(json.dumps({"step": 1, "metrics": {"loss": 1.0, "internal_buf": 999, "optim/lr": 1e-4}}) + "\n")
    rl.render_training(checkpoints=[{"step": 1, "label": "final", "is_final": True}])
    csv_text = (tmp_path / "run" / "human" / "04_training" / "metrics.csv").read_text()
    assert "loss" in csv_text and "lr" in csv_text
    assert "internal_buf" not in csv_text
    ck = (tmp_path / "run" / "human" / "04_training" / "checkpoints.md").read_text()
    assert "final" in ck


def test_log_eval_and_summary(tmp_path):
    rl = RunLog(tmp_path / "run", experiment_name="exp", run_name="armA", hypothesis="h")
    rl.log_eval("bench1", {"pass_rate": 0.75, "n_tasks": 4.0},
                predictions=[{"task_id": "t1", "reward": 0.0, "instruction": "do x", "expected": "y"},
                             {"task_id": "t2", "reward": 1.0, "instruction": "do z", "expected": "w"}],
                breakdowns={"difficulty": {"easy": {"mean_reward": 1.0}, "hard": {"mean_reward": 0.0}}})
    rl.write_summary(status="completed", best_metric="pass_rate", best_value=0.75,
                     conclusion="it worked")
    results = (tmp_path / "run" / "human" / "05_benchmark" / "results.md").read_text()
    assert "bench1" in results and "pass_rate" in results and "difficulty" in results
    preds = (tmp_path / "run" / "human" / "05_benchmark" / "predictions.md").read_text()
    assert "t1" in preds
    summary = (tmp_path / "run" / "human" / "summary.md").read_text()
    assert "h" in summary and "completed" in summary and "it worked" in summary
    assert "bench1" in summary


def test_decodes_via_tokenizer_when_no_text(tmp_path):
    class FakeTok:
        def decode(self, ids):
            return "DECODED:" + ",".join(str(i) for i in ids)

    rl = RunLog(tmp_path / "run")
    g = _Group([_Traj(turns=[_Turn(completion_tokens=[7, 8])], reward=1.0)])
    rl.note_rollouts("eval", [g], tokenizer=FakeTok())
    md = (tmp_path / "run" / "human" / "02_rollouts" / "rollouts.md").read_text()
    assert "DECODED:7,8" in md


def test_logging_never_raises_on_bad_input(tmp_path):
    rl = RunLog(tmp_path / "run")
    rl.note_rollouts("train", [])
    rl.log_data([])
    rl.render_training()
    rl.log_target_tokens([])
    rl.write_summary()

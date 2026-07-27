"""Local dashboard API — the on-disk mirror reshaped into the exact contract
the frontend already consumes, so its existing components render local runs."""

from __future__ import annotations

import json

from evsys_sdk.ui.local_api import LocalDashboard


def _mirror(tmp_path):
    """A mirror shaped exactly like LocalStore writes one."""
    exp_id, run_a, run_b = "exp1", "runA", "runB"
    ed = tmp_path / "experiments" / exp_id
    ed.mkdir(parents=True)
    (ed / "experiment.json").write_text(json.dumps({
        "id": exp_id, "experiment_name": "distill-0727", "hypothesis": "raise depth",
        "status": "completed", "best_score": 0.87, "_created_at": "1785000000.0",
    }))
    (ed / "groups.jsonl").write_text(json.dumps({"id": "g1", "name": "lora_rank4"}) + "\n")

    for rid, gid in ((run_a, "g1"), (run_b, None)):
        gd = tmp_path / "generations" / rid
        gd.mkdir(parents=True)
        rec = {"id": rid, "experiment_id": exp_id, "seed": "42",
               "recipe_kind": "sdft", "status": "completed"}
        if gid:
            rec["group_id"] = gid
        (gd / "generation.json").write_text(json.dumps(rec))
        (gd / "metrics.jsonl").write_text("".join(
            json.dumps({"step": s, "split": "train",
                        "metrics": {"loss": 1.0 - s / 10, "lr": 1e-4}}) + "\n"
            for s in range(3)))
        (gd / "evals.jsonl").write_text(
            json.dumps({"id": f"{rid}-e1", "step": 5,
                        "metrics": {"pass_rate": 0.5, "n_tasks": 2.0}}) + "\n")
    # rollouts, all three kinds, on run A only
    rows = []
    for kind, n in (("train", 2), ("validation", 3), ("eval", 1)):
        for i in range(n):
            rows.append({"kind": kind, "eval_id": f"{run_a}-e1" if kind != "train" else None,
                         "task_id": f"t{i}", "sample_idx": i, "step": 5,
                         "instruction": f"do {i}", "expected": "x", "reward": 1.0,
                         "completion": f"out-{kind}-{i}",
                         "completion_token_ids": [1, 2],
                         "metadata": {"latency_s": 0.5}})
    (tmp_path / "generations" / run_a / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return LocalDashboard(tmp_path)


class TestExperiments:
    def test_lists_experiments_with_run_counts(self, tmp_path):
        db = _mirror(tmp_path)
        exps = db.experiments()
        assert len(exps) == 1
        assert exps[0]["experiment_name"] == "distill-0727"
        assert exps[0]["n_runs"] == 2

    def test_empty_mirror_is_empty_not_an_error(self, tmp_path):
        assert LocalDashboard(tmp_path).experiments() == []
        assert LocalDashboard(tmp_path).experiment_detail("nope")["groups"] == []


class TestExperimentDetail:
    def test_shape_matches_the_frontend_contract(self, tmp_path):
        d = _mirror(tmp_path).experiment_detail("exp1")
        assert set(d) == {"experiment", "groups", "ungrouped_runs"}
        assert d["experiment"]["best_score"] == 0.87
        # grouped vs ungrouped split by the run's group_id
        assert [g["name"] for g in d["groups"]] == ["lora_rank4"]
        assert [r["id"] for r in d["groups"][0]["runs"]] == ["runA"]
        assert [r["id"] for r in d["ungrouped_runs"]] == ["runB"]

    def test_metrics_are_flattened_to_frontend_points(self, tmp_path):
        run = _mirror(tmp_path).experiment_detail("exp1")["groups"][0]["runs"][0]
        pts = run["metrics"]
        # {step, split, metrics:{loss, lr}} x3 steps -> 6 flat points
        assert len(pts) == 6
        assert set(pts[0]) == {"step", "split", "name", "value"}
        loss = [p for p in pts if p["name"] == "loss"]
        assert [p["step"] for p in loss] == [0, 1, 2]
        assert loss[0]["value"] == 1.0 and loss[2]["value"] == 0.8

    def test_evals_carry_step_and_metrics(self, tmp_path):
        run = _mirror(tmp_path).experiment_detail("exp1")["groups"][0]["runs"][0]
        assert run["evals"][0]["step"] == 5
        assert run["evals"][0]["metrics"]["pass_rate"] == 0.5

    def test_seed_is_a_number_again(self, tmp_path):
        """LocalStore stringifies on the way out; the UI wants the number."""
        run = _mirror(tmp_path).experiment_detail("exp1")["groups"][0]["runs"][0]
        assert run["seed"] == 42

    def test_checkpoints_are_empty_not_invented(self, tmp_path):
        """There is no local checkpoint writer — say so rather than fabricate."""
        run = _mirror(tmp_path).experiment_detail("exp1")["groups"][0]["runs"][0]
        assert run["checkpoints"] == []

    def test_rollout_counts_per_kind(self, tmp_path):
        d = _mirror(tmp_path).experiment_detail("exp1")
        run_a = d["groups"][0]["runs"][0]
        assert run_a["rollout_counts"] == {"train": 2, "validation": 3, "eval": 1}
        assert d["ungrouped_runs"][0]["rollout_counts"] == {}


class TestPredictions:
    def test_eval_predictions_returns_that_evals_rollouts(self, tmp_path):
        page = _mirror(tmp_path).eval_predictions("runA-e1")
        assert page["total"] == 4                      # 3 validation + 1 eval
        assert page["eval"]["metrics"]["pass_rate"] == 0.5
        p = page["predictions"][0]
        assert set(p) >= {"id", "run_id", "kind", "instruction", "model_output", "reward"}
        assert p["model_output"] == "out-validation-0"

    def test_paging(self, tmp_path):
        db = _mirror(tmp_path)
        page = db.eval_predictions("runA-e1", limit=2, offset=1)
        assert len(page["predictions"]) == 2
        assert page["offset"] == 1 and page["total"] == 4

    def test_unknown_eval_is_empty_not_an_error(self, tmp_path):
        page = _mirror(tmp_path).eval_predictions("nope")
        assert page["predictions"] == [] and page["total"] == 0

    def test_run_predictions_filter_by_kind(self, tmp_path):
        db = _mirror(tmp_path)
        assert len(db.run_predictions("runA")) == 6
        train = db.run_predictions("runA", kind="train")
        assert len(train) == 2
        assert all(r["kind"] == "train" for r in train)
        assert train[0]["model_output"] == "out-train-0"

    def test_token_ids_travel_in_metadata(self, tmp_path):
        p = _mirror(tmp_path).run_predictions("runA")[0]
        assert p["metadata"]["completion_token_ids"] == [1, 2]
        assert p["metadata"]["latency_s"] == 0.5


class TestResilience:
    def test_half_written_final_line_is_skipped(self, tmp_path):
        """A run appends while you look; a torn last line must not blank the
        whole panel."""
        db = _mirror(tmp_path)
        mp = tmp_path / "generations" / "runA" / "metrics.jsonl"
        mp.write_text(mp.read_text() + '{"step": 3, "split": "train", "metr')
        pts = db.experiment_detail("exp1")["groups"][0]["runs"][0]["metrics"]
        assert len(pts) == 6          # the three good rows still render

    def test_missing_files_degrade_to_empty(self, tmp_path):
        db = _mirror(tmp_path)
        (tmp_path / "generations" / "runB" / "metrics.jsonl").unlink()
        run_b = db.experiment_detail("exp1")["ungrouped_runs"][0]
        assert run_b["metrics"] == []
        assert run_b["evals"]          # unaffected


class TestAgentScoping:
    """The autoresearch view's core query: what did this agent run produce?"""

    def _stamped(self, tmp_path):
        import json as _json
        db = _mirror(tmp_path)
        # a second experiment, stamped as an autoresearch agent's work
        ed = tmp_path / "experiments" / "exp2"
        ed.mkdir(parents=True)
        (ed / "experiment.json").write_text(_json.dumps({
            "id": "exp2", "experiment_name": "agent-try-1", "status": "completed",
            "config": {"trigger": {"escalation": "escalation-000015",
                                   "agent": "autoresearch", "sandbox": "e2b"}},
            "_created_at": "1785000100.0",
        }))
        return db

    def test_filter_by_escalation(self, tmp_path):
        db = self._stamped(tmp_path)
        assert len(db.experiments()) == 2
        scoped = db.experiments(escalation="escalation-000015")
        assert [e["id"] for e in scoped] == ["exp2"]
        assert scoped[0]["trigger"]["sandbox"] == "e2b"

    def test_filter_by_agent(self, tmp_path):
        db = self._stamped(tmp_path)
        assert [e["id"] for e in db.experiments(agent="autoresearch")] == ["exp2"]
        assert db.experiments(agent="trigger") == []

    def test_unstamped_experiments_have_a_null_trigger(self, tmp_path):
        db = self._stamped(tmp_path)
        manual = next(e for e in db.experiments() if e["id"] == "exp1")
        assert manual["trigger"] is None

    def test_agent_runs_index(self, tmp_path):
        runs = self._stamped(tmp_path).agent_runs()
        assert len(runs) == 1                      # the hand-run experiment is not one
        assert runs[0]["escalation"] == "escalation-000015"
        assert runs[0]["agent"] == "autoresearch"
        assert [e["experiment_name"] for e in runs[0]["experiments"]] == ["agent-try-1"]

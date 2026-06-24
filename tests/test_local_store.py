"""LocalStore — the no-backend store that writes the ``.evsys`` mirror with the
same contract as EvsysStore — plus ``resolve_store`` mode selection."""

from __future__ import annotations

from evsys_sdk.local_store import LocalStore
from evsys_sdk.store import EvsysStore, resolve_store


def _store(tmp_path) -> LocalStore:
    return LocalStore(log_dir=str(tmp_path))


# -- write → read round-trip -------------------------------------------------


def test_experiment_run_roundtrip(tmp_path):
    s = _store(tmp_path)
    exp = s.create_experiment(experiment_name="e", tags=["x"])
    assert exp["id"] and exp["experiment_name"] == "e"
    run = s.create_run(experiment_id=exp["id"], recipe_kind="sft", seed=7)
    assert run["id"] and run["experiment_id"] == exp["id"]

    assert [e["experiment_name"] for e in s.list_experiments()] == ["e"]
    assert s.get_experiment(exp["id"])["tags"] == ["x"]
    assert [r["id"] for r in s.list_runs(experiment_id=exp["id"])] == [run["id"]]
    assert s.get_run(run["id"])["recipe_kind"] == "sft"
    # run not linked to a different experiment
    assert s.list_runs(experiment_id="other") == []


def test_metrics_long_format_and_split_filter(tmp_path):
    s = _store(tmp_path)
    run = s.create_run(experiment_id="e1")
    s.log_metrics(run_id=run["id"], step=1, metrics={"loss": 0.5})
    s.log_metrics(run_id=run["id"], step=2, metrics={"loss": 0.4})
    s.log_metrics(run_id=run["id"], step=1, metrics={"loss": 0.7}, split="val")

    rows = s.get_metrics(run["id"])
    assert all(set(r) == {"step", "name", "value", "split"} for r in rows)  # long format
    train = s.get_metrics(run["id"], split="train", name="loss")
    assert [(r["step"], r["value"]) for r in train] == [(1, 0.5), (2, 0.4)]
    assert len(s.get_metrics(run["id"], split="val")) == 1


def test_eval_gets_id_and_predictions_link(tmp_path):
    s = _store(tmp_path)
    run = s.create_run(experiment_id="e1")
    ev = s.create_eval(run_id=run["id"], benchmark_id="b", metrics={"pass_rate": 0.8})
    assert ev["id"]
    s.log_predictions(run["id"], [{"kind": "eval", "eval_id": ev["id"], "task_id": "t0", "reward": 1.0}])

    evals = s.list_evals(run["id"])
    assert evals[0]["id"] == ev["id"] and evals[0]["metrics"]["pass_rate"] == 0.8
    preds = s.list_predictions(run["id"], kind="eval")
    assert preds[0]["eval_id"] == ev["id"]
    assert s.list_predictions(run["id"], kind="rollout") == []


def test_checkpoints_and_groups(tmp_path):
    s = _store(tmp_path)
    g = s.create_group("exp1", "base")
    assert g["id"] and g["name"] == "base"
    run = s.create_run(experiment_id="exp1", group_id=g["id"])
    s.add_checkpoint(run_id=run["id"], uri="tinker://ckpt", is_final=True)
    assert s.list_checkpoints(run["id"])[0]["uri"] == "tinker://ckpt"
    assert [r["id"] for r in s.list_runs(group_id=g["id"])] == [run["id"]]


def test_add_prediction_single_row(tmp_path):
    s = _store(tmp_path)
    s.add_prediction(run_id="r1", kind="eval", task_id="t0", reward=0.5)
    rows = s.list_predictions("r1")
    assert len(rows) == 1 and rows[0]["task_id"] == "t0" and rows[0]["kind"] == "eval"


# -- resolve_store mode selection (the dashboard ↔ local switch) -------------


def test_resolve_explicit_wins():
    sentinel = object()
    assert resolve_store(sentinel) is sentinel


def test_resolve_offline_forces_local_even_with_creds(monkeypatch, tmp_path):
    monkeypatch.setenv("EVSYS_OFFLINE", "1")
    monkeypatch.setenv("EVSYS_API_KEY", "sk-x")  # offline overrides creds
    assert isinstance(resolve_store(log_dir=str(tmp_path)), LocalStore)


def test_resolve_no_creds_is_local(monkeypatch, tmp_path):
    monkeypatch.delenv("EVSYS_OFFLINE", raising=False)
    monkeypatch.delenv("EVSYS_API_KEY", raising=False)
    assert isinstance(resolve_store(log_dir=str(tmp_path)), LocalStore)


def test_resolve_creds_is_dashboard(monkeypatch):
    monkeypatch.delenv("EVSYS_OFFLINE", raising=False)
    monkeypatch.setenv("EVSYS_API_KEY", "sk-x")
    monkeypatch.setenv("EVSYS_PROJECT_ID", "p")
    assert isinstance(resolve_store(), EvsysStore)

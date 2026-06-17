"""Local UI: JSON-shaping (api.py, socket-free) + a server smoke test."""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

from evsys_sdk.local_store import LocalStore
from evsys_sdk.ui import api
from evsys_sdk.ui.server import _make_handler


def _seed(tmp_path) -> tuple[LocalStore, str, str]:
    s = LocalStore(log_dir=str(tmp_path))
    exp = s.create_experiment(experiment_name="e", tags=["x"])
    run = s.create_run(experiment_id=exp["id"], recipe_kind="sft", seed=7)
    s.log_metrics(run_id=run["id"], step=1, metrics={"loss": 0.5})
    s.log_metrics(run_id=run["id"], step=2, metrics={"loss": 0.4})
    s.log_metrics(run_id=run["id"], step=1, metrics={"loss": 0.7}, split="val")
    ev = s.create_eval(run_id=run["id"], benchmark_id="b", metrics={"pass_rate": 0.8, "cost_per_task": 0.01})
    s.log_predictions(run["id"], [{
        "kind": "eval", "eval_id": ev["id"], "task_id": "t0", "sample_idx": 0,
        "reward": 1.0, "instruction": "do it", "model_output": "done",
        "completion_token_ids": [1, 2, 3], "metadata": {"latency_s": 2.0, "prompt_tokens": 5, "completion_tokens": 2},
    }])
    return s, exp["id"], run["id"]


# -- api shaping -------------------------------------------------------------


def test_experiments_lists_with_run_count(tmp_path):
    s, exp_id, _ = _seed(tmp_path)
    rows = api.experiments(s)
    assert len(rows) == 1
    assert rows[0]["id"] == exp_id and rows[0]["name"] == "e" and rows[0]["n_runs"] == 1


def test_experiment_runs(tmp_path):
    s, exp_id, run_id = _seed(tmp_path)
    runs = api.experiment_runs(s, exp_id)
    assert [r["id"] for r in runs] == [run_id]
    assert runs[0]["recipe_kind"] == "sft"


def test_run_detail_indexes_artifacts(tmp_path):
    s, _, run_id = _seed(tmp_path)
    d = api.run_detail(s, run_id)
    assert d["metric_names"] == ["loss"]
    assert set(d["splits"]) == {"train", "val"}
    assert d["has_metrics"] and d["has_evals"] and d["has_predictions"]
    assert api.run_detail(s, "nope") is None


def test_metrics_pivoted_for_charts(tmp_path):
    s, _, run_id = _seed(tmp_path)
    m = api.metrics(s, run_id)
    assert set(m["splits"]) == {"train", "val"}
    assert m["series"]["loss"]["train"] == [[1, 0.5], [2, 0.4]]
    assert m["series"]["loss"]["val"] == [[1, 0.7]]


def test_predictions_paginated_and_slim(tmp_path):
    s, _, run_id = _seed(tmp_path)
    out = api.predictions(s, run_id, limit=10)
    assert out["total"] == 1
    p = out["predictions"][0]
    assert p["task_id"] == "t0"
    assert "completion_token_ids" not in p          # bulky field dropped
    assert p["metadata"]["latency_s"] == 2.0


# -- server smoke ------------------------------------------------------------


def test_server_serves_api_and_index(tmp_path):
    s, exp_id, _ = _seed(tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(s))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        exps = json.loads(urlopen(f"{base}/api/experiments").read())
        assert exps[0]["id"] == exp_id
        # 404 for an unknown endpoint
        from urllib.error import HTTPError
        try:
            urlopen(f"{base}/api/nope")
            assert False, "expected 404"
        except HTTPError as e:
            assert e.code == 404
        # index.html served (SPA shell)
        html = urlopen(f"{base}/").read().decode()
        assert "<title>evsys" in html
    finally:
        httpd.shutdown()
        httpd.server_close()

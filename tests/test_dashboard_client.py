"""Tests for DashboardClient + ExperimentRun.

We don't hit a real backend — everything is mocked at the requests.Session
level so the tests are fast and offline. Verifies:

  * URL construction, Bearer header, JSON body shape
  * Each public method posts to the right route
  * ExperimentRun lifecycle: enter → create_experiment + create_generation,
    clean exit → mark completed, exception → mark failed
  * Error surfaces: missing api_key, non-2xx response
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from evsys_sdk.dashboard_client import (
    DashboardClient,
    DashboardClientError,
    ExperimentRun,
    EvsysAuthError,
)


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Keep tests hermetic: no ambient creds, and mirror writes go to tmp."""
    monkeypatch.delenv("EVSYS_API_KEY", raising=False)
    monkeypatch.delenv("EVSYS_PROJECT_ID", raising=False)
    monkeypatch.delenv("EVSYS_API_URL", raising=False)
    monkeypatch.delenv("EVSYS_OFFLINE", raising=False)
    monkeypatch.setenv("EVSYS_LOG_DIR", str(tmp_path / "mirror"))


def _mock_session(response_json: dict, status: int = 200):
    """Build a mock requests.Session whose .post returns the given JSON."""
    sess = mock.MagicMock()
    resp = mock.MagicMock()
    resp.status_code = status
    resp.text = json.dumps(response_json)
    resp.json.return_value = response_json
    sess.post.return_value = resp
    return sess


def _make_client(session=None) -> DashboardClient:
    c = DashboardClient(base_url="http://test.local", api_key="sk_test", project_id="proj_test")
    if session is not None:
        c._session = session
    return c


class TestConstruction:
    def test_requires_credentials_when_online(self, monkeypatch):
        # No api_key / project_id and not offline → auth error.
        with pytest.raises(EvsysAuthError, match="EVSYS_API_KEY"):
            DashboardClient(base_url="http://x")

    def test_offline_needs_no_credentials(self):
        c = DashboardClient(base_url="http://x", offline=True)
        assert c.offline is True
        assert c._session is None

    def test_offline_via_env(self, monkeypatch):
        monkeypatch.setenv("EVSYS_OFFLINE", "true")
        c = DashboardClient()
        assert c.offline is True

    def test_picks_up_env_vars(self, monkeypatch):
        monkeypatch.setenv("EVSYS_API_KEY", "sk_from_env")
        monkeypatch.setenv("EVSYS_PROJECT_ID", "proj_from_env")
        monkeypatch.setenv("EVSYS_API_URL", "https://x.test")
        c = DashboardClient()
        assert c.api_key == "sk_from_env"
        assert c.project_id == "proj_from_env"
        assert c.base_url == "https://x.test"

    def test_strips_trailing_slash(self):
        c = DashboardClient(base_url="http://t.com/", api_key="sk", project_id="p")
        assert c.base_url == "http://t.com"

    def test_sets_bearer_header(self):
        c = DashboardClient(base_url="http://t.com", api_key="sk_abc", project_id="p")
        assert c._session.headers["Authorization"] == "Bearer sk_abc"
        assert c._session.headers["Content-Type"] == "application/json"

    def test_create_experiment_includes_project_id(self):
        sess = _mock_session({"experiment": {"id": "e1"}}, status=201)
        client = _make_client(sess)
        client.create_experiment(experiment_name="x")
        body = sess.post.call_args[1]["json"]
        assert body["project_id"] == "proj_test"


class TestExperiments:
    def test_create_experiment_post_shape(self):
        sess = _mock_session({"experiment": {"id": "e1", "experiment_name": "x"}}, status=201)
        client = _make_client(sess)
        row = client.create_experiment(
            experiment_name="sft_run_v9",
            hypothesis="LoRA r=32 raises pass@1",
            tags=["axis:lora"],
        )
        assert row["id"] == "e1"
        url, kwargs = sess.post.call_args[0][0], sess.post.call_args[1]
        assert url == "http://test.local/api/dashboard/api/sdk/experiments/"
        body = kwargs["json"]
        assert body["experiment_name"] == "sft_run_v9"
        assert body["hypothesis"] == "LoRA r=32 raises pass@1"
        assert body["tags"] == ["axis:lora"]
        # base_model / client are no longer experiment fields
        assert "base_model" not in body and "client" not in body

    def test_create_experiment_carries_reasoning_and_plan(self):
        sess = _mock_session({"experiment": {"id": "e1"}}, status=201)
        client = _make_client(sess)
        client.create_experiment(
            experiment_name="x",
            hypothesis="r=32 lifts pass@1",
            hypothesis_reasoning="r=8 plateaued while loss still dropping",
            plan="SFT 2ep, lr=1e-5, bs=16",
        )
        body = sess.post.call_args[1]["json"]
        assert body["hypothesis_reasoning"] == "r=8 plateaued while loss still dropping"
        assert body["plan"] == "SFT 2ep, lr=1e-5, bs=16"

    def test_update_experiment_accepts_conclusion(self):
        sess = _mock_session({"experiment_id": "e1", "patched": ["conclusion"]})
        client = _make_client(sess)
        client.update_experiment("e1", conclusion="r=32 confirmed; promote.")
        body = sess.post.call_args[1]["json"]
        assert body["conclusion"] == "r=32 confirmed; promote."

    def test_create_experiment_drops_none_fields(self):
        sess = _mock_session({"experiment": {"id": "e1"}}, status=201)
        client = _make_client(sess)
        client.create_experiment(experiment_name="x")
        body = sess.post.call_args[1]["json"]
        # None-valued kwargs should not be sent (so the backend uses defaults).
        for k in ("client", "hypothesis", "tags", "problem_statement_id"):
            assert k not in body

    def test_update_experiment(self):
        sess = _mock_session({"experiment_id": "e1", "patched": ["status"]})
        client = _make_client(sess)
        client.update_experiment("e1", status="completed", best_score=0.83)
        url = sess.post.call_args[0][0]
        assert url == "http://test.local/api/dashboard/api/sdk/experiments/e1/"
        body = sess.post.call_args[1]["json"]
        assert body["status"] == "completed"
        assert body["best_score"] == 0.83


class TestRuns:
    def test_create_run(self):
        sess = _mock_session({"run": {"id": "r1"}}, status=201)
        client = _make_client(sess)
        row = client.create_run(
            experiment_id="e1",
            group_id="grp1",
            seed=2,
            recipe_kind="sft",
            run_config={"model": "Qwen/Qwen3-4B", "lr": 1e-5},
            wandb_run_url="https://wandb.ai/x",
        )
        assert row["id"] == "r1"
        url = sess.post.call_args[0][0]
        body = sess.post.call_args[1]["json"]
        assert url == "http://test.local/api/dashboard/api/sdk/runs/"
        assert body["experiment_id"] == "e1"
        assert body["group_id"] == "grp1"
        assert body["seed"] == 2
        assert body["run_config"]["model"] == "Qwen/Qwen3-4B"   # base model is a hyperparam
        assert body["status"] == "pending"  # default

    def test_update_run(self):
        sess = _mock_session({"id": "r1", "patched": ["status"]})
        client = _make_client(sess)
        client.update_run("r1", status="completed", duration_seconds=12300.5)
        url = sess.post.call_args[0][0]
        assert url == "http://test.local/api/dashboard/api/sdk/runs/r1/"
        body = sess.post.call_args[1]["json"]
        assert body["status"] == "completed"
        assert body["duration_seconds"] == 12300.5


class TestLogging:
    def test_log_step_metric(self):
        sess = _mock_session({"ok": True})
        client = _make_client(sess)
        # D15: arbitrary **metrics + split; metrics nested under body["metrics"].
        client.log_step_metric("r1", step=100, loss=0.42, learning_rate=1e-5,
                               split="val", val_loss=0.55)
        url = sess.post.call_args[0][0]
        body = sess.post.call_args[1]["json"]
        assert url == "http://test.local/api/dashboard/api/sdk/runs/r1/metrics/"
        assert body["step"] == 100
        assert body["split"] == "val"
        assert body["metrics"]["loss"] == 0.42
        assert body["metrics"]["learning_rate"] == 1e-5
        assert body["metrics"]["val_loss"] == 0.55          # arbitrary named series
        # None-valued metrics should be omitted.
        assert "grad_norm" not in body["metrics"]

    def test_create_eval(self):
        sess = _mock_session({"ok": True})
        client = _make_client(sess)
        client.create_eval("r1", benchmark_id="bm1", step=500,
                           metrics={"pass_at_1": 0.83, "pass_at_3": 0.94})
        url = sess.post.call_args[0][0]
        body = sess.post.call_args[1]["json"]
        assert url == "http://test.local/api/dashboard/api/sdk/runs/r1/evals/"
        assert body["benchmark_id"] == "bm1"
        assert body["step"] == 500
        assert body["metrics"] == {"pass_at_1": 0.83, "pass_at_3": 0.94}

    def test_add_checkpoint(self):
        sess = _mock_session({"ok": True})
        client = _make_client(sess)
        client.add_checkpoint("r1", uri="tinker://ckpt/final", label="final",
                              step=100, is_final=True)
        url = sess.post.call_args[0][0]
        body = sess.post.call_args[1]["json"]
        assert url == "http://test.local/api/dashboard/api/sdk/runs/r1/checkpoints/"
        assert body["uri"] == "tinker://ckpt/final"
        assert body["is_final"] is True

    def test_log_predictions_bulk(self):
        sess = _mock_session({"ok": True, "inserted": 3})
        client = _make_client(sess)
        result = client.log_predictions("r1", [
            {"task_id": f"t{i}", "instruction": "i", "model_output": "o",
             "expected": "x", "reward": 1.0, "kind": "eval"}
            for i in range(3)
        ])
        assert result["inserted"] == 3
        url = sess.post.call_args[0][0]
        body = sess.post.call_args[1]["json"]
        assert url == "http://test.local/api/dashboard/api/sdk/runs/r1/predictions/"
        assert len(body["predictions"]) == 3

    def test_log_predictions_empty_no_request(self):
        sess = _mock_session({"ok": True})
        client = _make_client(sess)
        result = client.log_predictions("r1", [])
        assert result["inserted"] == 0
        sess.post.assert_not_called()


class TestErrors:
    def test_non_2xx_raises(self):
        sess = mock.MagicMock()
        resp = mock.MagicMock(); resp.status_code = 401; resp.text = "invalid api key"
        sess.post.return_value = resp
        client = _make_client(sess)
        with pytest.raises(DashboardClientError) as exc:
            client.create_experiment(experiment_name="x")
        assert exc.value.status == 401
        assert "invalid" in exc.value.body
        assert "/sdk/experiments/" in exc.value.path

    def test_5xx_degrades_gracefully(self):
        # A server error is treated like an outage: no raise, local fallback.
        sess = mock.MagicMock()
        resp = mock.MagicMock(); resp.status_code = 500; resp.text = "boom"
        sess.post.return_value = resp
        client = _make_client(sess)
        result = client.log_step_metric("g1", step=1, loss=0.1)
        assert result == {"ok": True}
        assert client._remote_down is True

    def test_connection_error_degrades_gracefully(self):
        import requests as _rq
        sess = mock.MagicMock()
        sess.post.side_effect = _rq.exceptions.ConnectionError("refused")
        client = _make_client(sess)
        # Should not raise; experiment still gets a (local) id.
        row = client.create_experiment(experiment_name="x")
        assert row["id"]
        assert client._remote_down is True


class TestExperimentRun:
    def test_clean_exit_marks_completed(self):
        # Each call returns a different JSON; sess.post is called many times.
        responses = [
            {"experiment": {"id": "e1"}},    # create_experiment
            {"run": {"id": "r1"}},    # create_generation
            {"ok": True},                    # log_step
            {"id": "r1"},         # update_generation (completed)
            {"experiment_id": "e1"},         # update_experiment (completed)
        ]
        sess = mock.MagicMock()
        def _post(*a, **kw):
            r = mock.MagicMock(); r.status_code = 201
            payload = responses.pop(0); r.text = json.dumps(payload); r.json.return_value = payload
            return r
        sess.post.side_effect = _post
        client = _make_client(sess)

        with ExperimentRun(client, experiment_name="x",
                           hypothesis="h", recipe_kind="sft",
                           run_config={"lr": 1e-5}) as run:
            assert run.experiment_id == "e1"
            assert run.run_id == "r1"
            run.log_step(1, loss=0.5)
            run.set_best_score(0.83)

        # Last two posts should be the completed-status updates.
        called_urls = [c[0][0] for c in sess.post.call_args_list]
        assert called_urls[-2] == "http://test.local/api/dashboard/api/sdk/runs/r1/"
        assert called_urls[-1] == "http://test.local/api/dashboard/api/sdk/experiments/e1/"
        # run completed-update carries best_score in runs.summary; the
        # experiment holds best_score directly.
        run_patch = sess.post.call_args_list[-2][1]["json"]
        assert run_patch["status"] == "completed"
        assert run_patch["summary"]["best_score"] == 0.83
        exp_patch = sess.post.call_args_list[-1][1]["json"]
        assert exp_patch["status"] == "completed"
        assert exp_patch["best_score"] == 0.83

    def test_exception_marks_failed_with_message(self):
        responses = [
            {"experiment": {"id": "e1"}},
            {"run": {"id": "r1"}},
            {"id": "r1"},   # failed update
            {"experiment_id": "e1"},   # failed update
        ]
        sess = mock.MagicMock()
        def _post(*a, **kw):
            r = mock.MagicMock(); r.status_code = 201
            payload = responses.pop(0); r.text = json.dumps(payload); r.json.return_value = payload
            return r
        sess.post.side_effect = _post
        client = _make_client(sess)

        with pytest.raises(ValueError, match="boom"):
            with ExperimentRun(client, experiment_name="x", recipe_kind="sft"):
                raise ValueError("boom")

        # Final two posts should be failed-status updates with the error message.
        gen_patch = sess.post.call_args_list[-2][1]["json"]
        assert gen_patch["status"] == "failed"
        assert "boom" in gen_patch["error_message"]
        exp_patch = sess.post.call_args_list[-1][1]["json"]
        assert exp_patch["status"] == "failed"

    def test_clean_exit_flushes_conclusion(self):
        responses = [
            {"experiment": {"id": "e1"}},
            {"run": {"id": "r1"}},
            {"id": "r1"},      # update_generation
            {"experiment_id": "e1"},      # update_experiment
        ]
        sess = mock.MagicMock()
        def _post(*a, **kw):
            r = mock.MagicMock(); r.status_code = 201
            payload = responses.pop(0); r.text = json.dumps(payload); r.json.return_value = payload
            return r
        sess.post.side_effect = _post
        client = _make_client(sess)

        with ExperimentRun(
            client, experiment_name="x", recipe_kind="sft",
            hypothesis_reasoning="prior runs plateaued at 0.71",
            plan="SFT 2ep, lr=1e-5",
        ) as run:
            run.set_best_score(0.83)
            run.set_conclusion("r=32 raised pass@1 to 0.83; promote.")

        # The exp-create call should carry the reasoning + plan.
        create_body = sess.post.call_args_list[0][1]["json"]
        assert create_body["hypothesis_reasoning"] == "prior runs plateaued at 0.71"
        assert create_body["plan"] == "SFT 2ep, lr=1e-5"

        # The final experiment patch should carry the conclusion.
        exp_patch = sess.post.call_args_list[-1][1]["json"]
        assert exp_patch["conclusion"] == "r=32 raised pass@1 to 0.83; promote."

    def test_threads_existing_experiment(self):
        responses = [
            {"run": {"id": "r1"}},     # create_generation only — no create_experiment
            {"id": "r1"},
            {"experiment_id": "e_given"},
        ]
        sess = mock.MagicMock()
        def _post(*a, **kw):
            r = mock.MagicMock(); r.status_code = 201
            payload = responses.pop(0); r.text = json.dumps(payload); r.json.return_value = payload
            return r
        sess.post.side_effect = _post
        client = _make_client(sess)

        with ExperimentRun(client, experiment_name="x", recipe_kind="sft",
                           experiment_id="e_given") as run:
            assert run.experiment_id == "e_given"

        # First call should be the run, NOT an experiment create.
        first_url = sess.post.call_args_list[0][0][0]
        assert "/sdk/runs/" in first_url
        assert "/sdk/experiments/" not in first_url

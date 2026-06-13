"""EvsysStore request-shape tests (mocked gateway).

The SDK no longer talks to Supabase directly — every call routes through the
backend gateway (`/api/dashboard/api/sdk/data/`) with a Bearer API key. These
tests mock the HTTP layer and assert the client posts the right `{op, args}`
and threads results. (A full live round-trip is a backend integration test:
it needs the running backend + a real API key + project membership.)
"""

from __future__ import annotations

from unittest import mock

import pytest

from evsys_sdk import EvsysStore, EvsysStoreError


def _client(monkeypatch, result=None, capture=None):
    """EvsysStore whose requests.post returns {"result": result} and, if
    given, appends each posted body to `capture`."""
    def fake_post(url, headers=None, json=None, timeout=None):
        if capture is not None:
            capture.append({"url": url, "headers": headers, "json": json})
        r = mock.MagicMock()
        r.status_code = 200
        r.json.return_value = {"result": result}
        return r
    monkeypatch.setattr("evsys_sdk.store.requests.post", fake_post)
    return EvsysStore(base_url="http://test.local", api_key="sk_test", project_id="p1")


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("EVSYS_API_KEY", raising=False)
    with pytest.raises(EvsysStoreError):
        EvsysStore(base_url="http://test.local", api_key=None)


def test_posts_to_gateway_with_bearer(monkeypatch):
    cap: list = []
    store = _client(monkeypatch, result={"id": "g1", "version": 1}, capture=cap)
    out = store.set_goal("beat baseline")
    assert out == {"id": "g1", "version": 1}
    call = cap[-1]
    assert call["url"] == "http://test.local/api/dashboard/api/sdk/data/"
    assert call["headers"]["Authorization"].startswith("Bearer sk_test")
    assert call["json"]["op"] == "set_goal"
    assert call["json"]["args"] == {"project_id": "p1", "goal": "beat baseline"}


def test_create_experiment_defaults_project_and_omits_user_id(monkeypatch):
    cap: list = []
    store = _client(monkeypatch, result={"id": "e1"}, capture=cap)
    store.create_experiment(experiment_name="qwen_4b_vs_9b",
                            hypothesis="4B ≥ 9B on the benchmark")
    args = cap[-1]["json"]["args"]
    assert cap[-1]["json"]["op"] == "create_experiment"
    assert args["project_id"] == "p1"                       # defaulted from store
    assert args["experiment_name"] == "qwen_4b_vs_9b"
    assert args["hypothesis"] == "4B ≥ 9B on the benchmark"
    assert "user_id" not in args                            # backend sets it from API key
    assert "project_goal_id" not in args and "tags" not in args  # None dropped


def test_run_lifecycle_ops(monkeypatch):
    cap: list = []
    store = _client(monkeypatch, result={"id": "r1", "seed": 1}, capture=cap)

    store.create_run(experiment_id="e1", group_id="grp1", seed=1,
                     recipe_kind="sft", run_config={"model": "Qwen/Qwen3-4B"})
    assert cap[-1]["json"]["op"] == "create_run"
    assert cap[-1]["json"]["args"]["run_config"]["model"] == "Qwen/Qwen3-4B"
    assert cap[-1]["json"]["args"]["seed"] == 1             # 0/falsy-safe (None-only drop)

    store.log_metrics(run_id="r1", step=10, split="val", metrics={"val_loss": 1.45})
    assert cap[-1]["json"]["op"] == "log_metrics"
    assert cap[-1]["json"]["args"] == {"run_id": "r1", "step": 10, "split": "val",
                                       "metrics": {"val_loss": 1.45}}

    store.add_checkpoint(run_id="r1", uri="tinker://final", label="final",
                         step=100, is_final=True)
    assert cap[-1]["json"]["args"]["is_final"] is True      # False would also be kept

    store.create_eval(run_id="r1", benchmark_id="bm1", metrics={"pass_at_1": 0.83})
    assert cap[-1]["json"]["op"] == "create_eval"
    assert cap[-1]["json"]["args"]["metrics"] == {"pass_at_1": 0.83}


def test_update_and_invalidate_use_patch(monkeypatch):
    cap: list = []
    store = _client(monkeypatch, result={}, capture=cap)
    store.set_conclusion("e1", "4B matched 9B at half cost")
    assert cap[-1]["json"]["op"] == "update_experiment"
    assert cap[-1]["json"]["args"]["patch"]["conclusion"].startswith("4B matched")
    store.invalidate_experiment("e1", reason="seed leakage")
    patch = cap[-1]["json"]["args"]["patch"]
    assert patch["is_valid"] is False and patch["error_message"] == "seed leakage"


def test_agent_reads(monkeypatch):
    cap: list = []
    store = _client(monkeypatch, result=[], capture=cap)
    store.experiment_summaries()                            # default project
    assert cap[-1]["json"] == {"op": "experiment_summaries",
                               "args": {"project_id": "p1", "valid_only": False}}
    store.experiment_detail("e1", include_metrics=True)
    assert cap[-1]["json"]["op"] == "experiment_detail"
    assert cap[-1]["json"]["args"] == {"experiment_id": "e1", "include_metrics": True}


def test_gateway_error_raises(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        r = mock.MagicMock(); r.status_code = 403; r.text = "not a member of this project"
        return r
    monkeypatch.setattr("evsys_sdk.store.requests.post", fake_post)
    store = EvsysStore(base_url="http://test.local", api_key="sk_test", project_id="p1")
    with pytest.raises(EvsysStoreError) as ei:
        store.list_experiments()
    assert ei.value.status == 403

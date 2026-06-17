"""Fireworks deployer + the post-training deploy hook.

The external bits (tinker weight download, firectl CLI) are monkeypatched via
the module-level seams, so these run without tinker_cookbook or firectl.
"""

from __future__ import annotations

import pytest

from evsys_sdk.config import ExperimentConfig
from evsys_sdk.deploy import build_deployer, deploy_checkpoint
from evsys_sdk.experiment import Experiment
from evsys_sdk.protocols import RunResult
from evsys_sdk.registry import list_deployers, register_deployer


@pytest.fixture
def fw_recorder(monkeypatch):
    """Mock the IO seams: weight download, model upload, deployment create."""
    calls: dict[str, list] = {"upload": [], "deploy": []}

    def fake_upload(*, account_id, model_id, local_dir, kind, base_model, api_key):
        calls["upload"].append(
            {"account_id": account_id, "model_id": model_id, "kind": kind,
             "base_model": base_model})
        return f"accounts/{account_id}/models/{model_id}"

    def fake_create_deployment(*, model_ref, account_id, api_key, params):
        calls["deploy"].append({"model_ref": model_ref, "params": params})
        return f"{model_ref}#deployment"

    monkeypatch.setattr("evsys_sdk.deploy.fireworks._download_weights", lambda uri, out: out)
    monkeypatch.setattr("evsys_sdk.deploy.fireworks._upload_model", fake_upload)
    monkeypatch.setattr("evsys_sdk.deploy.fireworks._create_deployment", fake_create_deployment)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    return calls


def test_fireworks_registered():
    assert "fireworks" in list_deployers()


def test_lora_deploy_uploads_then_deploys(fw_recorder):
    dep = build_deployer({"kind": "fireworks", "params": {
        "account_id": "acct",
        "base_model": "accounts/fireworks/models/qwen3-4b",
    }})
    res = dep.deploy("tinker://run123/sampler_weights/final")

    assert res.provider == "fireworks"
    assert res.model_ref.startswith("accounts/acct/models/")
    assert res.deployed is True
    assert res.deployment_id is not None
    assert res.endpoint == "https://api.fireworks.ai/inference/v1"

    up = fw_recorder["upload"][0]
    assert up["kind"] == "HF_PEFT_ADDON"
    assert up["base_model"] == "accounts/fireworks/models/qwen3-4b"
    assert fw_recorder["deploy"][0]["model_ref"] == res.model_ref


def test_upload_only_when_create_deployment_false(fw_recorder):
    dep = build_deployer({"kind": "fireworks", "params": {
        "account_id": "a", "base_model": "b", "create_deployment": False,
    }})
    res = dep.deploy("tinker://x")
    assert res.deployed is False and res.deployment_id is None
    assert fw_recorder["deploy"] == []


def test_merged_form_uploads_base_model(fw_recorder, monkeypatch):
    monkeypatch.setattr("evsys_sdk.deploy.fireworks._build_hf", lambda **kw: None)
    dep = build_deployer({"kind": "fireworks", "params": {
        "account_id": "a", "upload_form": "merged", "merged_base_model": "Qwen/Qwen3-4B",
    }})
    dep.deploy("tinker://x")
    assert fw_recorder["upload"][0]["kind"] == "HF_BASE_MODEL"


def test_lora_requires_base_model(fw_recorder):
    dep = build_deployer({"kind": "fireworks", "params": {"account_id": "a"}})
    with pytest.raises(RuntimeError, match="base_model"):
        dep.deploy("tinker://x")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    dep = build_deployer({"kind": "fireworks", "params": {"account_id": "a", "base_model": "b"}})
    with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
        dep.deploy("tinker://x")


def test_deploy_checkpoint_standalone(fw_recorder):
    res = deploy_checkpoint(
        "fireworks", {"account_id": "acct"}, "tinker://x",
        base_model="accounts/fireworks/models/qwen3-4b",
    )
    assert res.deployed is True
    assert fw_recorder["upload"][0]["account_id"] == "acct"


# --- inline hook in Experiment.run -----------------------------------------


@register_deployer("recording_deployer")
class _RecordingDeployer:
    name = "recording_deployer"
    Config = None  # build_deployer falls back to dict(params)

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def deploy(self, checkpoint_uri, *, base_model=None, model_id=None):
        from evsys_sdk.deploy import DeployResult
        _RecordingDeployer.last_checkpoint = checkpoint_uri
        return DeployResult(provider="rec", model_ref="accounts/x/models/y", deployed=True)


def test_experiment_deploys_best_arm(tmp_path):
    def fake_train(cfg):
        return [RunResult(
            run_id="r", status="completed", metrics={"loss": 0.1},
            artifacts={"checkpoint-final": "tinker://trained/sampler/final"},
        )]

    cfg = ExperimentConfig(
        name="dep", output_dir=str(tmp_path / "out"),
        run={
            "name": "base",
            "data": {"source_kind": "in_memory", "rows": [{"a": 1}],
                     "transforms": [{"kind": "identity"}]},
            "model": {"name": "m"},
            "algorithm": {"kind": "mock_sft"},
            "backend": {"kind": "mock"},
            "eval": {"enabled": False},
        },
        deploy={"kind": "recording_deployer", "params": {}},
    )
    res = Experiment(cfg, train_fn=fake_train).run()

    assert res.deploy_result is not None
    assert res.deploy_result.deployed is True
    assert _RecordingDeployer.last_checkpoint == "tinker://trained/sampler/final"

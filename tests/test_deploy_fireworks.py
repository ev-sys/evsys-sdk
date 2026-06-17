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
def firectl_recorder(monkeypatch):
    calls: list[list[str]] = []

    def fake_firectl(args, *, firectl_path, env):
        calls.append(list(args))
        return ""

    monkeypatch.setattr("evsys_sdk.deploy.fireworks._firectl", fake_firectl)
    monkeypatch.setattr(
        "evsys_sdk.deploy.fireworks._download_weights",
        lambda uri, out: out,  # pretend the adapter landed at `out`
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    return calls


def test_fireworks_registered():
    assert "fireworks" in list_deployers()


def test_lora_deploy_uploads_then_deploys(firectl_recorder):
    dep = build_deployer({"kind": "fireworks", "params": {
        "account_id": "acct",
        "base_model": "accounts/fireworks/models/qwen3-4b",
    }})
    res = dep.deploy("tinker://run123/sampler_weights/final")

    assert res.provider == "fireworks"
    assert res.model_ref.startswith("accounts/acct/models/")
    assert res.deployed is True
    assert res.endpoint == "https://api.fireworks.ai/inference/v1"

    model_create = next(c for c in firectl_recorder if c[:2] == ["model", "create"])
    assert "--base-model" in model_create
    assert "accounts/fireworks/models/qwen3-4b" in model_create
    dep_create = next(c for c in firectl_recorder if c[:2] == ["deployment", "create"])
    assert "--wait" in dep_create


def test_upload_only_when_create_deployment_false(firectl_recorder):
    dep = build_deployer({"kind": "fireworks", "params": {
        "account_id": "a", "base_model": "b", "create_deployment": False,
    }})
    res = dep.deploy("tinker://x")
    assert res.deployed is False and res.deployment_id is None
    assert not any(c[:2] == ["deployment", "create"] for c in firectl_recorder)


def test_lora_requires_base_model(firectl_recorder):
    dep = build_deployer({"kind": "fireworks", "params": {"account_id": "a"}})
    with pytest.raises(RuntimeError, match="base_model"):
        dep.deploy("tinker://x")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    dep = build_deployer({"kind": "fireworks", "params": {"account_id": "a", "base_model": "b"}})
    with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
        dep.deploy("tinker://x")


def test_deploy_checkpoint_standalone(firectl_recorder):
    res = deploy_checkpoint(
        "fireworks", {"account_id": "acct"}, "tinker://x",
        base_model="accounts/fireworks/models/qwen3-4b",
    )
    assert res.deployed is True
    assert any(c[:2] == ["model", "create"] for c in firectl_recorder)


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

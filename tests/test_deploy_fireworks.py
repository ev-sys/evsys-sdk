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

    def fake_upload(*, account_id, model_id, local_dir, kind, base_model, api_key,
                    display_name=None):
        calls["upload"].append(
            {"account_id": account_id, "model_id": model_id, "kind": kind,
             "base_model": base_model, "display_name": display_name})
        return f"accounts/{account_id}/models/{model_id}"

    def fake_create_deployment(*, model_ref, base_model=None, account_id, api_key, params):
        calls["deploy"].append(
            {"model_ref": model_ref, "base_model": base_model, "params": params})
        return f"accounts/{account_id}/deployments/dep"

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


# --- payload shape + adapter prep (the live-API fixes) ---------------------


def test_peft_addon_create_payload_has_required_fields_and_no_file_list():
    """HF_PEFT_ADDON create must carry displayName + peftDetails{r,targetModules}
    and must NOT include a top-level file list (files go via the upload endpoint)."""
    from evsys_sdk.deploy.fireworks import _model_data

    data = _model_data(
        kind="HF_PEFT_ADDON",
        base_model="accounts/fireworks/models/llama-v3p2-3b",
        files=["adapter_model.safetensors", "adapter_config.json"],
        peft={"baseModel": "accounts/fireworks/models/llama-v3p2-3b",
              "r": 8, "targetModules": ["q_proj"], "baseModelType": "llama"},
        display_name="dn",
    )
    assert data["kind"] == "HF_PEFT_ADDON"
    assert "huggingFaceFiles" not in data          # the field Fireworks rejects
    assert data["displayName"] == "dn"
    assert data["peftDetails"]["r"] == 8
    assert data["peftDetails"]["targetModules"] == ["q_proj"]
    assert data["peftDetails"]["baseModelType"] == "llama"


def test_base_model_create_payload_keeps_file_list():
    from evsys_sdk.deploy.fireworks import _model_data

    data = _model_data(kind="HF_BASE_MODEL", base_model=None,
                       files=["config.json", "model.safetensors"], display_name="m")
    assert data["kind"] == "HF_BASE_MODEL"
    assert data["baseModelDetails"]["huggingfaceFiles"] == ["config.json", "model.safetensors"]
    assert data["baseModelDetails"]["checkpointFormat"] == "HUGGINGFACE"


def test_prune_lora_adapter_strips_lm_head(tmp_path):
    """tinker trains all-linear LoRA (incl. lm_head); tied-embedding bases reject
    it. The deployer must drop those tensors + target_modules before upload."""
    torch = pytest.importorskip("torch")
    import json

    from safetensors.torch import load_file, save_file

    from evsys_sdk.deploy.fireworks import _prune_lora_adapter

    save_file({
        "base_model.model.layers.0.q_proj.lora_A.weight": torch.zeros(2, 2),
        "base_model.model.lm_head.lora_A.weight": torch.zeros(2, 2),
        "base_model.model.lm_head.lora_B.weight": torch.zeros(2, 2),
    }, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": 8, "target_modules": "all-linear"}))

    removed = _prune_lora_adapter(str(tmp_path), ["lm_head", "embed_tokens"])

    assert removed and all("lm_head" in k for k in removed)
    kept = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert not any("lm_head" in k for k in kept)
    assert any("q_proj" in k for k in kept)
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert "lm_head" not in cfg["target_modules"] and "q_proj" in cfg["target_modules"]


class _Resp:
    def __init__(self, body):
        import json as _json
        self.status_code = 200
        self.text = ""
        self._body = body
        self.content = _json.dumps(body).encode()

    def json(self):
        return self._body

    def raise_for_status(self):
        pass


def test_create_deployment_merged_posts_unwrapped_resource(monkeypatch):
    """Merged/full model (no base_model): the Deployment resource is posted
    DIRECTLY (not under a 'deployment' key), baseModel == the model, no attach."""
    import requests

    from evsys_sdk.deploy.fireworks import _create_deployment

    posts: list = []
    monkeypatch.setattr(requests, "post", lambda url, headers=None, json=None:
                        posts.append((url, json)) or _Resp({"name": "accounts/a/deployments/d"}))

    ref = _create_deployment(model_ref="accounts/a/models/m", base_model=None,
                             account_id="a", api_key="k", params={})
    assert ref == "accounts/a/deployments/d"
    assert len(posts) == 1                       # no deployedModels attach
    url, payload = posts[0]
    assert "deployment" not in payload           # the wrapper bug
    assert payload["baseModel"] == "accounts/a/models/m"
    assert payload["acceleratorType"] == "NVIDIA_H100_80GB"


def test_create_deployment_addon_waits_then_attaches(monkeypatch):
    """LoRA addon: deploy the BASE (enableAddons), wait for READY, then attach
    the addon as a deployedModel (attaching before READY is rejected)."""
    import requests

    from evsys_sdk.deploy.fireworks import _create_deployment

    posts: list = []
    monkeypatch.setattr(requests, "post", lambda url, headers=None, json=None:
                        posts.append((url, json)) or _Resp({"name": "accounts/a/deployments/d"}))
    # Base deployment polls READY immediately.
    monkeypatch.setattr(requests, "get",
                        lambda url, headers=None: _Resp({"state": "READY"}))

    ref = _create_deployment(model_ref="accounts/a/models/addon",
                             base_model="accounts/fireworks/models/llama-v3p2-3b",
                             account_id="a", api_key="k", params={})
    assert ref == "accounts/a/deployments/d"
    dep_url, dep_payload = posts[0]
    assert dep_payload["baseModel"] == "accounts/fireworks/models/llama-v3p2-3b"
    assert dep_payload["enableAddons"] is True
    attach_url, attach_payload = posts[1]
    assert attach_url.endswith("/deployedModels")
    assert attach_payload == {"model": "accounts/a/models/addon",
                              "deployment": "accounts/a/deployments/d"}


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

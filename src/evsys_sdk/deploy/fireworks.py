"""Fireworks deployer — upload a trained checkpoint and stand up a deployment
via the Fireworks **REST API** (plain ``requests``, a core dependency).

No firectl binary and no ``fireworks-ai`` package: the SDK's own deploy path is
just these HTTP endpoints, and the heavy ``fireworks-ai`` dependency conflicts
with harbor (betterproto/grpc), so we call the API directly. Nothing extra to
install beyond the normal SDK.

Tinker training is always LoRA, so the default path downloads just the LoRA
adapter from the checkpoint and uploads it as a Fireworks ``HF_PEFT_ADDON`` on a
base model, then creates the (dedicated) deployment. Uploaded LoRA addons can
only run on on-demand deployments — so "deploy" means a live, billed GPU
deployment.

The external/IO bits sit behind module-level seams (:func:`_download_weights`,
:func:`_upload_model`, :func:`_create_deployment`) so the orchestration is unit
testable without network/``tinker_cookbook`` and without ever standing up a real
(billed) deployment. The weight download needs ``tinker-cookbook`` (the
``tinker`` extra), which a tinker checkpoint requires anyway. Auth: set
``FIREWORKS_API_KEY``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..registry import register_deployer

logger = logging.getLogger(__name__)

FIREWORKS_ENDPOINT = "https://api.fireworks.ai/inference/v1"
FIREWORKS_API_BASE = "https://api.fireworks.ai/v1"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class FireworksDeployConfig(BaseModel):
    """Params for the ``fireworks`` deployer."""

    model_config = ConfigDict(extra="forbid")

    account_id: str
    """Fireworks account id (the ``<acct>`` in ``accounts/<acct>/...``)."""
    api_key_env: str = "FIREWORKS_API_KEY"
    """Env var holding the Fireworks API key (the SDK reads ``FIREWORKS_API_KEY``)."""

    base_model: str | None = None
    """For ``upload_form='lora'``: the Fireworks base model the adapter attaches
    to, e.g. ``accounts/fireworks/models/qwen3-4b``. May be overridden per call."""
    model_id: str | None = None
    """Target Fireworks model id. Defaults to a slug derived from the checkpoint."""
    display_name: str | None = None
    """Human-readable model name (Fireworks requires one). Defaults to the model id."""

    upload_form: str = "lora"
    """``lora`` (default — upload the LoRA adapter as an HF_PEFT_ADDON) or
    ``merged`` (merge into the base and upload a full HF model; needs a GPU +
    ``merged_base_model``)."""
    merged_base_model: str | None = None
    """HF base model name for ``upload_form='merged'`` (passed to build_hf_model)."""

    create_deployment: bool = True
    """Also create a live (dedicated) deployment. False = upload to READY only."""
    deployment_params: dict[str, Any] = Field(default_factory=dict)
    """Extra fields merged into the deployment create payload (accelerator, etc.)."""

    exclude_modules: list[str] = Field(default_factory=lambda: ["lm_head", "embed_tokens"])
    """LoRA target modules to strip from the adapter before upload (tensors +
    ``target_modules``). Defaults to ``lm_head``/``embed_tokens``: tinker trains
    ``all-linear`` LoRA, but bases that tie embeddings (e.g. Llama-3.2) have no
    standalone ``lm_head.weight`` and Fireworks rejects LoRA keys referencing it."""

    generation_defaults: dict[str, Any] | None = None
    """If set (LoRA only), written as ``fireworks.json`` ``defaults`` alongside
    the adapter (stop/max_tokens/temperature/...)."""

    endpoint: str = FIREWORKS_ENDPOINT
    work_dir: str | None = None
    """Where to stage downloaded weights. Defaults to a temp dir."""


# ---------------------------------------------------------------------------
# Seams (monkeypatched in tests)
# ---------------------------------------------------------------------------


def _download_weights(checkpoint_uri: str, output_dir: str) -> str:
    """Download a tinker checkpoint to a local HF/adapter directory.
    Returns the local directory. Lazy-imports the cookbook weights helper."""
    from tinker_cookbook.weights import download  # type: ignore

    return download(tinker_path=checkpoint_uri, output_dir=output_dir)


def _build_hf(*, base_model: str, adapter_path: str, output_path: str) -> None:
    """Merge a LoRA adapter into its base → a full HF model directory.
    Lazy-imports the cookbook export helper (needs torch + compute)."""
    from tinker_cookbook.weights import build_hf_model  # type: ignore

    build_hf_model(base_model=base_model, adapter_path=adapter_path, output_path=output_path)


def _upload_model(
    *, account_id: str, model_id: str, local_dir: str,
    kind: str, base_model: str | None, api_key: str,
    display_name: str | None = None,
) -> str:
    """Create the Fireworks model, upload the local files, and validate to READY
    via the documented REST pipeline (``requests``). Returns the model ref.

    ``kind`` is ``HF_PEFT_ADDON`` (LoRA) or ``HF_BASE_MODEL`` (merged)."""
    import time

    import requests

    base = FIREWORKS_API_BASE
    auth = {"Authorization": f"Bearer {api_key}"}
    json_h = {**auth, "Content-Type": "application/json"}
    # Only upload real model files — drop tinker's bookkeeping markers (e.g. the
    # zero-byte ``checkpoint_complete``), which aren't HF artifacts.
    files = [
        f for f in os.listdir(local_dir)
        if os.path.isfile(os.path.join(local_dir, f)) and f not in _NON_MODEL_FILES
    ]
    sizes = {f: os.path.getsize(os.path.join(local_dir, f)) for f in files}

    # For a LoRA addon, Fireworks requires r + target_modules at create time;
    # read them from the adapter's own config and resolve the base model type.
    peft = (
        _peft_details(local_dir, base_model=base_model, api_key=api_key)
        if kind == "HF_PEFT_ADDON" else None
    )

    # 1. Create the model (idempotent — ALREADY_EXISTS is fine on retry).
    r = requests.post(
        f"{base}/accounts/{account_id}/models", headers=json_h,
        json={
            "modelId": model_id,
            "model": _model_data(
                kind=kind, base_model=base_model, files=files,
                peft=peft, display_name=display_name or model_id,
            ),
        },
    )
    if r.status_code not in (200, 201) and "ALREADY_EXISTS" not in r.text:
        r.raise_for_status()

    # 2. Get a signed URL per file, then 3. upload each.
    ep = requests.post(
        f"{base}/accounts/{account_id}/models/{model_id}:getUploadEndpoint",
        headers=json_h, json={"filenameToSize": sizes},
    )
    ep.raise_for_status()
    for fn, url in ep.json()["filenameToSignedUrls"].items():
        with open(os.path.join(local_dir, fn), "rb") as fh:
            up = requests.put(
                url, data=fh.read(),
                headers={
                    "Content-Type": "application/octet-stream",
                    "x-goog-content-length-range": f"{sizes[fn]},{sizes[fn]}",
                },
            )
            up.raise_for_status()

    # 4. Validate — poll until READY (FAILED_PRECONDITION = files still landing).
    validate_url = f"{base}/accounts/{account_id}/models/{model_id}:validateUpload"
    for _ in range(60):
        v = requests.get(validate_url, headers=auth)
        if v.status_code == 200:
            break
        if "FAILED_PRECONDITION" in v.text:
            time.sleep(10)
            continue
        v.raise_for_status()
    else:
        raise RuntimeError(f"model {model_id} did not reach READY in time")

    return f"accounts/{account_id}/models/{model_id}"


# Defaults for a small dedicated deployment (overridable via deployment_params).
_DEPLOYMENT_DEFAULTS: dict[str, Any] = {
    "acceleratorType": "NVIDIA_H100_80GB",
    "acceleratorCount": 1,
    "minReplicaCount": 1,
    "maxReplicaCount": 1,
}


def _wait_deployment_ready(
    *, account_id: str, deployment_id: str, api_key: str,
    tries: int = 80, delay: float = 15.0,
) -> None:
    """Poll a deployment until it reaches READY (raise on FAILED / timeout)."""
    import time

    import requests

    url = f"{FIREWORKS_API_BASE}/accounts/{account_id}/deployments/{deployment_id}"
    auth = {"Authorization": f"Bearer {api_key}"}
    for _ in range(tries):
        r = requests.get(url, headers=auth)
        if r.status_code == 200:
            state = r.json().get("state")
            if state == "READY":
                return
            if state == "FAILED":
                raise RuntimeError(f"deployment {deployment_id} reached FAILED state")
        time.sleep(delay)
    raise RuntimeError(f"deployment {deployment_id} not READY after {tries} polls")


def _create_deployment(
    *, model_ref: str, base_model: str | None, account_id: str, api_key: str,
    params: dict[str, Any],
) -> str:
    """Stand up a dedicated deployment and return its ref. The Deployment resource
    is posted DIRECTLY (the ``{"deployment": ...}`` wrapper is rejected).

    A merged/full upload (``base_model`` falsy, or == ``model_ref``) deploys on
    itself. A LoRA addon deploys on its BASE (addons enabled), then attaches as a
    ``deployedModel`` — which Fireworks only allows once the base deployment is
    READY (attaching while CREATING returns "deployment is in state CREATING")."""
    import requests

    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    is_addon = bool(base_model) and base_model != model_ref
    deploy_base = base_model if is_addon else model_ref

    payload = {"baseModel": deploy_base, **_DEPLOYMENT_DEFAULTS, **params}
    if is_addon:
        payload.setdefault("enableAddons", True)
    r = requests.post(
        f"{FIREWORKS_API_BASE}/accounts/{account_id}/deployments", headers=h, json=payload,
    )
    r.raise_for_status()
    body = r.json() if r.content else {}
    dep_ref = body.get("name", "") if isinstance(body, dict) else ""

    if is_addon and dep_ref:
        # Attach the addon — only possible once the base deployment is READY.
        _wait_deployment_ready(
            account_id=account_id, deployment_id=dep_ref.split("/")[-1], api_key=api_key,
        )
        a = requests.post(
            f"{FIREWORKS_API_BASE}/accounts/{account_id}/deployedModels", headers=h,
            json={"model": model_ref, "deployment": dep_ref},
        )
        if a.status_code not in (200, 201) and "ALREADY_EXISTS" not in a.text:
            a.raise_for_status()
    return dep_ref or model_ref


# Tinker checkpoint bookkeeping files that are not part of the HF model upload.
_NON_MODEL_FILES = frozenset({"checkpoint_complete"})

# LoRA target modules to register when the adapter config uses the PEFT shorthand
# ``"all-linear"`` (Fireworks needs the explicit attention + MLP projections).
_ALL_LINEAR_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def _prune_lora_adapter(adapter_dir: str, exclude_modules: list[str]) -> list[str]:
    """Strip LoRA tensors targeting ``exclude_modules`` from the adapter (and from
    ``adapter_config.json`` ``target_modules``), rewriting both in place. Returns
    the removed tensor keys. Lets tinker ``all-linear`` adapters deploy onto bases
    that tie embeddings (no standalone ``lm_head``/``embed_tokens`` to attach to)."""
    if not exclude_modules:
        return []
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    path = Path(adapter_dir) / "adapter_model.safetensors"
    if not path.exists():
        return []
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()
    tensors = load_file(str(path))
    removed = [k for k in tensors if any(f".{m}." in k or f"{m}." in k for m in exclude_modules)]
    if not removed:
        return []
    for k in removed:
        del tensors[k]
    save_file(tensors, str(path), metadata=metadata)

    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        tm = cfg.get("target_modules")
        tm = tm if isinstance(tm, list) else list(_ALL_LINEAR_TARGETS)
        cfg["target_modules"] = [m for m in tm if m not in exclude_modules]
        cfg_path.write_text(json.dumps(cfg, indent=2))
    return removed


def _peft_details(
    local_dir: str, *, base_model: str | None, api_key: str,
) -> dict[str, Any]:
    """Resolve the ``peftDetails`` Fireworks requires for an HF_PEFT_ADDON create:
    ``r`` and ``targetModules`` come from the adapter's own ``adapter_config.json``;
    ``baseModelType`` is read off the Fireworks base model (best-effort)."""
    cfg: dict[str, Any] = {}
    cfg_path = Path(local_dir) / "adapter_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())

    targets = cfg.get("target_modules")
    if not isinstance(targets, list):  # e.g. the "all-linear" shorthand
        targets = list(_ALL_LINEAR_TARGETS)

    details: dict[str, Any] = {
        "baseModel": base_model,
        "r": int(cfg.get("r", 8)),
        "targetModules": targets,
    }
    model_type = _base_model_type(base_model, api_key) if base_model else None
    if model_type:
        details["baseModelType"] = model_type
    return details


def _base_model_type(base_model: str, api_key: str) -> str | None:
    """GET the Fireworks base model and return its ``modelType`` (e.g. ``llama``).
    Best-effort: returns None if it can't be fetched."""
    import requests

    acct, _, name = base_model.removeprefix("accounts/").partition("/models/")
    try:
        r = requests.get(
            f"{FIREWORKS_API_BASE}/accounts/{acct}/models/{name}",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if r.status_code == 200:
            return r.json().get("baseModelDetails", {}).get("modelType") or None
    except Exception:
        logger.debug("could not fetch base model type for %s", base_model, exc_info=True)
    return None


def _model_data(
    *, kind: str, base_model: str | None, files: list[str],
    peft: dict[str, Any] | None = None, display_name: str | None = None,
) -> dict[str, Any]:
    """Build the ``model`` payload for create_model. The uploaded file list is NOT
    part of this payload — files go through the separate ``getUploadEndpoint``."""
    if kind == "HF_PEFT_ADDON":
        return {
            "kind": "HF_PEFT_ADDON",
            "displayName": display_name or "",
            "peftDetails": peft or {"baseModel": base_model},
        }
    return {
        "kind": "HF_BASE_MODEL",
        "displayName": display_name or "",
        "baseModelDetails": {
            "checkpointFormat": "HUGGINGFACE",
            "worldSize": 1,
            "huggingfaceFiles": files,
        },
    }


# ---------------------------------------------------------------------------
# Deployer
# ---------------------------------------------------------------------------


def _default_model_id(checkpoint_uri: str) -> str:
    """Deterministic, Fireworks-safe model id derived from the checkpoint URI."""
    slug = re.sub(r"[^a-z0-9]+", "-", checkpoint_uri.lower()).strip("-")
    return f"evsys-{slug[-40:].strip('-')}" if slug else "evsys-model"


@register_deployer("fireworks")
class FireworksDeployer:
    name: ClassVar[str] = "fireworks"
    Config: ClassVar[type] = FireworksDeployConfig

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = FireworksDeployConfig.model_validate(kwargs)

    def _api_key(self) -> str:
        key = os.environ.get(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(
                f"{self.cfg.api_key_env} not set — needed to authenticate with Fireworks."
            )
        return key

    def deploy(
        self, checkpoint_uri: str, *,
        base_model: str | None = None, model_id: str | None = None,
    ):
        from . import DeployResult

        acct = self.cfg.account_id
        mid = model_id or self.cfg.model_id or _default_model_id(checkpoint_uri)
        api_key = self._api_key()
        work = Path(self.cfg.work_dir) if self.cfg.work_dir else Path(
            tempfile.mkdtemp(prefix="evsys_deploy_")
        )

        if self.cfg.upload_form == "merged":
            if not self.cfg.merged_base_model:
                raise RuntimeError("upload_form='merged' requires merged_base_model")
            adapter_dir = _download_weights(checkpoint_uri, str(work / "adapter"))
            merged_dir = str(work / "merged")
            _build_hf(base_model=self.cfg.merged_base_model,
                      adapter_path=adapter_dir, output_path=merged_dir)
            model_ref = _upload_model(
                account_id=acct, model_id=mid, local_dir=merged_dir,
                kind="HF_BASE_MODEL", base_model=None, api_key=api_key,
                display_name=self.cfg.display_name,
            )
        else:  # lora
            base = base_model or self.cfg.base_model
            if not base:
                raise RuntimeError(
                    "upload_form='lora' requires a base_model (Fireworks base the "
                    "adapter attaches to), e.g. accounts/fireworks/models/qwen3-4b"
                )
            adapter_dir = _download_weights(checkpoint_uri, str(work / "adapter"))
            removed = _prune_lora_adapter(adapter_dir, self.cfg.exclude_modules)
            if removed:
                logger.info("deploy: pruned %d LoRA tensor(s) for excluded modules %s",
                            len(removed), self.cfg.exclude_modules)
            if self.cfg.generation_defaults:
                _write_fireworks_json(adapter_dir, self.cfg.generation_defaults)
            model_ref = _upload_model(
                account_id=acct, model_id=mid, local_dir=adapter_dir,
                kind="HF_PEFT_ADDON", base_model=base, api_key=api_key,
                display_name=self.cfg.display_name,
            )

        deployment_id = None
        served_model = model_ref
        if self.cfg.create_deployment:
            # LoRA addons deploy on their base (then attach); merged on themselves.
            deploy_base = None if self.cfg.upload_form == "merged" else (base_model or self.cfg.base_model)
            deployment_id = _create_deployment(
                model_ref=model_ref, base_model=deploy_base, account_id=acct, api_key=api_key,
                params=dict(self.cfg.deployment_params),
            )
            # A dedicated deployment is addressed for inference by a
            # deployment-qualified model id: ``<model_ref>#<deployment_ref>``.
            served_model = f"{model_ref}#{deployment_id}"

        return DeployResult(
            provider="fireworks",
            model_ref=model_ref,
            endpoint=self.cfg.endpoint,
            deployment_id=deployment_id,
            deployed=deployment_id is not None,
            metadata={
                "upload_form": self.cfg.upload_form,
                "checkpoint": checkpoint_uri,
                "served_model": served_model,
            },
        )


def _write_fireworks_json(adapter_dir: str, defaults: dict[str, Any]) -> None:
    payload = {"defaults": dict(defaults), "has_lora": True}
    (Path(adapter_dir) / "fireworks.json").write_text(json.dumps(payload, indent=2))


__all__ = ["FireworksDeployer", "FireworksDeployConfig"]

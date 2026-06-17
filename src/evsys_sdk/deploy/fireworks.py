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
) -> str:
    """Create the Fireworks model, upload the local files, and validate to READY
    via the documented REST pipeline (``requests``). Returns the model ref.

    ``kind`` is ``HF_PEFT_ADDON`` (LoRA) or ``HF_BASE_MODEL`` (merged)."""
    import time

    import requests

    base = FIREWORKS_API_BASE
    auth = {"Authorization": f"Bearer {api_key}"}
    json_h = {**auth, "Content-Type": "application/json"}
    files = [f for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))]
    sizes = {f: os.path.getsize(os.path.join(local_dir, f)) for f in files}

    # 1. Create the model (idempotent — ALREADY_EXISTS is fine on retry).
    r = requests.post(
        f"{base}/accounts/{account_id}/models", headers=json_h,
        json={"modelId": model_id, "model": _model_data(kind=kind, base_model=base_model, files=files)},
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


def _create_deployment(
    *, model_ref: str, account_id: str, api_key: str, params: dict[str, Any],
) -> str:
    """Create a (dedicated) deployment of ``model_ref`` via REST. Returns the
    deployment ref."""
    import requests

    r = requests.post(
        f"{FIREWORKS_API_BASE}/accounts/{account_id}/deployments",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"deployment": {"baseModel": model_ref, **params}},
    )
    r.raise_for_status()
    body = r.json() if r.content else {}
    return body.get("name", model_ref) if isinstance(body, dict) else model_ref


def _model_data(*, kind: str, base_model: str | None, files: list[str]) -> dict[str, Any]:
    """Build the ``model`` payload for create_model. NOTE: the exact field names
    for HF_PEFT_ADDON are the one bit to confirm on a live run (the public REST
    docs fully document only HF_BASE_MODEL)."""
    if kind == "HF_PEFT_ADDON":
        return {
            "kind": "HF_PEFT_ADDON",
            "peftDetails": {"baseModel": base_model},
            "huggingFaceFiles": files,
        }
    return {
        "kind": "HF_BASE_MODEL",
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
            )
        else:  # lora
            base = base_model or self.cfg.base_model
            if not base:
                raise RuntimeError(
                    "upload_form='lora' requires a base_model (Fireworks base the "
                    "adapter attaches to), e.g. accounts/fireworks/models/qwen3-4b"
                )
            adapter_dir = _download_weights(checkpoint_uri, str(work / "adapter"))
            if self.cfg.generation_defaults:
                _write_fireworks_json(adapter_dir, self.cfg.generation_defaults)
            model_ref = _upload_model(
                account_id=acct, model_id=mid, local_dir=adapter_dir,
                kind="HF_PEFT_ADDON", base_model=base, api_key=api_key,
            )

        deployment_id = None
        if self.cfg.create_deployment:
            deployment_id = _create_deployment(
                model_ref=model_ref, account_id=acct, api_key=api_key,
                params=dict(self.cfg.deployment_params),
            )

        return DeployResult(
            provider="fireworks",
            model_ref=model_ref,
            endpoint=self.cfg.endpoint,
            deployment_id=deployment_id,
            deployed=deployment_id is not None,
            metadata={"upload_form": self.cfg.upload_form, "checkpoint": checkpoint_uri},
        )


def _write_fireworks_json(adapter_dir: str, defaults: dict[str, Any]) -> None:
    payload = {"defaults": dict(defaults), "has_lora": True}
    (Path(adapter_dir) / "fireworks.json").write_text(json.dumps(payload, indent=2))


__all__ = ["FireworksDeployer", "FireworksDeployConfig"]

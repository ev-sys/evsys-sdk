"""Fireworks deployer — upload a trained checkpoint and stand up a deployment
via the ``fireworks-ai`` Python SDK (a single ``pip install``, no separate
``firectl`` binary).

Tinker training is always LoRA, so the default path downloads just the LoRA
adapter from the checkpoint and uploads it as a Fireworks ``HF_PEFT_ADDON`` on a
base model, then creates the (dedicated) deployment. Uploaded LoRA addons can
only run on on-demand deployments — so "deploy" means a live, billed GPU
deployment.

The external/IO bits sit behind module-level seams (:func:`_download_weights`,
:func:`_upload_model`, :func:`_create_deployment`) so the orchestration is unit
testable without ``fireworks-ai`` or ``tinker_cookbook``, and without ever
standing up a real (billed) deployment.

Install: ``pip install evsys-sdk[deploy]`` (brings ``fireworks-ai`` +
``tinker-cookbook``). Auth: set ``FIREWORKS_API_KEY``.
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
    via the fireworks-ai SDK's REST pipeline. Returns the model ref.

    ``kind`` is ``HF_PEFT_ADDON`` (LoRA) or ``HF_BASE_MODEL`` (merged)."""
    import asyncio
    import time

    import httpx
    from fireworks.flumina import crud  # type: ignore

    os.environ["FIREWORKS_API_KEY"] = api_key
    files = [f for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))]
    sizes = {f: os.path.getsize(os.path.join(local_dir, f)) for f in files}
    model_data = _model_data(kind=kind, base_model=base_model, files=files)

    async def _run() -> None:
        await crud.create_model(account_id, model_id, model_data)
        ep = await crud.get_model_upload_endpoint(account_id, model_id, sizes)
        urls = ep["filenameToSignedUrls"]
        async with httpx.AsyncClient(timeout=None) as client:
            for fn, url in urls.items():
                with open(os.path.join(local_dir, fn), "rb") as fh:
                    resp = await client.put(
                        url, content=fh.read(),
                        headers={
                            "Content-Type": "application/octet-stream",
                            "x-goog-content-length-range": f"{sizes[fn]},{sizes[fn]}",
                        },
                    )
                    resp.raise_for_status()
        # validateUpload returns FAILED_PRECONDITION while files finalize.
        for _ in range(60):
            try:
                await crud.validate_model_upload(account_id, model_id)
                return
            except Exception as e:  # noqa: BLE001
                if "FAILED_PRECONDITION" in str(e):
                    time.sleep(10)
                    continue
                raise
        raise RuntimeError(f"model {model_id} did not reach READY in time")

    asyncio.run(_run())
    return f"accounts/{account_id}/models/{model_id}"


def _create_deployment(
    *, model_ref: str, account_id: str, api_key: str, params: dict[str, Any],
) -> str:
    """Create a (dedicated) deployment of ``model_ref`` and wait for it. Returns
    the deployment ref. Uses the fireworks-ai SDK."""
    import asyncio

    from fireworks.flumina import crud  # type: ignore

    os.environ["FIREWORKS_API_KEY"] = api_key
    deployment_data = {"baseModel": model_ref, **params}

    async def _run() -> Any:
        return await crud.create_deployment(account_id, deployment_data)

    result = asyncio.run(_run())
    return (result or {}).get("name", model_ref) if isinstance(result, dict) else model_ref


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

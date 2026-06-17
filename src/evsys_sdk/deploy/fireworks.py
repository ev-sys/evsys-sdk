"""Fireworks deployer — upload a trained checkpoint and stand up a deployment
via the ``firectl`` CLI.

Tinker training is always LoRA, so the default path downloads just the LoRA
adapter from the checkpoint and uploads it as a Fireworks LoRA addon on a base
model (``firectl model create … --base-model …``), then creates the deployment
(``firectl deployment create … --wait``). Uploaded LoRA addons can only run on
on-demand (dedicated) deployments — so "deploy" here means a live, billed GPU
deployment.

The heavy/external bits are isolated behind two module-level seams,
:func:`_download_weights` and :func:`_firectl`, so the orchestration is unit
testable without ``tinker_cookbook`` or the ``firectl`` binary.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
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
    """Env var holding the Fireworks API key (exported to firectl)."""

    base_model: str | None = None
    """For ``upload_form='lora'``: the Fireworks base model the adapter attaches
    to, e.g. ``accounts/fireworks/models/qwen3-4b``. May be overridden per call."""
    model_id: str | None = None
    """Target Fireworks model id. Defaults to a slug derived from the checkpoint."""

    upload_form: str = "lora"
    """``lora`` (default — upload the LoRA adapter) or ``merged`` (merge into the
    base and upload a full HF model; needs a GPU + ``merged_base_model``)."""
    merged_base_model: str | None = None
    """HF base model name for ``upload_form='merged'`` (passed to build_hf_model)."""

    create_deployment: bool = True
    """Also create a live (dedicated) deployment via ``firectl deployment create
    --wait``. False = upload to READY only."""
    deployment_args: list[str] = Field(default_factory=list)
    """Extra args appended to ``firectl deployment create`` (e.g. GPU/accelerator)."""

    generation_defaults: dict[str, Any] | None = None
    """If set (LoRA only), written as ``fireworks.json`` ``defaults`` alongside
    the adapter (stop/max_tokens/temperature/...)."""

    firectl_path: str = "firectl"
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
    Lazy-imports the cookbook export helper (needs torch + GPU/CPU compute)."""
    from tinker_cookbook.weights import build_hf_model  # type: ignore

    build_hf_model(base_model=base_model, adapter_path=adapter_path, output_path=output_path)


def _firectl(args: list[str], *, firectl_path: str, env: dict[str, str]) -> str:
    """Run a firectl command; raise on non-zero. Returns stdout."""
    cmd = [firectl_path, *args]
    logger.info("firectl: %s", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"firectl {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


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

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        key = env.get(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(
                f"{self.cfg.api_key_env} not set — needed to authenticate firectl."
            )
        env["FIREWORKS_API_KEY"] = key  # firectl reads this name
        return env

    def deploy(
        self, checkpoint_uri: str, *,
        base_model: str | None = None, model_id: str | None = None,
    ):
        from . import DeployResult

        acct = self.cfg.account_id
        mid = model_id or self.cfg.model_id or _default_model_id(checkpoint_uri)
        env = self._env()
        work = Path(self.cfg.work_dir) if self.cfg.work_dir else Path(
            tempfile.mkdtemp(prefix="evsys_deploy_")
        )

        # 1. Materialize weights locally, then 2. upload via firectl.
        if self.cfg.upload_form == "merged":
            if not self.cfg.merged_base_model:
                raise RuntimeError("upload_form='merged' requires merged_base_model")
            adapter_dir = _download_weights(checkpoint_uri, str(work / "adapter"))
            merged_dir = str(work / "merged")
            _build_hf(
                base_model=self.cfg.merged_base_model,
                adapter_path=adapter_dir, output_path=merged_dir,
            )
            _firectl(["model", "create", mid, merged_dir],
                     firectl_path=self.cfg.firectl_path, env=env)
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
            _firectl(["model", "create", mid, adapter_dir, "--base-model", base],
                     firectl_path=self.cfg.firectl_path, env=env)

        model_ref = f"accounts/{acct}/models/{mid}"

        # 3. Stand up the (dedicated) deployment.
        deployed = False
        if self.cfg.create_deployment:
            _firectl(["deployment", "create", model_ref, "--wait", *self.cfg.deployment_args],
                     firectl_path=self.cfg.firectl_path, env=env)
            deployed = True

        return DeployResult(
            provider="fireworks",
            model_ref=model_ref,
            endpoint=self.cfg.endpoint,
            deployment_id=model_ref if deployed else None,
            deployed=deployed,
            metadata={
                "upload_form": self.cfg.upload_form,
                "checkpoint": checkpoint_uri,
            },
        )


def _write_fireworks_json(adapter_dir: str, defaults: dict[str, Any]) -> None:
    payload = {"defaults": dict(defaults), "has_lora": True}
    (Path(adapter_dir) / "fireworks.json").write_text(json.dumps(payload, indent=2))


__all__ = ["FireworksDeployer", "FireworksDeployConfig"]

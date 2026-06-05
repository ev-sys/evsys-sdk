"""TinkerInference — generate via tinker SamplingClient."""

from __future__ import annotations

import os
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..checkpoint import Checkpoint, find_manifest, read_manifest
from ..registry import register_default_inference_factory, register_inference

import tinker  # noqa: E402


class TinkerInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    checkpoint_path: str | None = None
    api_key_env: str = "TINKER_API_KEY"


@register_inference("tinker")
class TinkerInference:
    name: ClassVar[str] = "tinker"
    Config: ClassVar[type] = TinkerInferenceConfig

    def __init__(
        self,
        *,
        model_name: str,
        checkpoint_path: str | None = None,
        api_key_env: str = "TINKER_API_KEY",
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} not set in env")
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        sc = tinker.ServiceClient()
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        self._tokenizer = get_tokenizer(model_name)
        if checkpoint_path:
            self._client = sc.create_sampling_client(
                base_model=model_name, model_path=checkpoint_path
            )
        else:
            self._client = sc.create_sampling_client(base_model=model_name)

    def _submit(self, prompt: str, max_tokens: int, temperature: float, stop: list[str] | None):
        from tinker import ModelInput, SamplingParams

        ids = self._tokenizer.encode(prompt)
        model_input = ModelInput.from_ints(ids)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )
        return self._client.sample(prompt=model_input, sampling_params=params, num_samples=1)

    def _decode(self, future) -> str:
        result = future.result() if hasattr(future, "result") else future
        try:
            seq = result.sequences[0]
        except Exception:
            return ""
        token_ids = getattr(seq, "tokens", None) or getattr(seq, "token_ids", None)
        if token_ids:
            return self._tokenizer.decode(list(token_ids))
        return ""

    def generate(
        self,
        *,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        return self._decode(self._submit(prompt, max_tokens, temperature, stop))

    def generate_batch(
        self,
        *,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Submit all prompts concurrently, then collect results in order."""
        futures = [self._submit(p, max_tokens, temperature, stop) for p in prompts]
        return [self._decode(f) for f in futures]

    @classmethod
    def from_run_result(cls, run_result: Any, run_cfg: Any, *,
                        label: str = "final") -> "TinkerInference":
        """Build a TinkerInference pointing at the run's final sampler checkpoint.

        Reads ``run_result.artifacts["run_dir"]``, locates ``checkpoints.jsonl``
        via :func:`find_manifest`, picks the checkpoint matching ``label``
        (default ``"final"``), and instantiates with ``run_cfg.model.name`` +
        that sampler URI. Raises a clear ``RuntimeError`` for each missing
        piece so callers can diagnose without spelunking.
        """
        artifacts = getattr(run_result, "artifacts", None) or {}
        run_dir = artifacts.get("run_dir")
        if not run_dir:
            raise RuntimeError("run_result.artifacts has no 'run_dir'")
        manifest = find_manifest(run_dir)
        if manifest is None:
            raise RuntimeError(f"no checkpoints.jsonl under {run_dir}")
        chosen = Checkpoint.pick_final(read_manifest(manifest))
        if chosen is None or not chosen.sampler_path:
            raise RuntimeError(
                f"no usable sampler checkpoint at {label!r} in {manifest}"
            )
        return cls(model_name=run_cfg.model.name, checkpoint_path=chosen.sampler_path)


# Default factory for `backend.kind: tinker`. Registered at module load so
# `Experiment._resolve_inference_factory` can look it up without importing
# this module directly.
@register_default_inference_factory("tinker")
def _default_tinker_factory(run_result: Any, run_cfg: Any) -> TinkerInference:
    return TinkerInference.from_run_result(run_result, run_cfg)

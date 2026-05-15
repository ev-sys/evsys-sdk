"""TinkerInference — generate via tinker SamplingClient."""

from __future__ import annotations

import os
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import register_inference

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

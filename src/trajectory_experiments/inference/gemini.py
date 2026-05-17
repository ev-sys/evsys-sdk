"""GeminiInference — Google ``google-genai`` SDK.

Requires the ``google-genai`` package. Authenticates from
``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) env var unless overridden.
"""

from __future__ import annotations

import os
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import register_inference


class GeminiInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model:          str = "gemini-2.5-flash"
    api_key_env:    str = "GEMINI_API_KEY"
    fallback_api_key_env: str = "GOOGLE_API_KEY"
    default_max_tokens:  int = 1024
    default_temperature: float = 0.0
    system_instruction: str | None = None


@register_inference("gemini")
class GeminiInference:
    name: ClassVar[str] = "gemini"
    Config: ClassVar[type] = GeminiInferenceConfig

    def __init__(self, **kwargs) -> None:
        try:
            from google import genai  # type: ignore[import-not-found]
            from google.genai import types as genai_types  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "GeminiInference requires the `google-genai` package. "
                "Install with: pip install google-genai"
            ) from e

        self.cfg = GeminiInferenceConfig.model_validate(kwargs)
        self._types = genai_types
        key = os.environ.get(self.cfg.api_key_env) or os.environ.get(self.cfg.fallback_api_key_env)
        if not key:
            raise RuntimeError(
                f"GeminiInference: env var {self.cfg.api_key_env} (or "
                f"{self.cfg.fallback_api_key_env}) is not set"
            )
        self._client = genai.Client(api_key=key)

    def generate(
        self,
        *,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        gen_config = self._types.GenerateContentConfig(
            max_output_tokens=max_tokens or self.cfg.default_max_tokens,
            temperature=temperature if temperature is not None else self.cfg.default_temperature,
            stop_sequences=stop or None,
            system_instruction=self.cfg.system_instruction,
        )
        resp = self._client.models.generate_content(
            model=self.cfg.model,
            contents=prompt,
            config=gen_config,
        )
        return getattr(resp, "text", None) or ""

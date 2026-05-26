"""OpenAIInference — OpenAI ``openai`` SDK (Chat Completions).

Requires the ``openai`` package. Authenticates from ``OPENAI_API_KEY``.
``base_url`` can be set to point at a compatible endpoint (e.g. a local
vLLM or Together AI proxy).
"""

from __future__ import annotations

import os
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import register_inference


class OpenAIInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model:          str = "gpt-4o-mini"
    api_key_env:    str = "OPENAI_API_KEY"
    base_url:       str | None = None
    organization:   str | None = None
    default_max_tokens:  int = 1024
    default_temperature: float = 0.0
    system_prompt:  str | None = None
    timeout_s:      float = 60.0


@register_inference("openai")
class OpenAIInference:
    name: ClassVar[str] = "openai"
    Config: ClassVar[type] = OpenAIInferenceConfig

    def __init__(self, **kwargs) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "OpenAIInference requires the `openai` package. "
                "Install with: pip install openai"
            ) from e

        self.cfg = OpenAIInferenceConfig.model_validate(kwargs)
        key = os.environ.get(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(
                f"OpenAIInference: env var {self.cfg.api_key_env} is not set"
            )
        ctor: dict = {"api_key": key, "timeout": self.cfg.timeout_s}
        if self.cfg.base_url:     ctor["base_url"]     = self.cfg.base_url
        if self.cfg.organization: ctor["organization"] = self.cfg.organization
        self._client = OpenAI(**ctor)

    def generate(
        self,
        *,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        messages: list[dict] = []
        if self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens or self.cfg.default_max_tokens,
            "temperature": temperature if temperature is not None else self.cfg.default_temperature,
        }
        if stop:
            kwargs["stop"] = stop

        resp = self._client.chat.completions.create(**kwargs)
        choices = getattr(resp, "choices", []) or []
        if not choices:
            return ""
        msg = choices[0].message
        return getattr(msg, "content", None) or ""

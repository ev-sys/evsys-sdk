"""ClaudeInference — Anthropic Messages API.

Requires the ``anthropic`` package. Authenticates from the standard
``ANTHROPIC_API_KEY`` env var unless overridden.
"""

from __future__ import annotations

import os
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..registry import register_inference


class ClaudeInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model:          str = "claude-sonnet-4-6"
    api_key_env:    str = "ANTHROPIC_API_KEY"
    base_url:       str | None = None
    default_max_tokens:  int = 1024
    default_temperature: float = 0.0
    system_prompt:  str | None = None
    """Optional system prompt prepended to every call. Per-call `prompt` is
    sent as the single user message."""
    timeout_s:      float = 60.0
    extra_headers:  dict[str, str] = Field(default_factory=dict)


@register_inference("claude")
class ClaudeInference:
    name: ClassVar[str] = "claude"
    Config: ClassVar[type] = ClaudeInferenceConfig

    def __init__(self, **kwargs) -> None:
        try:
            from anthropic import Anthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "ClaudeInference requires the `anthropic` package. "
                "Install with: pip install anthropic"
            ) from e

        self.cfg = ClaudeInferenceConfig.model_validate(kwargs)
        key = os.environ.get(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(
                f"ClaudeInference: env var {self.cfg.api_key_env} is not set"
            )
        ctor_kwargs: dict = {"api_key": key, "timeout": self.cfg.timeout_s}
        if self.cfg.base_url:
            ctor_kwargs["base_url"] = self.cfg.base_url
        self._client = Anthropic(**ctor_kwargs)

    def generate(
        self,
        *,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        kwargs: dict = {
            "model": self.cfg.model,
            "max_tokens": max_tokens or self.cfg.default_max_tokens,
            "temperature": temperature if temperature is not None else self.cfg.default_temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.cfg.system_prompt:
            kwargs["system"] = self.cfg.system_prompt
        if stop:
            kwargs["stop_sequences"] = stop
        if self.cfg.extra_headers:
            kwargs["extra_headers"] = self.cfg.extra_headers

        resp = self._client.messages.create(**kwargs)
        # Anthropic returns a list of content blocks; we concatenate the text ones.
        parts: list[str] = []
        for block in (resp.content or []):
            if getattr(block, "type", "") == "text":
                parts.append(getattr(block, "text", ""))
        return "".join(parts)

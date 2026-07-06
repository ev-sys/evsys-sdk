"""MockInference — deterministic stub completions for tests."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import register_inference


class MockInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    template: str = "<think>mock thinking</think>\n<answer>MOCK_ANSWER</answer>"
    """What every generate() returns. {prompt} placeholder is filled in."""


@register_inference("mock")
class MockInference:
    name: ClassVar[str] = "mock"
    Config: ClassVar[type] = MockInferenceConfig

    def __init__(self, *, template: str | None = None) -> None:
        self.template = (
            template
            if template is not None
            else "<think>mock thinking</think>\n<answer>MOCK_ANSWER</answer>"
        )

    def generate(
        self,
        *,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        try:
            out = self.template.format(prompt=prompt)
        except (KeyError, IndexError):
            out = self.template
        if stop:
            for s in stop:
                idx = out.find(s)
                if idx >= 0:
                    out = out[:idx]
        return out

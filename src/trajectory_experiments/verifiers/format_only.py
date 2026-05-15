"""FormatOnlyVerifier — rewards only structural compliance with <think>/<answer>."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..protocols import VerificationResult
from ..registry import register_verifier


class FormatOnlyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    has_think_reward: float = 0.5
    has_answer_reward: float = 0.5


@register_verifier("format_only")
class FormatOnlyVerifier:
    name: ClassVar[str] = "format_only"
    Config: ClassVar[type] = FormatOnlyConfig

    def __init__(self, *, has_think_reward: float = 0.5, has_answer_reward: float = 0.5) -> None:
        self.has_think_reward = has_think_reward
        self.has_answer_reward = has_answer_reward

    def verify(self, *, prompt: str, completion: str, target: dict[str, Any]) -> VerificationResult:
        has_think = "<think>" in completion and "</think>" in completion
        has_answer = "<answer>" in completion and "</answer>" in completion
        r = 0.0
        if has_think:
            r += self.has_think_reward
        if has_answer:
            r += self.has_answer_reward
        return VerificationResult(reward=r, info={"has_think": has_think, "has_answer": has_answer})

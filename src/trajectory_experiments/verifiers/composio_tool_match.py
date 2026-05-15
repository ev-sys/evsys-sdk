"""ComposioToolMatch — port of the grumpy/god grpo reward function.

+1.0  exact tool slug match
+0.3  correct toolkit prefix
+0.05 each for <think> and <answer> tags present
-0.5  no <answer> tags
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..protocols import VerificationResult
from ..registry import register_verifier


class ComposioToolMatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exact_match_reward: float = 1.0
    toolkit_reward: float = 0.3
    has_think_reward: float = 0.05
    has_answer_reward: float = 0.05
    no_answer_penalty: float = -0.5


_ANSWER_RE = re.compile(r"<answer>\s*([\w]+)\s*</answer>")


def _extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else ""


@register_verifier("composio_tool_match")
class ComposioToolMatchVerifier:
    name: ClassVar[str] = "composio_tool_match"
    Config: ClassVar[type] = ComposioToolMatchConfig

    def __init__(
        self,
        *,
        exact_match_reward: float = 1.0,
        toolkit_reward: float = 0.3,
        has_think_reward: float = 0.05,
        has_answer_reward: float = 0.05,
        no_answer_penalty: float = -0.5,
    ) -> None:
        self.exact_match_reward = exact_match_reward
        self.toolkit_reward = toolkit_reward
        self.has_think_reward = has_think_reward
        self.has_answer_reward = has_answer_reward
        self.no_answer_penalty = no_answer_penalty

    def verify(
        self,
        *,
        prompt: str,
        completion: str,
        target: dict[str, Any],
    ) -> VerificationResult:
        expected_slug = target.get("tool_slug", "")
        expected_toolkit = target.get("toolkit", "")

        r = 0.0
        has_think = "<think>" in completion and "</think>" in completion
        has_answer = "<answer>" in completion and "</answer>" in completion

        if has_think:
            r += self.has_think_reward
        if has_answer:
            r += self.has_answer_reward
        else:
            r += self.no_answer_penalty

        predicted = _extract_answer(completion)
        is_exact = bool(expected_slug) and predicted == expected_slug
        is_toolkit = bool(expected_toolkit) and predicted.startswith(expected_toolkit + "_")

        if is_exact:
            r += self.exact_match_reward
        elif is_toolkit:
            r += self.toolkit_reward

        return VerificationResult(
            reward=r,
            info={
                "predicted": predicted,
                "exact_match": is_exact,
                "toolkit_match": is_toolkit,
                "has_think": has_think,
                "has_answer": has_answer,
            },
        )

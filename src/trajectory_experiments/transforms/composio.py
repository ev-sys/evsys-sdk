"""Composio tool-detection transforms — no-tools-in-prompt variants.

These are the transforms used by the campaign in this repo. They convert
{query, tool_slug, toolkit, description} rows into chat-shaped rows where
the model sees ONLY the query (no tool list in the user message).

The legacy training data in composio-bench/training/data/sft_train.jsonl
embedded "Available tools: ...\\n\\nQuery: ..." in the user message; here we
strip that out and leave just the query.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterable

from pydantic import BaseModel, ConfigDict

from ..registry import register_transform


_DEFAULT_SYSTEM = (
    "You are a tool search engine. Match user queries to the correct API tool. "
    "Think step by step inside <think></think> tags, then give your answer "
    "inside <answer></answer> tags."
)


class ComposioSFTNoToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system_prompt: str = _DEFAULT_SYSTEM
    include_assistant: bool = True
    """If True, builds an assistant message with templated reasoning + answer."""


@register_transform("composio_sft_no_tools")
class ComposioSFTNoToolsTransform:
    """Build SFT chat rows from raw composio rows — no tool list in prompt.

    Input row schema:  {query, tool_slug, toolkit, description}
    Output row schema: {messages: [{role, content}], tool_slug, toolkit, ...}
    """

    name: ClassVar[str] = "composio_sft_no_tools"
    Config: ClassVar[type] = ComposioSFTNoToolsConfig

    def __init__(
        self,
        *,
        system_prompt: str = _DEFAULT_SYSTEM,
        include_assistant: bool = True,
    ) -> None:
        self.system_prompt = system_prompt
        self.include_assistant = include_assistant

    def _build_assistant(self, row: dict[str, Any]) -> str:
        slug = row.get("tool_slug", "")
        desc = (row.get("description") or "").strip()
        # Truncate description to keep chains short.
        desc_short = desc[:120].rstrip(".") + ("..." if len(desc) > 120 else "")
        thought = (
            f"The user wants: {row.get('query', '')}. "
            f"{slug} is the correct tool because: {desc_short}."
        )
        return f"<think>{thought}</think>\n<answer>{slug}</answer>"

    def __call__(self, rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for row in rows:
            user_msg = f"Query: {row['query']}"
            messages: list[dict[str, str]] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ]
            if self.include_assistant:
                messages.append({"role": "assistant", "content": self._build_assistant(row)})
            yield {**row, "messages": messages}


class ComposioRLNoToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system_prompt: str = _DEFAULT_SYSTEM


@register_transform("composio_rl_no_tools")
class ComposioRLNoToolsTransform:
    """Build RL prompt rows from raw composio rows — no tool list in prompt.

    Input row schema:  {query, tool_slug, toolkit, [description]}
    Output row schema: {prompt: <chat-formatted str>, messages, tool_slug, toolkit}
    """

    name: ClassVar[str] = "composio_rl_no_tools"
    Config: ClassVar[type] = ComposioRLNoToolsConfig

    def __init__(self, *, system_prompt: str = _DEFAULT_SYSTEM) -> None:
        self.system_prompt = system_prompt

    def __call__(self, rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for row in rows:
            user_msg = f"Query: {row['query']}"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ]
            # ChatML-rendered prompt for non-tinker backends.
            prompt = (
                f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            yield {**row, "prompt": prompt, "messages": messages}

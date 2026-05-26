"""Generic JSONL row → {messages: [...]} chat-shaped row."""

from __future__ import annotations

from typing import Any, ClassVar, Iterable

from pydantic import BaseModel, ConfigDict, Field

from ..registry import register_transform


class JSONLToChatConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system_prompt: str = ""
    user_template: str = "{query}"
    """Format string with {field} placeholders pulled from each row."""
    assistant_template: str | None = None
    """If set, builds an assistant message too (for SFT). e.g. '{answer}'."""


@register_transform("jsonl_to_chat")
class JSONLToChatTransform:
    name: ClassVar[str] = "jsonl_to_chat"
    Config: ClassVar[type] = JSONLToChatConfig

    def __init__(
        self,
        *,
        system_prompt: str = "",
        user_template: str = "{query}",
        assistant_template: str | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.assistant_template = assistant_template

    def __call__(self, rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for row in rows:
            messages: list[dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append(
                {"role": "user", "content": self.user_template.format(**row)}
            )
            if self.assistant_template is not None:
                messages.append(
                    {"role": "assistant", "content": self.assistant_template.format(**row)}
                )
            yield {**row, "messages": messages}

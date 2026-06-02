"""Composio-specific transforms for tool-selection training.

Two transforms are provided:

``composio_sft_no_tools``
    Converts (query, tool_slug) rows into SFT chat format.  The model learns
    to predict the correct tool slug from a natural language query without any
    tool schemas in the context ("no_tools" = no schema lookup needed).

``composio_doc_pairs``
    Converts (query, tool_slug, description) rows into (anchor, positive)
    embedding pairs for contrastive / bi-encoder training.  The anchor is the
    user query; the positive is the formatted tool documentation string.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterable

from pydantic import BaseModel, ConfigDict

from ..registry import register_transform

_DEFAULT_SYSTEM = (
    "You are a tool-selection assistant. "
    "Given a user request, identify the single most appropriate Composio tool."
)

_DEFAULT_USER_TEMPLATE = (
    "User request: {query}\n\n"
    "Respond in this exact format:\n"
    "<think>brief reasoning</think>\n"
    "<answer>TOOL_SLUG</answer>"
)


class ComposioSFTNoToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system_prompt: str = _DEFAULT_SYSTEM
    user_template: str = _DEFAULT_USER_TEMPLATE


@register_transform("composio_sft_no_tools")
class ComposioSFTNoToolsTransform:
    """SFT chat format for tool-slug prediction without tool schemas.

    Input row fields:
        query (str): natural-language user request
        tool_slug (str): correct Composio tool slug  [target]
        description (str, optional): one-line tool description
        toolkit (str, optional): toolkit prefix, e.g. ``"OUTLOOK"``

    Output row adds:
        messages (list[dict]): system + user + assistant chat turns
    """

    name: ClassVar[str] = "composio_sft_no_tools"
    Config: ClassVar[type] = ComposioSFTNoToolsConfig

    def __init__(
        self,
        *,
        system_prompt: str = _DEFAULT_SYSTEM,
        user_template: str = _DEFAULT_USER_TEMPLATE,
    ) -> None:
        self.system_prompt = system_prompt
        self.user_template = user_template

    def __call__(self, rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for row in rows:
            tool_slug = row["tool_slug"]
            description = row.get("description", "")

            think = f"The user wants to {description.lower().rstrip('.')}." if description else f"The appropriate tool is {tool_slug}."
            assistant_content = f"<think>{think}</think>\n<answer>{tool_slug}</answer>"

            messages: list[dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({
                "role": "user",
                "content": self.user_template.format(**row),
            })
            messages.append({"role": "assistant", "content": assistant_content})

            yield {**row, "messages": messages}


class ComposioDocPairsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_parameters: bool = True
    include_toolkit: bool = True
    humanize_slug: bool = True
    """Add the slug as readable words (e.g. 'OUTLOOK_CREATE_EVENT' -> 'outlook
    create event'). Embedding models handle natural words better than UPPER_SNAKE."""
    include_description: bool = True
    """Include the row's description in the doc text. Set False when the query is
    itself derived from the description, to avoid train/eval text leakage."""


def _humanize_slug(slug: str) -> str:
    """OUTLOOK_CREATE_CALENDAR_EVENT -> 'outlook create calendar event'."""
    return slug.replace("_", " ").lower().strip()


@register_transform("composio_doc_pairs")
class ComposioDocPairsTransform:
    """Embedding training pairs from Composio tool documentation.

    Produces (anchor, positive) pairs for contrastive / bi-encoder training:
    * anchor  — the natural-language user query
    * positive — the tool's documentation text (readable slug + optional
      toolkit / description / parameters)

    The positive is built so it is *aligned with but not identical to* the
    query: it represents the tool's identity, not a copy of the user request.
    When the query is derived from the row's description, set
    ``include_description=False`` to keep the doc leakage-free.

    Input row fields:
        query (str): natural-language user request  [anchor]
        tool_slug (str): correct Composio tool slug
        description (str, optional): one-line tool description
        toolkit (str, optional): toolkit prefix, e.g. ``"OUTLOOK"``
        parameters (str, optional): comma-separated parameter names

    Output row adds:
        anchor (str): the query unchanged
        positive (str): formatted tool doc string
    """

    name: ClassVar[str] = "composio_doc_pairs"
    Config: ClassVar[type] = ComposioDocPairsConfig

    def __init__(
        self,
        *,
        include_parameters: bool = True,
        include_toolkit: bool = True,
        humanize_slug: bool = True,
        include_description: bool = True,
    ) -> None:
        self.include_parameters = include_parameters
        self.include_toolkit = include_toolkit
        self.humanize_slug = humanize_slug
        self.include_description = include_description

    def __call__(self, rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for row in rows:
            tool_slug = row["tool_slug"]
            description = row.get("description", "")
            toolkit = row.get("toolkit", "")
            parameters = row.get("parameters", "")

            # Core doc text = the tool's identity (readable slug words).
            core = _humanize_slug(tool_slug) if self.humanize_slug else tool_slug
            if self.include_description and description:
                core = f"{core}: {description}"

            parts = [core]
            if self.include_toolkit and toolkit:
                parts[0] = f"[{toolkit}] {parts[0]}"
            if self.include_parameters and parameters:
                parts.append(f"Parameters: {parameters}")

            yield {
                **row,
                "anchor": row["query"],
                "positive": " ".join(parts),
            }

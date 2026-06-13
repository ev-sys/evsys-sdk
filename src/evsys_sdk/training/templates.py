"""Chat-template helpers — thin wrappers over ``tokenizer.apply_chat_template``.

We deliberately do NOT reproduce tinker_cookbook's ``Renderer`` class hierarchy
— Hugging Face's tokenizers already implement the template via
``apply_chat_template``, and the cookbook's ``Renderer`` subclasses just wrap
that call with per-model defaults (``enable_thinking=False`` for Qwen3.5-disable
variants, etc.). We forward the kwarg directly when set, which keeps every
HF chat-template-aware tokenizer working uniformly.

The functions here return ``tinker.ModelInput`` so they're drop-in for any
sampling / forward call.
"""

from __future__ import annotations

from typing import Any, Sequence

import tinker

Message = dict[str, Any]
"""Role-tagged chat message — ``{"role": "system" | "user" | "assistant", "content": "..."}``.
Matches the HF ``apply_chat_template`` input shape."""


def apply_template(
    tokenizer: Any,
    messages: Sequence[Message],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
) -> str:
    """Call ``apply_chat_template`` with ``enable_thinking`` forwarded only
    when explicitly set.

    Why the gate: Qwen3.5 tokenizers accept ``enable_thinking``, but most
    others do NOT and raise on the unknown kwarg. Forwarding only when set
    keeps the helper tokenizer-agnostic — same approach already in
    ``ChatTemplatedInference`` (``inference/chat_templated.py``).
    """
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(list(messages), **kwargs)


def messages_to_model_input(
    tokenizer: Any,
    messages: Sequence[Message],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = None,
) -> tinker.ModelInput:
    """Apply chat template → encode → build ``tinker.ModelInput``.

    Use ``add_generation_prompt=True`` for sampling prompts (the assistant
    turn isn't included yet; the model writes it). ``False`` when building
    a completed sequence (e.g. SFT teacher-forcing).
    """
    text = apply_template(
        tokenizer, messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tinker.ModelInput.from_ints(ids)


def text_to_model_input(tokenizer: Any, text: str) -> tinker.ModelInput:
    """Bypass chat-templating; tokenize ``text`` verbatim. For callers that
    already have a fully rendered prompt string."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tinker.ModelInput.from_ints(ids)


__all__ = [
    "Message",
    "apply_template",
    "messages_to_model_input",
    "text_to_model_input",
]

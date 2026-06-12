"""Tests for ``evsys_sdk.training.templates``.

Thin chat-template helpers. The interesting bit is ``enable_thinking``
being forwarded only when explicitly set so non-Qwen tokenizers (which
raise on the unknown kwarg) keep working.
"""

from __future__ import annotations

from typing import Any

import pytest

from evsys_sdk.training.templates import (
    apply_template,
    messages_to_model_input,
    text_to_model_input,
)


class _Tokenizer:
    """Records every apply_chat_template call's kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.return_text = "<|im_start|>assistant\nA<|im_end|>"

    def apply_chat_template(
        self, messages, *, tokenize=True, add_generation_prompt=False, **extra
    ):
        self.calls.append({
            "messages": messages, "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt, **extra,
        })
        return self.return_text

    def encode(self, text, add_special_tokens=False):
        # toy mapping: char codes mod 100, deterministic
        return [ord(c) % 100 for c in text]


def test_apply_template_omits_enable_thinking_when_none():
    tok = _Tokenizer()
    apply_template(tok, [{"role": "user", "content": "x"}],
                   add_generation_prompt=True, enable_thinking=None)
    call = tok.calls[0]
    assert "enable_thinking" not in call
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True


def test_apply_template_forwards_enable_thinking_false():
    tok = _Tokenizer()
    apply_template(tok, [{"role": "user", "content": "x"}],
                   add_generation_prompt=True, enable_thinking=False)
    assert tok.calls[0]["enable_thinking"] is False


def test_apply_template_forwards_enable_thinking_true():
    tok = _Tokenizer()
    apply_template(tok, [{"role": "user", "content": "x"}],
                   add_generation_prompt=False, enable_thinking=True)
    assert tok.calls[0]["enable_thinking"] is True


def test_messages_to_model_input_returns_tinker_modelinput():
    tok = _Tokenizer()
    mi = messages_to_model_input(tok, [{"role": "user", "content": "hi"}])
    # tinker.ModelInput exposes .length and .to_ints()
    assert mi.length == len(tok.encode(tok.return_text))
    assert isinstance(mi.to_ints(), list)


def test_messages_to_model_input_add_generation_prompt_default():
    """Default mode (for sampling): add_generation_prompt=True."""
    tok = _Tokenizer()
    messages_to_model_input(tok, [{"role": "user", "content": "hi"}])
    assert tok.calls[0]["add_generation_prompt"] is True


def test_text_to_model_input_bypasses_template():
    tok = _Tokenizer()
    mi = text_to_model_input(tok, "raw prompt")
    assert tok.calls == []  # template NOT applied
    assert mi.length == len("raw prompt")

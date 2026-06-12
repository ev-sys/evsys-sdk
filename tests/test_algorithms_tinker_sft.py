"""Tests for ``evsys_sdk.algorithms.tinker_sft`` chat-template
plumbing — specifically that ``enable_thinking`` is forwarded to the
tokenizer when set, and omitted when None (so non-Qwen tokenizers that
don't accept the kwarg keep working).

The heavy training path (sft_train.main) is exercised by the existing
integration coverage; here we only validate the tokenizer-facing slice.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("tinker")  # optional dep; not installed in base CI
pytest.importorskip("tinker_cookbook")

from evsys_sdk.algorithms.tinker_sft import (
    TinkerSFTConfig,
    _apply_template,
)


class _RecordingTokenizer:
    """apply_chat_template stand-in that records every kwarg it received."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **extra: Any,
    ) -> str:
        self.calls.append({
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            **extra,
        })
        return "<TEMPLATED>"


# ---------------------------------------------------------------------------
# _apply_template
# ---------------------------------------------------------------------------


def test_apply_template_omits_enable_thinking_when_none():
    """The kwarg must NOT be passed at all when caller didn't ask for it,
    so non-Qwen tokenizers (which raise on the unknown kwarg) don't break."""
    tok = _RecordingTokenizer()
    _apply_template(tok, [{"role": "user", "content": "hi"}],
                    add_generation_prompt=True, enable_thinking=None)
    call = tok.calls[0]
    assert "enable_thinking" not in call
    assert call["add_generation_prompt"] is True
    assert call["tokenize"] is False


def test_apply_template_forwards_enable_thinking_false():
    """enable_thinking=False must reach the tokenizer — that's the whole
    point of the kwarg: Qwen3.5 then renders <think></think> as a closed
    empty block and the loss-target span begins right at the answer."""
    tok = _RecordingTokenizer()
    _apply_template(tok, [{"role": "user", "content": "hi"}],
                    add_generation_prompt=True, enable_thinking=False)
    assert tok.calls[0]["enable_thinking"] is False


def test_apply_template_forwards_enable_thinking_true():
    tok = _RecordingTokenizer()
    _apply_template(tok, [{"role": "user", "content": "hi"}],
                    add_generation_prompt=False, enable_thinking=True)
    assert tok.calls[0]["enable_thinking"] is True


# ---------------------------------------------------------------------------
# TinkerSFTConfig
# ---------------------------------------------------------------------------


def test_config_defaults_enable_thinking_to_none():
    """None is the safe default — preserves tokenizer's own default
    (Qwen3.5: True; non-Qwen: no-op since we omit the kwarg)."""
    cfg = TinkerSFTConfig()
    assert cfg.enable_thinking is None


def test_config_accepts_explicit_enable_thinking_false():
    cfg = TinkerSFTConfig(enable_thinking=False)
    assert cfg.enable_thinking is False


def test_config_rejects_unknown_kwargs():
    """extra='forbid' guard still in place — typo in config.yaml should fail."""
    with pytest.raises(Exception):  # pydantic.ValidationError
        TinkerSFTConfig(enabel_thinking=False)  # noqa: typo intentional

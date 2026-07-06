"""Tests for ``evsys_sdk.inference.chat_templated.ChatTemplatedInference``.

The wrapper applies a (system + user) chat template to each raw prompt
before forwarding to the base client, so eval-time inputs match the
chat-templated distribution the model was trained on. Tokenizer is
duck-typed: any object with an ``apply_chat_template`` method works.
"""

from __future__ import annotations

from typing import Any

import pytest

from evsys_sdk.inference.chat_templated import ChatTemplatedInference

# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Records every call to apply_chat_template; returns a canned string."""

    def __init__(self, return_value: str = "<TEMPLATED>") -> None:
        self.calls: list[dict] = []
        self.return_value = return_value

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
        return self.return_value


class _FakeBase:
    """Inference client stand-in with a `_tokenizer` and a recording generate()."""

    def __init__(self, *, tokenizer: _FakeTokenizer, completion: str = "DONE") -> None:
        self._tokenizer = tokenizer
        self._completion = completion
        self.generate_calls: list[dict] = []

    def generate(self, *, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.0, stop: list[str] | None = None) -> str:
        self.generate_calls.append({
            "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "stop": stop,
        })
        return self._completion


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_raises_when_base_has_no_tokenizer():
    """A base without `_tokenizer` is structurally incompatible — fail loudly."""
    class _Bare:
        def generate(self, **_): return ""
    with pytest.raises(TypeError, match="_tokenizer"):
        ChatTemplatedInference(_Bare(), system_prompt="hi")


def test_constructs_with_default_user_template():
    base = _FakeBase(tokenizer=_FakeTokenizer())
    w = ChatTemplatedInference(base, system_prompt="sys")
    assert w.system_prompt == "sys"
    assert w.user_template == "{prompt}"


# ---------------------------------------------------------------------------
# Generate path
# ---------------------------------------------------------------------------


def test_apply_chat_template_called_with_system_and_user():
    """generate() must build a system+user message list and route through
    apply_chat_template before forwarding the templated string to base."""
    tok = _FakeTokenizer(return_value="<RENDERED>")
    base = _FakeBase(tokenizer=tok)
    w = ChatTemplatedInference(base, system_prompt="You route tools.",
                               user_template="Query: {prompt}")
    w.generate(prompt="Spin up an airtable base")

    assert len(tok.calls) == 1
    call = tok.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "You route tools."},
        {"role": "user", "content": "Query: Spin up an airtable base"},
    ]
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True


def test_user_template_substitutes_prompt():
    """{prompt} placeholder gets the raw instruction text."""
    tok = _FakeTokenizer()
    w = ChatTemplatedInference(_FakeBase(tokenizer=tok), system_prompt="s",
                               user_template="Available tools in AIRTABLE:\nA, B, C\n\nQuery: {prompt}")
    w.generate(prompt="add a row")
    user_content = tok.calls[0]["messages"][1]["content"]
    assert user_content == ("Available tools in AIRTABLE:\nA, B, C\n\nQuery: add a row")


def test_default_user_template_is_raw_prompt():
    """Without a custom user_template the raw prompt becomes the user content verbatim."""
    tok = _FakeTokenizer()
    w = ChatTemplatedInference(_FakeBase(tokenizer=tok), system_prompt="s")
    w.generate(prompt="hello world")
    assert tok.calls[0]["messages"][1]["content"] == "hello world"


def test_passes_through_generation_params():
    """max_tokens, temperature, stop must reach the base unchanged."""
    base = _FakeBase(tokenizer=_FakeTokenizer())
    w = ChatTemplatedInference(base, system_prompt="s")
    w.generate(prompt="x", max_tokens=42, temperature=0.7, stop=["</end>"])

    assert len(base.generate_calls) == 1
    call = base.generate_calls[0]
    assert call["max_tokens"] == 42
    assert call["temperature"] == 0.7
    assert call["stop"] == ["</end>"]


def test_enable_thinking_default_none_omits_kwarg():
    """No enable_thinking passed → kwarg is NOT forwarded (preserves
    tokenizer default; non-Qwen tokenizers don't accept the kwarg)."""
    tok = _FakeTokenizer()
    w = ChatTemplatedInference(_FakeBase(tokenizer=tok), system_prompt="s")
    w.generate(prompt="hi")
    assert "enable_thinking" not in tok.calls[0]


def test_enable_thinking_false_is_forwarded():
    """enable_thinking=False must reach apply_chat_template — that's the
    whole point: Qwen3.5 then renders a closed empty <think></think>
    block and the model generates straight into the answer."""
    tok = _FakeTokenizer()
    w = ChatTemplatedInference(
        _FakeBase(tokenizer=tok), system_prompt="s", enable_thinking=False
    )
    w.generate(prompt="hi")
    assert tok.calls[0]["enable_thinking"] is False


def test_enable_thinking_true_is_forwarded():
    tok = _FakeTokenizer()
    w = ChatTemplatedInference(
        _FakeBase(tokenizer=tok), system_prompt="s", enable_thinking=True
    )
    w.generate(prompt="hi")
    assert tok.calls[0]["enable_thinking"] is True


def test_forwarded_prompt_is_the_templated_string():
    """The base receives the tokenizer's output, not the raw instruction."""
    tok = _FakeTokenizer(return_value="<|im_start|>system\n…<|im_end|>")
    base = _FakeBase(tokenizer=tok)
    w = ChatTemplatedInference(base, system_prompt="s")
    w.generate(prompt="raw")
    assert base.generate_calls[0]["prompt"] == "<|im_start|>system\n…<|im_end|>"

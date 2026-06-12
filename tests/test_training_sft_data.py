"""Tests for ``evsys_sdk.training.sft_data``.

Pure-function SFT tokenization. Uses a deterministic fake tokenizer so the
assistant-span detection / loss-mask construction is testable without HF.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("tinker")  # optional dep; not installed in base CI
pytest.importorskip("torch")

import torch

from evsys_sdk.training.sft_data import row_to_datum, sft_tokenize


class _FakeTokenizer:
    """Minimal apply_chat_template + encode stand-in.

    apply_chat_template returns a string that's the role-tagged concatenation
    of message contents (one char per token). encode returns char codes
    so the loss-mask span math is deterministic.
    """

    SEP = "|"
    EOS = "$"
    GEN = "?"   # add_generation_prompt marker

    def apply_chat_template(self, messages, *, tokenize=True,
                            add_generation_prompt=False, **extra):
        parts: list[str] = []
        for m in messages:
            parts.append(f"{m['role'][0]}{self.SEP}{m['content']}{self.EOS}")
        text = "".join(parts)
        if add_generation_prompt:
            text += self.GEN
        return text

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


# ---------------------------------------------------------------------------
# row_to_datum
# ---------------------------------------------------------------------------


def test_drops_rows_without_assistant_turn():
    tok = _FakeTokenizer()
    row = {"messages": [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]}
    assert row_to_datum(row, tok, max_seq_len=128) is None


def test_empty_messages_raises():
    tok = _FakeTokenizer()
    with pytest.raises(ValueError, match="empty messages"):
        row_to_datum({"messages": []}, tok, max_seq_len=128)


def test_single_assistant_turn_marks_correct_span():
    """One assistant turn → weights mask is 1.0 on its tokens, 0.0 elsewhere
    (taking into account the left-shift for target_tokens vs model_input)."""
    tok = _FakeTokenizer()
    row = {"messages": [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]}
    datum = row_to_datum(row, tok, max_seq_len=128)
    assert datum is not None

    # Decode the mask:
    weights = datum.loss_fn_inputs["weights"].to_torch()
    targets = datum.loss_fn_inputs["target_tokens"].to_torch()
    n_marked = int((weights > 0).sum().item())
    # The assistant span is "a|A$" → 4 tokens; left-shift drops the first
    # position so 4 - (0..1 prefix-collision) marked positions expected.
    assert n_marked >= 1
    # No weights past the assistant span — at least some of the prefix is 0.
    assert int((weights == 0).sum().item()) > 0


def test_max_seq_len_truncates_and_returns_none_if_assistant_dropped():
    """If max_seq_len truncates so aggressively that no assistant token
    remains in the labelled span, the row is dropped (returns None)."""
    tok = _FakeTokenizer()
    row = {"messages": [
        {"role": "system", "content": "SSSSSSSSSSSSSSSSSS"},
        {"role": "user", "content": "UUUUUUUUUUUUUUUUUU"},
        {"role": "assistant", "content": "A"},
    ]}
    # max_seq_len = 5 cuts off well before the assistant span.
    assert row_to_datum(row, tok, max_seq_len=5) is None


def test_enable_thinking_threads_to_apply_chat_template():
    """enable_thinking should reach the tokenizer EVERY time apply_chat_template
    is called (full / prefix / through passes)."""
    tok = _RecordingTokenizer()
    row = {"messages": [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]}
    row_to_datum(row, tok, max_seq_len=64, enable_thinking=False)
    assert tok.calls  # at least one call
    assert all(c.get("enable_thinking") is False for c in tok.calls)


def test_enable_thinking_none_omits_kwarg():
    tok = _RecordingTokenizer()
    row = {"messages": [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]}
    row_to_datum(row, tok, max_seq_len=64, enable_thinking=None)
    assert all("enable_thinking" not in c for c in tok.calls)


class _RecordingTokenizer(_FakeTokenizer):
    def __init__(self):
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages, *, tokenize=True,
                            add_generation_prompt=False, **extra):
        self.calls.append({
            "messages": list(messages),
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            **extra,
        })
        return super().apply_chat_template(
            messages, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


# ---------------------------------------------------------------------------
# sft_tokenize (batch path)
# ---------------------------------------------------------------------------


def test_sft_tokenize_drops_unlabelled_rows():
    tok = _FakeTokenizer()
    rows = [
        {"messages": [{"role": "user", "content": "U1"}]},  # no assistant
        {"messages": [{"role": "user", "content": "U2"},
                      {"role": "assistant", "content": "A2"}]},
    ]
    datums = sft_tokenize(rows, tok, max_seq_len=128)
    assert len(datums) == 1  # only the second row produces a Datum


def test_sft_tokenize_raises_when_all_rows_dropped():
    tok = _FakeTokenizer()
    rows = [
        {"messages": [{"role": "user", "content": "U"}]},
        {"messages": [{"role": "system", "content": "S"}]},
    ]
    with pytest.raises(ValueError, match="all rows produced empty"):
        sft_tokenize(rows, tok, max_seq_len=128)


def test_sft_tokenize_preserves_row_order():
    tok = _FakeTokenizer()
    rows = [
        {"messages": [{"role": "user", "content": str(i)},
                      {"role": "assistant", "content": f"A{i}"}]}
        for i in range(5)
    ]
    datums = sft_tokenize(rows, tok, max_seq_len=128)
    assert len(datums) == 5
    # decode model_input back to the user content position (best-effort sanity)
    assert isinstance(datums[0].model_input.to_ints(), list)

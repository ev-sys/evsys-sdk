"""Tests for `evsys_sdk.training.sdft_data`.

Pure-function unit tests — no backend, no async. The interesting bits are:

* `build_teacher_prompt` produces a sensible HF chat-templated ModelInput
  and threads ``enable_thinking`` through.
* `student_datum_from_rollout` builds the Datum shape `extract_completion_tokens`
  and `build_topk_targets` consume — including the mask alignment that's
  load-bearing for SDFT correctness.
* `build_topk_targets` produces the right (N, K) tensors from a manually-
  constructed teacher response, plus the documented metrics.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("tinker")  # optional dep; not installed in base CI
pytest.importorskip("torch")

import tinker
import torch

from evsys_sdk.training.sdft_data import (
    CompletionSlice,
    DEFAULT_DEMO_TEMPLATE,
    build_teacher_forced_sequence,
    build_teacher_prompt,
    build_topk_targets,
    extract_completion_tokens,
    student_datum_from_rollout,
)


class _StubTokenizer:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages, *, tokenize=True,
                            add_generation_prompt=False, **extra):
        self.calls.append({
            "messages": list(messages),
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            **extra,
        })
        parts = [f"<{m['role'][0]}>{m['content']}" for m in messages]
        text = "$".join(parts)
        if add_generation_prompt:
            text += "?"
        return text

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


# ---------------------------------------------------------------------------
# build_teacher_prompt
# ---------------------------------------------------------------------------


def test_teacher_prompt_includes_system_when_set():
    tok = _StubTokenizer()
    mi = build_teacher_prompt(
        question="Q", golden_answer="G", tokenizer=tok,
        system_prompt="SYS", demo_template="{question}|{golden_answer}",
    )
    # one apply_chat_template call: messages should have system + user
    roles = [m["role"] for m in tok.calls[0]["messages"]]
    assert roles == ["system", "user"]
    # the user content is the demo_template formatted with Q + G
    assert tok.calls[0]["messages"][1]["content"] == "Q|G"
    assert isinstance(mi, tinker.ModelInput)


def test_teacher_prompt_omits_system_when_unset():
    tok = _StubTokenizer()
    build_teacher_prompt(
        question="Q", golden_answer="G", tokenizer=tok,
        demo_template="{question}|{golden_answer}",
    )
    roles = [m["role"] for m in tok.calls[0]["messages"]]
    assert roles == ["user"]


def test_teacher_prompt_forwards_enable_thinking_false():
    tok = _StubTokenizer()
    build_teacher_prompt(
        question="Q", golden_answer="G", tokenizer=tok,
        enable_thinking=False, demo_template="{question}|{golden_answer}",
    )
    assert tok.calls[0]["enable_thinking"] is False


def test_teacher_prompt_uses_default_demo_template_when_not_overridden():
    tok = _StubTokenizer()
    build_teacher_prompt(question="Q", golden_answer="G", tokenizer=tok)
    # default template injects both fields
    content = tok.calls[0]["messages"][0]["content"]
    assert "Q" in content
    assert "G" in content


# ---------------------------------------------------------------------------
# student_datum_from_rollout
# ---------------------------------------------------------------------------


def test_student_datum_marks_completion_span_in_mask():
    prompt = tinker.ModelInput.from_ints([100, 101, 102])
    completion = [200, 201, 202]
    datum = student_datum_from_rollout(prompt=prompt, completion_tokens=completion)
    mask = datum.loss_fn_inputs["mask"].to_torch().tolist()
    # full sequence: [100,101,102,200,201,202] → model_input drops last → 5 positions
    assert len(mask) == 5
    # positions 0,1 are prompt → mask 0; positions 2,3,4 are completion → mask 1
    assert mask == [0.0, 0.0, 1.0, 1.0, 1.0]
    # target_tokens is left-shifted view of the sequence
    targets = datum.loss_fn_inputs["target_tokens"].to_torch().tolist()
    assert targets == [101, 102, 200, 201, 202]


def test_student_datum_empty_completion_returns_zero_mask():
    """When the rollout returns no tokens (rare but possible), the Datum
    still constructs with an empty (or zero) mask — the loop logs through it
    and no learning signal lands."""
    datum = student_datum_from_rollout(
        prompt=tinker.ModelInput.from_ints([1, 2, 3]),
        completion_tokens=[],
    )
    mask = datum.loss_fn_inputs["mask"].to_torch().tolist()
    assert all(v == 0.0 for v in mask)


# ---------------------------------------------------------------------------
# extract_completion_tokens
# ---------------------------------------------------------------------------


def test_extract_completion_tokens_round_trips():
    prompt = tinker.ModelInput.from_ints([10, 11, 12])
    datum = student_datum_from_rollout(prompt=prompt, completion_tokens=[20, 21, 22])
    sl = extract_completion_tokens(datum, teacher_prompt_len=5, max_context_length=128)
    assert sl.tokens == [20, 21, 22]
    assert sl.teacher_prompt_len == 5
    assert sl.truncated is False


def test_extract_completion_tokens_truncates_to_available_budget():
    prompt = tinker.ModelInput.from_ints([10, 11])
    datum = student_datum_from_rollout(prompt=prompt, completion_tokens=[20, 21, 22, 23])
    sl = extract_completion_tokens(datum, teacher_prompt_len=10, max_context_length=12)
    # available = max_context - teacher_prompt = 2 → keep first 2 completion tokens
    assert sl.tokens == [20, 21]
    assert sl.truncated is True


def test_extract_completion_tokens_empty_when_no_budget():
    prompt = tinker.ModelInput.from_ints([10, 11])
    datum = student_datum_from_rollout(prompt=prompt, completion_tokens=[20, 21])
    sl = extract_completion_tokens(datum, teacher_prompt_len=128, max_context_length=128)
    assert sl.tokens == []
    assert sl.truncated is True


# ---------------------------------------------------------------------------
# build_teacher_forced_sequence
# ---------------------------------------------------------------------------


def test_build_teacher_forced_sequence_appends_tokens():
    base = tinker.ModelInput.from_ints([1, 2, 3])
    seq = build_teacher_forced_sequence(base, [10, 20, 30])
    assert seq.to_ints() == [1, 2, 3, 10, 20, 30]


# ---------------------------------------------------------------------------
# build_topk_targets
# ---------------------------------------------------------------------------


def _make_student_datum() -> tinker.Datum:
    return student_datum_from_rollout(
        prompt=tinker.ModelInput.from_ints([1, 2]),
        completion_tokens=[5, 6, 7, 8],
    )


def test_build_topk_targets_writes_NK_tensors_at_completion_positions():
    student = [_make_student_datum()]
    slice_ = CompletionSlice(tokens=[5, 6, 7, 8], teacher_prompt_len=3, truncated=False)
    # Teacher topK: one list per teacher position (teacher_prompt_len + completion_len).
    # Only completion positions (idx >= teacher_prompt_len) matter.
    # Set top-2 entries per completion position; skip_first_n=3 means only the
    # 4th completion position writes anything.
    teacher_resp = [None] * 3 + [None, None, None, [(50, -0.1), (51, -1.0)]]
    new_datums, metrics = build_topk_targets(
        student_data=student, completion_slices=[slice_],
        teacher_topk_logprobs=[teacher_resp],
        topk=2, skip_first_n=3,
    )
    targets = new_datums[0].loss_fn_inputs["target_tokens"].to_torch()
    weights = new_datums[0].loss_fn_inputs["weights"].to_torch()
    # Shape: (N, K). N = student_full_tokens - 1 = len(prompt+completion) - 1 = 5.
    # K = 2.
    assert targets.shape[-1] == 2
    # The 4th completion position (skip_first_n=3 → position index 3) is the
    # only one with non-zero weights. Find which student_pos that maps to.
    # mask positions (from student datum) are indices [1, 2, 3, 4] (4 completion tokens
    # after the 2-token prompt → 4 positions where mask=1).
    mask = new_datums[0].loss_fn_inputs.get
    # Verify SOMETHING got written
    assert (weights > 0).any()
    # Renormalized probs over top-K should sum to ~1 at the written position.
    rows = (weights > 0).any(dim=-1)
    written_positions = torch.where(rows)[0]
    assert len(written_positions) == 1
    p = written_positions[0].item()
    assert weights[p].sum().item() == pytest.approx(1.0, abs=1e-5)
    # The targets at the written position match the input token IDs.
    assert set(targets[p].tolist()) == {50, 51}


def test_build_topk_targets_emits_documented_metrics():
    student = [_make_student_datum(), _make_student_datum()]
    slices = [
        CompletionSlice(tokens=[5, 6, 7, 8], teacher_prompt_len=2, truncated=False),
        CompletionSlice(tokens=[5, 6, 7, 8], teacher_prompt_len=2, truncated=True),
    ]
    teacher = [
        [None] * 2 + [[(10, -0.0)], [(11, -0.0)], [(12, -0.0)], [(13, -0.0)]],
        [None] * 2 + [[(10, -0.0)], [(11, -0.0)], [(12, -0.0)], [(13, -0.0)]],
    ]
    _, metrics = build_topk_targets(
        student_data=student, completion_slices=slices,
        teacher_topk_logprobs=teacher,
        topk=4, skip_first_n=0,
    )
    assert metrics["sdft/num_datums"] == 2.0
    assert metrics["sdft/teacher_truncated_count"] == 1.0
    assert metrics["sdft/topk"] == 4.0
    assert "sdft/total_completion_tokens" in metrics
    assert "sdft/mean_teacher_entropy" in metrics
    # Each teacher position had K=1 entry → entropy is 0 (single-token distribution).
    assert metrics["sdft/mean_teacher_entropy"] == pytest.approx(0.0)


def test_build_topk_targets_filters_oov_tokens():
    """Teacher tokens at indices >= vocab_size should be dropped."""
    student = [_make_student_datum()]
    slice_ = CompletionSlice(tokens=[5, 6, 7, 8], teacher_prompt_len=2, truncated=False)
    teacher = [
        [None] * 2 + [
            [(200, -0.5), (201, -0.5)],  # both >= vocab=100 → drop both
            [(50, -0.5)],
            [(60, -0.5)],
            [(70, -0.5)],
        ],
    ]
    new_datums, _ = build_topk_targets(
        student_data=student, completion_slices=[slice_],
        teacher_topk_logprobs=teacher,
        topk=2, vocab_size=100, skip_first_n=0,
    )
    weights = new_datums[0].loss_fn_inputs["weights"].to_torch()
    # Three completion positions had valid teacher targets (skip_first_n=0 →
    # only the first oob got filtered, leaving positions 1/2/3 written).
    written = (weights > 0).any(dim=-1).sum().item()
    assert written == 3


def test_build_topk_targets_zero_completion_returns_zero_target_datum():
    student = [_make_student_datum()]
    slice_ = CompletionSlice(tokens=[], teacher_prompt_len=2, truncated=False)
    new_datums, _ = build_topk_targets(
        student_data=student, completion_slices=[slice_],
        teacher_topk_logprobs=[None], topk=2,
    )
    weights = new_datums[0].loss_fn_inputs["weights"].to_torch()
    assert (weights == 0).all()

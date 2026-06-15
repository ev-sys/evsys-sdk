"""SDFT data shaping — port the cookbook's distillation math, minus
orchestration. Pure functions over tinker types — no I/O, no
``tinker_cookbook`` imports.

The Self-Distillation Fine-Tuning algorithm (Shenfeld et al., 2026):

1. **Student rollout** — sample from the live student weights on the user
   question (no demo). Each sample is a completion + per-position logprobs.
2. **Teacher prompt** — build a frozen-teacher prompt that contains the
   golden answer as an in-context demonstration.
3. **Teacher topK** — append the student completion to the teacher prompt,
   ask the teacher for its top-K token distribution at each completion
   position via tinker's ``topk_prompt_logprobs`` sampling API.
4. **CE distillation** — train the student to match the teacher's
   renormalized top-K distribution at each position via cross_entropy.

This module owns Step 4's data shaping (turning teacher responses into
``tinker.Datum`` objects with ``(N, K)``-shaped target_tokens + weights)
plus the teacher-prompt helper for Step 2. The
:class:`~evsys_sdk.algorithms.sdft.SDFT` algorithm orchestrates 1-4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

import tinker
import torch

from ..data_types import PromptExample
from .templates import Message, messages_to_model_input

logger = logging.getLogger(__name__)


DEFAULT_DEMO_TEMPLATE = (
    "{question}\n\n"
    "This is an example for a response to the question:\n"
    "{golden_answer}\n\n"
    "Now answer with a response of your own, including the thinking process."
)
"""The cookbook's default demonstration template. Researchers usually
override this in the algorithm config (e.g. to bracket the golden answer
in ``<answer>...</answer>`` tags)."""


# ---------------------------------------------------------------------------
# Teacher prompt (Step 2)
# ---------------------------------------------------------------------------


def build_teacher_prompt(
    *,
    question: str,
    golden_answer: str,
    tokenizer: Any,
    system_prompt: str | None = None,
    demo_template: str = DEFAULT_DEMO_TEMPLATE,
    enable_thinking: bool | None = None,
) -> tinker.ModelInput:
    """Render the teacher prompt (system + user-with-demo) → ``ModelInput``.

    The teacher gets to see the golden answer as a soft hint in the user
    turn (via ``demo_template``). Student completions then get appended to
    this prompt for teacher-forced top-K scoring.

    We use :func:`~evsys_sdk.training.templates.messages_to_model_input`
    rather than the cookbook's ``Renderer.build_generation_prompt`` —
    same end shape (HF chat-template applied with add_generation_prompt=True),
    no Renderer class hierarchy needed.
    """
    user_content = demo_template.format(
        question=question, golden_answer=golden_answer,
    )
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages_to_model_input(
        tokenizer, messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


# ---------------------------------------------------------------------------
# Student rollout → Datum (Step 1 plumbing)
# ---------------------------------------------------------------------------


def student_datum_from_rollout(
    *,
    prompt: tinker.ModelInput,
    completion_tokens: Sequence[int],
) -> tinker.Datum:
    """Wrap a student rollout (prompt + sampled completion) as a ``tinker.Datum``.

    The Datum is what :func:`build_topk_targets` consumes: it carries the
    full sequence as ``model_input`` and a per-position mask indicating
    which positions are completion tokens (the ones the teacher scores).

    Position alignment (matches the cookbook convention):
      * ``model_input`` covers ``prompt + completion[:-1]`` (left-shifted)
      * ``target_tokens`` covers ``completion`` (the loss targets)
      * ``mask`` is ``1`` on completion positions and ``0`` on prompt positions
    """
    prompt_tokens = prompt.to_ints()
    if not completion_tokens:
        # Empty completion → an empty mask. The downstream step builder
        # treats this as "no learning signal this step" and the loop just
        # logs through it.
        full_ids = list(prompt_tokens)
        targets: list[int] = []
        mask: list[float] = [0.0] * len(full_ids)
    else:
        full_ids = list(prompt_tokens) + list(completion_tokens)
        targets = list(full_ids[1:])
        # 1.0 on every position whose TARGET is a completion token (i.e.
        # positions [len(prompt) - 1 : len(prompt) - 1 + len(completion)]).
        n = len(full_ids) - 1
        mask = [0.0] * n
        start = len(prompt_tokens) - 1
        end = start + len(completion_tokens)
        for j in range(max(0, start), min(n, end)):
            mask[j] = 1.0

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full_ids[:-1] if full_ids else []),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(
                torch.tensor(targets, dtype=torch.long)
            ),
            "mask": tinker.TensorData.from_torch(
                torch.tensor(mask, dtype=torch.float32)
            ),
        },
    )


# ---------------------------------------------------------------------------
# Completion-tokens extraction (Step 3 plumbing)
# ---------------------------------------------------------------------------


@dataclass
class CompletionSlice:
    tokens: list[int]
    """The completion tokens themselves."""
    teacher_prompt_len: int
    """Length of the teacher prompt in tokens (used to pull the right slice
    from teacher logprob arrays)."""
    truncated: bool


def extract_completion_tokens(
    datum: tinker.Datum,
    teacher_prompt_len: int,
    *,
    max_context_length: int,
) -> CompletionSlice:
    """Pull the completion tokens off a student ``Datum``, truncating if
    ``teacher_prompt + completion`` would exceed ``max_context_length``.

    The cookbook does the same step inline; here it's a named function so
    the :class:`~evsys_sdk.algorithms.sdft.SDFT` algorithm and tests
    share the implementation.
    """
    mask = datum.loss_fn_inputs["mask"].to_torch()
    completion_indices = torch.where(mask > 0)[0]
    if len(completion_indices) == 0:
        return CompletionSlice(tokens=[], teacher_prompt_len=teacher_prompt_len,
                               truncated=False)

    # Reconstruct full student sequence (model_input is left-shifted; the last
    # target is dropped by tinker's convention).
    target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch().tolist()
    if not target_tokens:
        return CompletionSlice(tokens=[], teacher_prompt_len=teacher_prompt_len,
                               truncated=False)
    student_full_tokens = datum.model_input.to_ints() + [int(target_tokens[-1])]
    completion_start = int(completion_indices[0].item()) + 1
    completion_tokens = student_full_tokens[completion_start:]

    available = max_context_length - teacher_prompt_len
    if available <= 0:
        return CompletionSlice(tokens=[], teacher_prompt_len=teacher_prompt_len,
                               truncated=True)
    if len(completion_tokens) > available:
        return CompletionSlice(tokens=completion_tokens[:available],
                               teacher_prompt_len=teacher_prompt_len,
                               truncated=True)
    return CompletionSlice(tokens=completion_tokens,
                           teacher_prompt_len=teacher_prompt_len,
                           truncated=False)


def build_teacher_forced_sequence(
    teacher_prompt: tinker.ModelInput,
    completion_tokens: Sequence[int],
) -> tinker.ModelInput:
    """Append completion tokens to a teacher prompt to form the sequence
    the teacher will score."""
    seq = teacher_prompt
    for tok in completion_tokens:
        seq = seq.append_int(int(tok))
    return seq


# ---------------------------------------------------------------------------
# Top-K target construction (Step 4 — the math)
# ---------------------------------------------------------------------------


def build_topk_targets(
    *,
    student_data: list[tinker.Datum],
    completion_slices: list[CompletionSlice],
    teacher_topk_logprobs: list[list[list[tuple[int, float]] | None] | None],
    topk: int = 20,
    vocab_size: int | None = None,
    skip_first_n: int = 3,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Build cross_entropy Datums with ``(N, K)`` soft targets from a
    batch of teacher top-K responses. Pure function — no I/O.

    Parameters
    ----------
    student_data:
        Student Datums from :func:`student_datum_from_rollout`. Their
        ``mask`` tells us which positions are completion tokens; the
        loss targets the K-best teacher tokens at each of those positions.
    completion_slices:
        Output of :func:`extract_completion_tokens` (one per datum). Carries
        the teacher_prompt_len needed to index into ``teacher_topk_logprobs``.
    teacher_topk_logprobs:
        One list per datum. Each is a per-position list (length =
        teacher_prompt + completion length); each position is either
        ``None`` or a list of ``(token_id, logprob)`` tuples (the teacher's
        top-K). On real tinker this comes from
        ``sample_async(topk_prompt_logprobs=K).topk_prompt_logprobs``.
    topk:
        How many of the teacher's top tokens to keep (truncates if the
        teacher returned fewer).
    vocab_size:
        If set, drop teacher tokens >= ``vocab_size`` (handles special
        tokens vLLM may emit outside the student's vocab).
    skip_first_n:
        Skip the first N completion positions from the loss. Matches the
        reference SDFT paper (default 3).

    Returns
    -------
    ``(new_datums, metrics)`` where each new Datum has cross_entropy
    ``loss_fn_inputs`` with target_tokens shape ``(N, K)`` and weights
    shape ``(N, K)``. ``metrics`` includes ``sdft/mean_teacher_entropy``,
    ``sdft/total_completion_tokens``, ``sdft/teacher_truncated_count``,
    ``sdft/num_datums``, ``sdft/topk``.
    """
    new_datums: list[tinker.Datum] = []
    truncated_count = 0
    total_completion_tokens = 0.0
    total_teacher_entropy = 0.0

    for i, datum in enumerate(student_data):
        slice_ = completion_slices[i]
        if slice_.truncated:
            truncated_count += 1

        N = datum.model_input.length
        target_tokens_NK = torch.zeros(N, topk, dtype=torch.long)
        weights_NK = torch.zeros(N, topk, dtype=torch.float32)

        mask = datum.loss_fn_inputs["mask"].to_torch()
        completion_mask_indices = torch.where(mask > 0)[0]
        completion_len = len(slice_.tokens)
        if completion_len == 0 or len(completion_mask_indices) == 0:
            new_datums.append(_make_topk_datum(datum, target_tokens_NK, weights_NK))
            continue

        topk_all = teacher_topk_logprobs[i] if i < len(teacher_topk_logprobs) else None
        if topk_all is None:
            new_datums.append(_make_topk_datum(datum, target_tokens_NK, weights_NK))
            continue

        n_positions = min(completion_len, len(completion_mask_indices))
        for t in range(n_positions):
            if t < skip_first_n:
                continue
            teacher_pos = slice_.teacher_prompt_len + t
            if teacher_pos >= len(topk_all):
                continue
            topk_entries = topk_all[teacher_pos]
            if not topk_entries:
                continue

            filtered = [
                (int(tok_id), float(lp))
                for tok_id, lp in topk_entries[:topk]
                if vocab_size is None or int(tok_id) < vocab_size
            ]
            if not filtered:
                continue

            k_actual = len(filtered)
            token_ids = torch.tensor([t_id for t_id, _ in filtered], dtype=torch.long)
            logprobs = torch.tensor([lp for _, lp in filtered], dtype=torch.float32)
            # Renormalize to a proper distribution over the K tokens.
            logprobs = logprobs - torch.logsumexp(logprobs, dim=0)
            probs = logprobs.exp()

            student_pos = int(completion_mask_indices[t].item())
            target_tokens_NK[student_pos, :k_actual] = token_ids
            weights_NK[student_pos, :k_actual] = probs
            total_teacher_entropy += -(probs * logprobs).sum().item()

        total_completion_tokens += n_positions
        new_datums.append(_make_topk_datum(datum, target_tokens_NK, weights_NK))

    metrics: dict[str, float] = {
        "sdft/teacher_truncated_count": float(truncated_count),
        "sdft/num_datums": float(len(student_data)),
        "sdft/topk": float(topk),
    }
    if total_completion_tokens > 0:
        metrics["sdft/total_completion_tokens"] = total_completion_tokens
        metrics["sdft/mean_teacher_entropy"] = (
            total_teacher_entropy / total_completion_tokens
        )
    return new_datums, metrics


def _make_topk_datum(
    source: tinker.Datum, targets_NK: torch.Tensor, weights_NK: torch.Tensor,
) -> tinker.Datum:
    return tinker.Datum(
        model_input=source.model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(targets_NK),
            "weights": tinker.TensorData.from_torch(weights_NK),
        },
    )


# ---------------------------------------------------------------------------
# Per-step dataset (consumed by the SDFT algorithm)
# ---------------------------------------------------------------------------


@runtime_checkable
class SDFTDataset(Protocol):
    """The data interface the SDFT algorithm consumes per step.

    Per the SDFT paper, each step needs ``batch_size`` ``(question, golden_answer)``
    pairs — the student rolls out on the question, the teacher scores
    teacher-forced through the question+golden_answer demo.
    """

    def __len__(self) -> int: ...

    def get_batch(self, step_idx: int) -> tuple[list[str], list[str]]:
        """Return ``(questions, golden_answers)`` of length ``batch_size``."""
        ...


@dataclass
class SimpleSDFTDataset:
    """Stock :class:`SDFTDataset` over :class:`~evsys_sdk.data_types.PromptExample`
    rows: the question lives in ``inputs['question']`` and the gold answer in
    ``expected``. Wraps modulo dataset length so the loop can exceed one epoch
    (no ``_RepeatingSDFTProvider`` hack needed)."""

    rows: list[PromptExample]
    batch_size: int

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("SimpleSDFTDataset: rows is empty")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0 (got {self.batch_size})")
        missing = [
            i for i, r in enumerate(self.rows[:5])
            if not r.inputs.get("question") or r.expected is None
        ]
        if missing:
            raise ValueError(
                f"SimpleSDFTDataset: rows need inputs['question'] + expected "
                f"(indices {missing} of first 5)"
            )

    def __len__(self) -> int:
        return max(1, len(self.rows) // self.batch_size)

    def get_batch(self, step_idx: int) -> tuple[list[str], list[str]]:
        n = len(self.rows)
        start = (step_idx * self.batch_size) % n
        end = start + self.batch_size
        if end <= n:
            slice_ = self.rows[start:end]
        else:
            slice_ = self.rows[start:] + self.rows[: end - n]
        return (
            [r.inputs["question"] for r in slice_],
            [str(r.expected) for r in slice_],
        )


__all__ = [
    "CompletionSlice",
    "DEFAULT_DEMO_TEMPLATE",
    "SDFTDataset",
    "SimpleSDFTDataset",
    "build_teacher_forced_sequence",
    "build_teacher_prompt",
    "build_topk_targets",
    "extract_completion_tokens",
    "student_datum_from_rollout",
]

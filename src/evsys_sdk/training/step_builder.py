"""Concrete StepBuilder implementations.

Each StepBuilder owns one concern: turn a step index into a
:class:`~evsys_sdk.training.loop.TrainingBatch` and compute per-step
algorithm-specific metrics from the forward-backward result. The loop
(``evsys_sdk.training.loop.TrainingLoop``) drives everything else.

This module currently ships :class:`SFTStepBuilder`. SDFT and RL builders
land in follow-up commits in the same file so researchers can write
inheritance-based variants (``class FocalSFT(SFTStepBuilder)``) without
imports across modules.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, Sequence, runtime_checkable

import tinker

from .loop import TrainingBatch
from .sdft_data import (
    CompletionSlice,
    build_teacher_forced_sequence,
    build_teacher_prompt,
    build_topk_targets,
    extract_completion_tokens,
    student_datum_from_rollout,
)
from .templates import Message, messages_to_model_input

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SFT — supervised fine-tuning on pre-tokenized Datums
# ---------------------------------------------------------------------------


@dataclass
class SFTStepBuilder:
    """Cycle through a pre-tokenized dataset, ``batch_size`` Datums per step.

    Parameters
    ----------
    datums:
        Output of :func:`evsys_sdk.training.sft_data.sft_tokenize` — list of
        :class:`tinker.Datum` with ``target_tokens`` + ``weights`` masks on
        the assistant span.
    batch_size:
        Datums per training step.
    seed:
        Shuffle seed. ``None`` keeps the row order (useful for parity with
        the cookbook which is deterministic-by-default).

    The builder wraps modulo dataset length so ``num_steps`` can exceed one
    epoch — this replaces the ``_RepeatingSDFTProvider`` hack the old
    cookbook wrapper needed (see :mod:`evsys_sdk.algorithms.tinker_sdft`).
    """

    datums: list[tinker.Datum]
    batch_size: int
    seed: int | None = None
    # populated lazily, on first build_batch — keeps the dataclass-with-default
    # invariant intact while allowing in-place shuffling.
    _order: list[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.datums:
            raise ValueError("SFTStepBuilder: datums is empty")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0 (got {self.batch_size})")
        self._order = list(range(len(self.datums)))
        if self.seed is not None:
            rng = random.Random(self.seed)
            rng.shuffle(self._order)

    @property
    def steps_per_epoch(self) -> int:
        """How many full batches fit in one epoch."""
        return max(1, len(self.datums) // self.batch_size)

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        """Slice ``batch_size`` Datums for ``step_idx``, wrapping around the
        dataset when needed."""
        n = len(self.datums)
        start = (step_idx * self.batch_size) % n
        end = start + self.batch_size
        # Wrap when the slice straddles the end.
        if end <= n:
            indices = self._order[start:end]
        else:
            indices = self._order[start:] + self._order[: end - n]
        data = [self.datums[i] for i in indices]
        return TrainingBatch(data=data, loss_fn="cross_entropy")

    def step_metrics(
        self,
        step_idx: int,
        batch: TrainingBatch,
        fb_result: Any,
    ) -> dict[str, float]:
        """Compute ``train_mean_nll`` from the per-position logprobs of
        each Datum, weighted by the loss mask.

        Tinker's cross_entropy loss returns ``loss_fn_outputs[i]["logprobs"]``:
        a per-position vector of log-probabilities of the target token (so
        a "perfect" prediction has logprob 0; otherwise it's negative).
        The mean NLL is ``-sum(logprob * weight) / sum(weight)`` over the
        loss-mask positions, averaged across the batch.
        """
        outputs = getattr(fb_result, "loss_fn_outputs", None)
        if not outputs:
            return {}

        total_logprob = 0.0
        total_weight = 0.0
        for datum, out in zip(batch.data, outputs):
            logprobs = _coerce_floats(out.get("logprobs") if isinstance(out, dict)
                                      else getattr(out, "logprobs", None))
            if logprobs is None:
                continue
            weights = _coerce_floats(_extract_weights(datum))
            if weights is None or len(weights) == 0:
                continue
            # Truncate to min length so we don't blow up on a token-count mismatch
            # between the per-position logprobs and the per-position mask.
            k = min(len(logprobs), len(weights))
            for j in range(k):
                total_logprob += logprobs[j] * weights[j]
                total_weight += weights[j]

        if total_weight <= 0:
            return {}
        return {"train_mean_nll": -float(total_logprob) / float(total_weight)}


# ---------------------------------------------------------------------------
# Helpers shared across step builders
# ---------------------------------------------------------------------------


def _coerce_floats(value: Any) -> list[float] | None:
    """Best-effort: turn a TensorData / list / torch.Tensor into list[float].

    The MockBackend emits Python lists; the real TinkerBackend emits
    ``tinker.TensorData`` (which exposes ``.to_torch()``). We handle both
    here so step_metrics works against either.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [float(v) for v in value]
    if hasattr(value, "to_torch"):
        return [float(v) for v in value.to_torch().tolist()]
    if hasattr(value, "tolist"):
        return [float(v) for v in value.tolist()]
    return None


def _extract_weights(datum: tinker.Datum) -> Any:
    """Pull the per-position weight mask out of a Datum's loss_fn_inputs."""
    inputs = getattr(datum, "loss_fn_inputs", None)
    if not inputs:
        return None
    return inputs.get("weights")


__all__ = ["SDFTDataset", "SDFTStepBuilder", "SFTStepBuilder"]


# ---------------------------------------------------------------------------
# SDFT — self-distillation fine-tuning (student rollout → teacher topK score)
# ---------------------------------------------------------------------------


@runtime_checkable
class SDFTDataset(Protocol):
    """The data interface SDFTStepBuilder consumes per step.

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
    """Stock :class:`SDFTDataset` over a row list with ``question`` /
    ``golden_answer`` fields. Wraps modulo dataset length so the loop can
    exceed one epoch (no ``_RepeatingSDFTProvider`` hack needed)."""

    rows: list[dict[str, str]]
    batch_size: int

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("SimpleSDFTDataset: rows is empty")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0 (got {self.batch_size})")
        missing = [
            i for i, r in enumerate(self.rows[:5])
            if not r.get("question") or not r.get("golden_answer")
        ]
        if missing:
            raise ValueError(
                f"SimpleSDFTDataset: rows missing question/golden_answer "
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
        return [r["question"] for r in slice_], [r["golden_answer"] for r in slice_]


SamplerProvider = Callable[[], Awaitable[Any]]
"""Closure that returns the latest student sampling client.

The SDFTStepBuilder calls this at the start of each ``build_batch`` so the
on-policy rollout uses the current weights. The composer (`native_sdft`)
binds it to ``backend.snapshot_sampling_client`` — matches the cookbook's
``save_checkpoint_and_get_sampling_client`` pattern but as an injectable
seam so tests can use a static mock client.
"""


@dataclass
class SDFTStepBuilder:
    """On-policy student rollout → teacher topK score → CE distillation.

    Parameters
    ----------
    dataset:
        :class:`SDFTDataset` providing per-step ``(questions, golden_answers)``.
    tokenizer:
        HF tokenizer. Used to build student + teacher prompts.
    teacher_client:
        Sampling client over the frozen teacher weights. Constructed once
        by the composer (the cookbook supports periodic teacher resync;
        we leave that as a future-followup).
    student_sampler_provider:
        Async callable returning the latest student sampling client. Called
        at the start of each ``build_batch`` so rollouts use the current
        weights (cookbook parity).
    system_prompt / demo_template / enable_thinking:
        Template knobs threaded into ``build_teacher_prompt`` AND into the
        student rollout prompt (student sees just system + user "{question}",
        no demo).
    max_tokens / temperature:
        Student rollout sampling params.
    topk / max_context_length / vocab_size / skip_first_n:
        Forwarded to :func:`~evsys_sdk.training.sdft_data.build_topk_targets`.
    user_template:
        How the student sees the question in its user turn (e.g.
        ``"Query: {prompt}"`` to match the eval-time chat template).
    """

    dataset: SDFTDataset
    tokenizer: Any
    teacher_client: Any
    student_sampler_provider: SamplerProvider
    system_prompt: str | None = None
    demo_template: str = ""   # set in __post_init__ to the DEFAULT
    enable_thinking: bool | None = None
    max_tokens: int = 256
    temperature: float = 1.0
    topk: int = 20
    max_context_length: int = 2048
    vocab_size: int | None = None
    skip_first_n: int = 3
    user_template: str = "{question}"

    def __post_init__(self) -> None:
        from .sdft_data import DEFAULT_DEMO_TEMPLATE
        if not self.demo_template:
            self.demo_template = DEFAULT_DEMO_TEMPLATE

    @property
    def steps_per_epoch(self) -> int:
        return len(self.dataset)

    # --- the work ----------------------------------------------------------

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        questions, golden = self.dataset.get_batch(step_idx)

        # 1. Build student + teacher prompts.
        student_prompts = [
            self._build_student_prompt(q) for q in questions
        ]
        teacher_prompts = [
            build_teacher_prompt(
                question=q, golden_answer=g, tokenizer=self.tokenizer,
                system_prompt=self.system_prompt,
                demo_template=self.demo_template,
                enable_thinking=self.enable_thinking,
            )
            for q, g in zip(questions, golden)
        ]

        # 2. On-policy student rollouts (one per question, batched in parallel).
        sampler = await self.student_sampler_provider()
        student_responses = await asyncio.gather(*[
            sampler.sample_async(
                prompt=sp,
                params=tinker.SamplingParams(
                    max_tokens=self.max_tokens, temperature=self.temperature,
                ),
                num_samples=1,
            )
            for sp in student_prompts
        ])

        # 3. Wrap each rollout as a student Datum (carrying the completion mask).
        student_datums: list[tinker.Datum] = []
        completion_slices: list[CompletionSlice] = []
        teacher_forced_seqs: list[tinker.ModelInput] = []

        for sp, tp, resp in zip(student_prompts, teacher_prompts, student_responses):
            completion = _extract_completion_tokens_from_response(resp)
            datum = student_datum_from_rollout(prompt=sp, completion_tokens=completion)
            student_datums.append(datum)
            slice_ = extract_completion_tokens(
                datum, teacher_prompt_len=tp.length,
                max_context_length=self.max_context_length,
            )
            completion_slices.append(slice_)
            teacher_forced_seqs.append(
                build_teacher_forced_sequence(tp, slice_.tokens)
            )

        # 4. Teacher topK at each completion position (one parallel call per datum).
        teacher_responses = await asyncio.gather(*[
            self.teacher_client.sample_async(
                prompt=seq,
                params=tinker.SamplingParams(max_tokens=1),
                num_samples=1,
                include_prompt_logprobs=True,
                topk_prompt_logprobs=self.topk,
            )
            for seq in teacher_forced_seqs
        ])
        teacher_topk = [
            getattr(r, "topk_prompt_logprobs", None) for r in teacher_responses
        ]

        # 5. Build CE Datums with (N, K) soft targets.
        ce_datums, sdft_metrics = build_topk_targets(
            student_data=student_datums,
            completion_slices=completion_slices,
            teacher_topk_logprobs=teacher_topk,
            topk=self.topk,
            vocab_size=self.vocab_size,
            skip_first_n=self.skip_first_n,
        )

        return TrainingBatch(
            data=ce_datums,
            loss_fn="cross_entropy",
            metrics=sdft_metrics,
        )

    def step_metrics(
        self, step_idx: int, batch: TrainingBatch, fb_result: Any,
    ) -> dict[str, float]:
        """SDFT's per-step training metric: mean negative-logprob of the
        teacher's preferred tokens under the student's distribution.

        The loop already merges ``batch.metrics`` (containing
        ``sdft/teacher_truncated_count``, ``sdft/mean_teacher_entropy``,
        etc.); we add ``train/mean_loss`` derived from the
        forward-backward output.
        """
        outputs = getattr(fb_result, "loss_fn_outputs", None)
        if not outputs:
            return {}
        # The cookbook returns per-position student logprobs (one entry per
        # Datum, with one float per position). Re-use the SFT-shaped math:
        # logprob is 0 on non-loss positions, negative on the rest; ignore
        # the zeros.
        total_logprob = 0.0
        n_tokens = 0
        for out in outputs:
            logprobs = _coerce_floats(out.get("logprobs") if isinstance(out, dict)
                                      else getattr(out, "logprobs", None))
            if not logprobs:
                continue
            for v in logprobs:
                if v != 0.0:
                    total_logprob += v
                    n_tokens += 1
        if n_tokens == 0:
            return {}
        mean_lp = total_logprob / n_tokens
        return {
            "train/mean_logprob": float(mean_lp),
            "train/mean_loss": -float(mean_lp),
            "train/loss_n_tokens": float(n_tokens),
        }

    # --- internals ---------------------------------------------------------

    def _build_student_prompt(self, question: str) -> tinker.ModelInput:
        user_content = self.user_template.format(question=question, prompt=question)
        messages: list[Message] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages_to_model_input(
            self.tokenizer, messages,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )


def _extract_completion_tokens_from_response(response: Any) -> list[int]:
    """Pull the token-id list out of a tinker SamplingResponse-shape object.

    Real tinker exposes ``.sequences[0].tokens``; MockSamplingClient does
    the same; either way we get a list of ints back.
    """
    seqs = getattr(response, "sequences", None)
    if not seqs:
        return []
    first = seqs[0]
    tokens = getattr(first, "tokens", None) or getattr(first, "token_ids", None)
    if not tokens:
        return []
    return [int(t) for t in tokens]

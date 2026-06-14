"""SDFT — self-distillation fine-tuning on the SDK training loop.

Replaces :class:`~evsys_sdk.algorithms.tinker_sdft.TinkerSDFT`. All the
composer plumbing lives in
:class:`~evsys_sdk.algorithms.base.BaseAlgorithm`; SDFT supplies the two
per-algorithm pieces:

* :meth:`setup` — parse rows → :class:`PromptExample` dataset, build the
  frozen teacher sampling client, and stash a per-step *student sampler
  provider* (snapshots the current weights each step so rollouts stay
  on-policy — cookbook parity without monkey-patching ``sdft.train_step``).
* :meth:`build_batch` — on-policy student rollout → teacher topK score → CE
  distillation Datums. :meth:`step_metrics` adds ``train/mean_loss`` from the
  forward-backward output.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, cast

import tinker

from ..data_types import PromptExample, TargetFormat, parse_rows
from ..protocols import RunContext
from ..registry import register_algorithm
from ..training.batch_utils import (
    coerce_floats,
    extract_completion_tokens_from_response,
)
from ..training.sdft_data import (
    DEFAULT_DEMO_TEMPLATE,
    CompletionSlice,
    SimpleSDFTDataset,
    build_teacher_forced_sequence,
    build_teacher_prompt,
    build_topk_targets,
    extract_completion_tokens,
    student_datum_from_rollout,
)
from ..training.loop import TrainingBatch
from ..training.templates import Message, messages_to_model_input
from ..training.tinker_backend import TinkerBackend, TinkerSamplingClient
from .base import BaseAlgorithm, BaseAlgorithmConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class SDFTConfig(BaseAlgorithmConfig):
    """Config for :class:`SDFT`. Inherits the shared training/save/eval knobs
    from :class:`BaseAlgorithmConfig`; adds SDFT-only fields."""

    # SDFT knobs
    topk: int = 20
    teacher_sync_every: int | None = None  # reserved; static teacher for now
    max_context_length: int = 2048
    demo_template: str = DEFAULT_DEMO_TEMPLATE
    system_prompt: str | None = None
    skip_first_n_tokens: int = 3
    user_template: str = "{question}"

    # Student rollout generation knobs
    max_tokens: int = 256
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


@register_algorithm("sdft")
class SDFT(BaseAlgorithm):
    name: ClassVar[str] = "sdft"
    Config: ClassVar[type] = SDFTConfig

    def _check_inputs(self, ctx: RunContext) -> None:
        rows = ctx.extras.get("train_rows")
        if not rows:
            raise RuntimeError("SDFT.train: ctx.extras['train_rows'] missing/empty")
        # Standardize raw rows → typed PromptExample (strict): inputs['question']
        # is the prompt, expected is the gold answer. Done here (pre-backend) so
        # row-shape errors surface before we allocate a (costly) tinker session.
        examples = cast("list[PromptExample]", parse_rows(rows, TargetFormat.PROMPT_DATASET))
        self._dataset = SimpleSDFTDataset(rows=examples, batch_size=self.cfg.batch_size)
        self._n_rows = len(rows)

    async def setup(self, ctx: RunContext, backend: TinkerBackend) -> None:
        self._tokenizer = backend.get_tokenizer()

        # Teacher sampling client (frozen; same base model). snapshot_sampling_client
        # would bind to the *student* weights, so build a separate sampler over the
        # untrained base via the underlying service client.
        teacher_client = backend._service.create_sampling_client(  # type: ignore[attr-defined]
            base_model=self._model_name,
        )
        self._teacher = TinkerSamplingClient(teacher_client, name="teacher")

        # Per-step student sampler provider: snapshot the current weights each
        # step so the on-policy rollout uses fresh weights (cookbook's
        # save_checkpoint_and_get_sampling_client, as an injectable seam).
        self._backend = backend
        self._snapshot_i = 0

        self._steps_per_epoch = max(1, len(self._dataset))

    async def _latest_student_sampler(self) -> Any:
        self._snapshot_i += 1
        return await self._backend.snapshot_sampling_client(
            name=f"student_snap_{self._snapshot_i}"
        )

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        questions, golden = self._dataset.get_batch(step_idx)

        # 1. Build student + teacher prompts.
        student_prompts = [self._build_student_prompt(q) for q in questions]
        teacher_prompts = [
            build_teacher_prompt(
                question=q, golden_answer=g, tokenizer=self._tokenizer,
                system_prompt=self.cfg.system_prompt,
                demo_template=self.cfg.demo_template,
                enable_thinking=self.cfg.enable_thinking,
            )
            for q, g in zip(questions, golden)
        ]

        # 2. On-policy student rollouts (one per question, batched in parallel).
        sampler = await self._latest_student_sampler()
        student_responses = await asyncio.gather(*[
            sampler.sample_async(
                prompt=sp,
                params=tinker.SamplingParams(
                    max_tokens=self.cfg.max_tokens, temperature=self.cfg.temperature,
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
            completion = extract_completion_tokens_from_response(resp)
            datum = student_datum_from_rollout(prompt=sp, completion_tokens=completion)
            student_datums.append(datum)
            slice_ = extract_completion_tokens(
                datum, teacher_prompt_len=tp.length,
                max_context_length=self.cfg.max_context_length,
            )
            completion_slices.append(slice_)
            teacher_forced_seqs.append(
                build_teacher_forced_sequence(tp, slice_.tokens)
            )

        # 4. Teacher topK at each completion position (one parallel call per datum).
        teacher_responses = await asyncio.gather(*[
            self._teacher.sample_async(
                prompt=seq,
                params=tinker.SamplingParams(max_tokens=1),
                num_samples=1,
                include_prompt_logprobs=True,
                topk_prompt_logprobs=self.cfg.topk,
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
            topk=self.cfg.topk,
            vocab_size=None,
            skip_first_n=self.cfg.skip_first_n_tokens,
        )

        return TrainingBatch(
            data=ce_datums,
            loss_fn="cross_entropy",
            metrics=sdft_metrics,
        )

    def step_metrics(
        self, step_idx: int, batch: TrainingBatch, fb_result: Any,
    ) -> dict[str, float]:
        """``train/mean_loss`` = mean negative-logprob of the teacher's
        preferred tokens under the student's distribution.

        The loop already merges ``batch.metrics`` (teacher entropy / truncated
        count); here we add the loss derived from the forward-backward output.
        Per-position student logprobs are 0 on non-loss positions, negative on
        the rest; ignore the zeros."""
        outputs = getattr(fb_result, "loss_fn_outputs", None)
        if not outputs:
            return {}
        total_logprob = 0.0
        n_tokens = 0
        for out in outputs:
            logprobs = coerce_floats(out.get("logprobs") if isinstance(out, dict)
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

    def _hyperparams_extra(self) -> dict[str, Any]:
        return {"n_train_rows": self._n_rows}

    # --- internals ---------------------------------------------------------

    def _build_student_prompt(self, question: str) -> tinker.ModelInput:
        user_content = self.cfg.user_template.format(question=question, prompt=question)
        messages: list[Message] = []
        if self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages_to_model_input(
            self._tokenizer, messages,
            add_generation_prompt=True,
            enable_thinking=self.cfg.enable_thinking,
        )


__all__ = ["SDFT", "SDFTConfig"]

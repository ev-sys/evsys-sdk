"""SDFT — self-distillation fine-tuning on the SDK training loop.

All the
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
import logging
from pathlib import Path
from typing import Any, ClassVar, cast

import tinker

logger = logging.getLogger(__name__)

from ..data_types import PromptExample, TargetFormat, parse_rows
from ..protocols import RunContext
from ..registry import register_algorithm
from ..training.batch_utils import coerce_floats
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

        # Per-step student rollouts go through harbor (on-policy: each step
        # saves a sampler checkpoint and points the harbor agent at it).
        self._backend = backend
        self._snapshot_i = 0
        # Rollouts persist under the run's workspace on disk; training rollouts
        # are NOT uploaded to the dashboard (only eval rollouts are).
        self._workspace = Path(ctx.output_dir) / "harbor_rollouts"

        self._steps_per_epoch = max(1, len(self._dataset))

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        from ..training.harbor_engine import run_harbor_generations

        questions, golden = self._dataset.get_batch(step_idx)

        # 1. Teacher prompts (golden answer shown as an in-context demo).
        teacher_prompts = [
            build_teacher_prompt(
                question=q, golden_answer=g, tokenizer=self._tokenizer,
                system_prompt=self.cfg.system_prompt,
                demo_template=self.cfg.demo_template,
                enable_thinking=self.cfg.enable_thinking,
            )
            for q, g in zip(questions, golden)
        ]

        # 2. On-policy student rollouts via harbor's engine (generation only —
        #    no verifier). Save a sampler checkpoint so the harbor agent samples
        #    from the current weights.
        self._snapshot_i += 1
        model_path = await self._backend.save_for_sampler(f"student_snap_{self._snapshot_i}")
        student_trajs = await run_harbor_generations(
            [self._student_user_content(q) for q in questions],
            model_name=self._model_name,
            model_path=model_path,
            workspace_dir=self._workspace,
            renderer_name=self.cfg.renderer_name,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system_prompt=self.cfg.system_prompt,
        )
        s_prompt = [len(t.turns[0].prompt_tokens) if t.turns else 0 for t in student_trajs]
        s_comp = [len(t.turns[0].completion_tokens) if t.turns else 0 for t in student_trajs]
        logger.info(
            "[sdft] step %d: batch=%d | student prompt_tokens=%s completion_tokens=%s | q0=%r gold0=%r",
            step_idx, len(questions), s_prompt, s_comp,
            (questions[0][:80] if questions else ""), (str(golden[0])[:50] if golden else ""),
        )

        # 3. Wrap each rollout as a student Datum (carrying the completion mask).
        student_datums: list[tinker.Datum] = []
        completion_slices: list[CompletionSlice] = []
        teacher_forced_seqs: list[tinker.ModelInput] = []

        for tp, traj in zip(teacher_prompts, student_trajs):
            turn = traj.turns[0] if traj.turns else None
            completion = list(turn.completion_tokens) if turn else []
            sp = tinker.ModelInput.from_ints(list(turn.prompt_tokens) if turn else [])
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
        # Alignment check: the teacher scores the teacher-forced sequence
        # (teacher_prompt + student_completion), so #topk positions should ≈
        # teacher_prompt_len + student_completion_len. The completion tail is
        # what's distilled (build_topk_targets slices it against the student).
        tp_lens = [tp.length for tp in teacher_prompts]
        comp_lens = [len(s.tokens) for s in completion_slices]
        tk_lens = [len(tk) if tk else 0 for tk in teacher_topk]
        logger.info(
            "[sdft] step %d teacher-forced: teacher_prompt_lens=%s + student_completion_lens=%s "
            "→ teacher_topk_positions=%s (topK=%d); expect positions ≈ prompt+completion",
            step_idx, tp_lens, comp_lens, tk_lens, self.cfg.topk,
        )

        # 5. Build CE Datums with (N, K) soft targets.
        ce_datums, sdft_metrics = build_topk_targets(
            student_data=student_datums,
            completion_slices=completion_slices,
            teacher_topk_logprobs=teacher_topk,
            topk=self.cfg.topk,
            vocab_size=None,
            skip_first_n=self.cfg.skip_first_n_tokens,
        )
        logger.info(
            "[sdft] step %d distill metrics: %s | ce_datums=%d",
            step_idx,
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in (sdft_metrics or {}).items()},
            len(ce_datums),
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
        logger.info(
            "[sdft] step %d loss: mean_loss=%.4f (mean_logprob=%.4f) over %d loss tokens",
            step_idx, -float(mean_lp), float(mean_lp), n_tokens,
        )
        return {
            "train/mean_logprob": float(mean_lp),
            "train/mean_loss": -float(mean_lp),
            "train/loss_n_tokens": float(n_tokens),
        }

    def _hyperparams_extra(self) -> dict[str, Any]:
        return {"n_train_rows": self._n_rows}

    # --- internals ---------------------------------------------------------

    def _student_user_content(self, question: str) -> str:
        """The user turn the student sees (no demo). The harbor agent's
        TinkerLLM renders it into a chat-templated prompt internally."""
        return self.cfg.user_template.format(question=question, prompt=question)


__all__ = ["SDFT", "SDFTConfig"]

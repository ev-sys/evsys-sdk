"""SDFT — self-distillation fine-tuning on the SDK training loop.

Supports pure SDFT, hybrid SFT-anchor, and multi-teacher continual
(frozen stage teachers + replay) while keeping catalog-style
``{candidate_tools}`` teacher prompts.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, ClassVar, cast

import tinker
from pydantic import Field

from ..data_types import PromptExample, TargetFormat, parse_rows
from ..protocols import RunContext
from ..registry import register_algorithm
from ..training.batch_utils import coerce_floats
from ..training.loop import TrainingBatch
from ..training.sdft_data import (
    DEFAULT_DEMO_TEMPLATE,
    CompletionSlice,
    MixedSDFTDataset,
    SimpleSDFTDataset,
    build_sft_anchor_datum,
    build_teacher_forced_sequence,
    build_teacher_prompt,
    build_topk_targets,
    extract_completion_tokens,
    merge_teacher_topk_logprobs,
    student_datum_from_rollout,
)
from ..training.templates import Message, messages_to_model_input
from ..training.tinker_backend import TinkerBackend, TinkerSamplingClient
from ..training.trajectory import Trajectory
from .base import BaseAlgorithm, BaseAlgorithmConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class SDFTConfig(BaseAlgorithmConfig):
    """Config for :class:`SDFT`. Inherits the shared training/save/eval knobs
    from :class:`BaseAlgorithmConfig`; adds SDFT-only fields."""

    # SDFT knobs
    topk: int = 20
    teacher_sync_every: int | None = None  # reserved; static teacher for now
    student_snapshot_every: int = 8
    """Refresh the on-policy sampler checkpoint every N steps (bounded
    staleness). Tinker counts each ``save_weights_for_sampler`` as a session
    and does not release them; snapshotting every step trips the
    "Too many active sessions" cap on long continual runs."""
    max_context_length: int = 2048
    demo_template: str = DEFAULT_DEMO_TEMPLATE
    system_prompt: str | None = None
    skip_first_n_tokens: int = 3
    user_template: str = "{question}"

    # Student rollout generation knobs
    max_tokens: int = 256
    temperature: float = 1.0

    # Hybrid SDFT + SFT anchor. alpha=0 -> pure SDFT.
    sft_anchor_alpha: float = 0.0
    """Weight of supervised golden CE: ``alpha * CE(golden) + (1-alpha) * SDFT``."""
    sft_anchor_zero_shot: bool = True
    """If True, SFT anchor uses zero-shot prompt (matches eval chat template)."""

    # Multi-teacher continual
    frozen_teacher_sampler_paths: list[str] = Field(default_factory=list)
    """Tinker sampler-weight URIs from completed prior stages."""
    replay_fraction: float = 0.0
    """Fraction of each batch replayed from prior-stage train data."""


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
        examples = cast(
            "list[PromptExample]", parse_rows(rows, TargetFormat.PROMPT_DATASET),
        )
        prior_rows = ctx.extras.get("replay_prior_rows") or []
        replay_frac = float(ctx.extras.get("replay_fraction", self.cfg.replay_fraction))
        if prior_rows and replay_frac > 0.0:
            prior_parsed: list[list[PromptExample]] = []
            for stage_rows in prior_rows:
                prior_parsed.append(
                    cast(
                        "list[PromptExample]",
                        parse_rows(stage_rows, TargetFormat.PROMPT_DATASET),
                    ),
                )
            self._dataset = MixedSDFTDataset(
                current_rows=examples,
                prior_rows=prior_parsed,
                batch_size=self.cfg.batch_size,
                replay_fraction=replay_frac,
            )
            logger.info(
                "SDFT: MixedSDFTDataset current=%d prior=%s replay_fraction=%s",
                len(examples), [len(p) for p in prior_parsed], replay_frac,
            )
        else:
            self._dataset = SimpleSDFTDataset(rows=examples, batch_size=self.cfg.batch_size)
        self._n_rows = len(rows)
        self._teacher_modes: list[str] = []

    async def setup(self, ctx: RunContext, backend: TinkerBackend) -> None:
        self._tokenizer = backend.get_tokenizer()

        teacher_client = backend._service.create_sampling_client(  # type: ignore[attr-defined]
            base_model=self._model_name,
        )
        self._teacher = TinkerSamplingClient(teacher_client, name="teacher")

        paths = list(
            ctx.extras.get("frozen_teacher_sampler_paths")
            or self.cfg.frozen_teacher_sampler_paths
        )
        self._frozen_teachers: list[TinkerSamplingClient] = []
        for j, path in enumerate(paths):
            raw = backend._service.create_sampling_client(  # type: ignore[attr-defined]
                base_model=self._model_name, model_path=path,
            )
            self._frozen_teachers.append(
                TinkerSamplingClient(raw, name=f"frozen_teacher_{j}"),
            )
        if self._frozen_teachers:
            logger.info(
                "SDFT.setup: loaded %d frozen stage teacher(s)",
                len(self._frozen_teachers),
            )

        self._backend = backend
        self._snapshot_i = 0
        self._student_model_path: str | None = None
        self._workspace = Path(ctx.output_dir) / "harbor_rollouts"
        self._steps_per_epoch = max(1, len(self._dataset))

    async def _student_sampler_path(self, step_idx: int) -> str:
        every = max(1, int(self.cfg.student_snapshot_every or 1))
        need_refresh = self._student_model_path is None or step_idx % every == 0
        if need_refresh:
            self._snapshot_i += 1
            self._student_model_path = await self._backend.save_for_sampler(
                f"student_snap_{self._snapshot_i}",
            )
        assert self._student_model_path is not None
        return self._student_model_path

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        from ..training.harbor_engine import run_harbor_rollouts

        batch_out = self._dataset.get_batch(step_idx)
        if len(batch_out) == 4:
            questions, golden, candidates, teacher_modes = batch_out
            self._teacher_modes = list(teacher_modes)
        else:
            questions, golden, candidates = batch_out  # type: ignore[misc]
            self._teacher_modes = ["ensemble"] * len(questions)

        teacher_prompts = [
            build_teacher_prompt(
                question=q, golden_answer=g, tokenizer=self._tokenizer,
                system_prompt=self.cfg.system_prompt,
                demo_template=self.cfg.demo_template,
                enable_thinking=self.cfg.enable_thinking,
                candidate_tools=c,
            )
            for q, g, c in zip(questions, golden, candidates)
        ]

        model_path = await self._student_sampler_path(step_idx)
        groups = await run_harbor_rollouts(
            [self._student_user_content(q) for q in questions],
            outcome_reward=False,
            model_name=self._model_name,
            model_path=model_path,
            workspace_dir=self._workspace,
            renderer_name=self.cfg.renderer_name,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system_prompt=self.cfg.system_prompt,
            n_concurrent=2,
            max_retries=1,
        )
        student_trajs = [
            g.trajectories[0] if g.trajectories else Trajectory(turns=[]) for g in groups
        ]

        student_datums: list[tinker.Datum] = []
        completion_slices: list[CompletionSlice] = []
        teacher_forced_seqs: list[tinker.ModelInput] = []
        kept_modes: list[str] = []
        kept_questions: list[str] = []
        kept_golden: list[str] = []

        for tp, traj, mode, q, g in zip(
            teacher_prompts, student_trajs, self._teacher_modes, questions, golden,
        ):
            turn = traj.turns[0] if traj.turns else None
            completion = list(turn.completion_tokens) if turn else []
            if not completion:
                continue
            sp = tinker.ModelInput.from_ints(list(turn.prompt_tokens) if turn else [])
            datum = student_datum_from_rollout(prompt=sp, completion_tokens=completion)
            student_datums.append(datum)
            slice_ = extract_completion_tokens(
                datum, teacher_prompt_len=tp.length,
                max_context_length=self.cfg.max_context_length,
            )
            completion_slices.append(slice_)
            teacher_forced_seqs.append(build_teacher_forced_sequence(tp, slice_.tokens))
            kept_modes.append(mode)
            kept_questions.append(q)
            kept_golden.append(g)

        if not student_datums:
            raise RuntimeError(
                "SDFT.build_batch: all student rollouts empty (harbor failures?). "
                "Often caused by Tinker 'Too many active sessions' — lower "
                "student_snapshot_every pressure or wait for sessions to drain."
            )

        teacher_topk = await self._query_teacher_topk(teacher_forced_seqs, kept_modes)

        alpha = float(self.cfg.sft_anchor_alpha)
        ce_datums, sdft_metrics = build_topk_targets(
            student_data=student_datums,
            completion_slices=completion_slices,
            teacher_topk_logprobs=teacher_topk,
            topk=self.cfg.topk,
            vocab_size=None,
            skip_first_n=self.cfg.skip_first_n_tokens,
            weight_scale=(1.0 - alpha) if alpha > 0.0 else 1.0,
        )

        if alpha > 0.0:
            anchors = self._build_sft_anchors(kept_questions, kept_golden, alpha)
            ce_datums = ce_datums + anchors
            sdft_metrics["sdft/sft_anchor_alpha"] = alpha
            sdft_metrics["sdft/n_sft_anchors"] = float(len(anchors))

        if self._frozen_teachers:
            sdft_metrics["sdft/n_frozen_teachers"] = float(len(self._frozen_teachers))
            sdft_metrics["sdft/n_ensemble_rows"] = float(
                sum(1 for m in kept_modes if m == "ensemble"),
            )
            sdft_metrics["sdft/n_replay_rows"] = float(
                sum(1 for m in kept_modes if m.startswith("frozen:")),
            )

        return TrainingBatch(
            data=ce_datums,
            loss_fn="cross_entropy",
            metrics=sdft_metrics,
        )

    async def _query_teacher_topk(
        self,
        teacher_forced_seqs: list[tinker.ModelInput],
        teacher_modes: list[str],
    ) -> list[list[list[tuple[int, float]] | None] | None]:
        """Query one or many teachers per datum; ensemble modes average top-K."""

        async def _topk_for_datum(seq: tinker.ModelInput, mode: str):
            if mode == "ensemble":
                teachers = self._frozen_teachers + [self._teacher]
            elif mode.startswith("frozen:"):
                idx = int(mode.split(":", 1)[1])
                if idx < 0 or idx >= len(self._frozen_teachers):
                    raise RuntimeError(
                        f"invalid frozen teacher index {idx} in mode {mode!r}",
                    )
                teachers = [self._frozen_teachers[idx]]
            else:
                teachers = [self._teacher]
            responses = await asyncio.gather(*[
                t.sample_async(
                    prompt=seq,
                    params=tinker.SamplingParams(max_tokens=1),
                    num_samples=1,
                    include_prompt_logprobs=True,
                    topk_prompt_logprobs=self.cfg.topk,
                )
                for t in teachers
            ])
            per_teacher = [getattr(r, "topk_prompt_logprobs", None) for r in responses]
            if len(per_teacher) == 1:
                return per_teacher[0]
            return merge_teacher_topk_logprobs(per_teacher, topk=self.cfg.topk)

        return list(await asyncio.gather(*[
            _topk_for_datum(seq, mode)
            for seq, mode in zip(teacher_forced_seqs, teacher_modes)
        ]))

    def _build_sft_anchors(
        self, questions: list[str], golden: list[str], alpha: float,
    ) -> list[tinker.Datum]:
        """Supervised golden-answer datums for the hybrid loss."""
        anchors: list[tinker.Datum] = []
        for q, g in zip(questions, golden):
            messages: list[Message] = []
            if self.cfg.system_prompt:
                messages.append({"role": "system", "content": self.cfg.system_prompt})
            messages.append({
                "role": "user",
                "content": self._student_user_content(q),
            })
            prompt_mi = messages_to_model_input(
                self._tokenizer, messages,
                add_generation_prompt=True,
                enable_thinking=self.cfg.enable_thinking,
            )
            golden_ids = self._tokenizer.encode(
                f"<answer>{g}</answer>", add_special_tokens=False,
            )
            anchors.append(build_sft_anchor_datum(
                prompt=prompt_mi, completion_tokens=golden_ids,
                topk=self.cfg.topk, weight=alpha,
            ))
        return anchors

    def step_metrics(
        self, step_idx: int, batch: TrainingBatch, fb_result: Any,
    ) -> dict[str, float]:
        outputs = getattr(fb_result, "loss_fn_outputs", None)
        if not outputs:
            return {}
        total_logprob = 0.0
        n_tokens = 0
        for out in outputs:
            logprobs = coerce_floats(
                out.get("logprobs") if isinstance(out, dict)
                else getattr(out, "logprobs", None),
            )
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

    def _student_user_content(self, question: str) -> str:
        return self.cfg.user_template.format(question=question, prompt=question)


__all__ = ["SDFT", "SDFTConfig"]

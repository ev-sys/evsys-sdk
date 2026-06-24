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
from pathlib import Path
from typing import Any, ClassVar, cast

import tinker

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
from ..training.trajectory import Trajectory
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
        # are NOT uploaded to the dashboard (only eval rollouts are). With a
        # RunLog present, route harbor's jobs dir into the agent track (no copy).
        self._run_log = ctx.extras.get("run_log")
        self._workspace = (
            self._run_log.harbor_dir("sdft") if self._run_log is not None
            else Path(ctx.output_dir) / "harbor_rollouts"
        )

        self._steps_per_epoch = max(1, len(self._dataset))

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        from ..training.harbor_engine import run_harbor_rollouts

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

        # 2. On-policy student rollouts via harbor's engine, generation-only
        #    (verify=False → no verifier/reward). Save a sampler checkpoint so the
        #    harbor agent samples from the current weights. One group per prompt
        #    (prompt order); take its single sample.
        self._snapshot_i += 1
        model_path = await self._backend.save_for_sampler(f"student_snap_{self._snapshot_i}")
        groups = await run_harbor_rollouts(
            [self._student_user_content(q) for q in questions],
            outcome_reward=False,        # raw prompts → generation-only (no verifier/reward)
            model_name=self._model_name,
            model_path=model_path,
            workspace_dir=self._workspace,
            renderer_name=self.cfg.renderer_name,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system_prompt=self.cfg.system_prompt,
        )
        if self._run_log is not None and (step_idx == 0 or step_idx % 10 == 0):
            self._run_log.log_training_rollouts(step_idx, groups, tokenizer=self._tokenizer)
        student_trajs = [
            g.trajectories[0] if g.trajectories else Trajectory(turns=[]) for g in groups
        ]

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

        # 4b. SDFT-specific logging: the teacher's top-K (token, logprob) per
        # completion position, decoded to text, into our own folder. Bounded
        # cadence + a couple datums so the log stays readable.
        if self._run_log is not None and (step_idx == 0 or step_idx % 10 == 0):
            for di, topk in enumerate(teacher_topk[:2]):
                start = (completion_slices[di].teacher_prompt_len
                         if di < len(completion_slices) else 0)
                rows = format_teacher_topk(
                    topk, decode=lambda t: self._tokenizer.decode([t]), start=start,
                )
                if rows:
                    self._run_log.record(
                        "teacher_logprobs",
                        {"step": step_idx, "datum": di, "positions": rows},
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

    def _student_user_content(self, question: str) -> str:
        """The user turn the student sees (no demo). The harbor agent's
        TinkerLLM renders it into a chat-templated prompt internally."""
        return self.cfg.user_template.format(question=question, prompt=question)


# ---------------------------------------------------------------------------
# Pure helper (harbor/tinker-free) — formats teacher top-K for the run log.
# ---------------------------------------------------------------------------


def format_teacher_topk(
    topk_per_pos: Any,
    *,
    decode: Any,
    start: int = 0,
    max_positions: int = 20,
    max_k: int = 8,
) -> list[dict[str, Any]]:
    """Turn one datum's teacher top-K (``[pos] -> [(token_id, logprob), …] | None``)
    into readable rows: ``[{"pos", "topk": [[token_text, logprob], …]}]``.

    ``decode(token_id) -> str`` makes it tinker-free (and unit-testable). Positions
    before ``start`` (the teacher-prompt region) and ``None`` positions are skipped;
    at most ``max_positions`` rows of ``max_k`` candidates each are kept."""
    rows: list[dict[str, Any]] = []
    for pos, cands in enumerate(topk_per_pos or []):
        if pos < start or not cands:
            continue
        rows.append({
            "pos": pos,
            "topk": [[decode(tid), round(float(lp), 4)] for tid, lp in list(cands)[:max_k]],
        })
        if len(rows) >= max_positions:
            break
    return rows


__all__ = ["SDFT", "SDFTConfig", "format_teacher_topk"]

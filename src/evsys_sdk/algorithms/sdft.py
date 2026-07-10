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
from typing import Any, ClassVar, cast

logger = logging.getLogger(__name__)

import tinker
from pydantic import Field

from ..data_types import PromptExample, TargetFormat, parse_rows
from ..protocols import RunContext
from ..registry import register_algorithm
from ..training.batch_utils import coerce_floats
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
    loss_mode: str = "forward_kl"
    """'forward_kl' (default): distill teacher topK as soft CE targets (our
    validated method). 'reverse_kl': paper-faithful on-policy reverse-KL
    (arXiv:2601.19897 Eq 1-2) via tinker's importance_sampling loss, with
    per-token advantage = log pi_teacher(y_t|x,c) - log pi_behavior(y_t|x)."""
    teacher_sync_every: int | None = None
    """If set (reverse_kl only): re-snapshot the teacher to the current student
    weights every N steps — a periodic-resync approximation of the paper's EMA
    teacher (true per-param EMA isn't possible on tinker's server-side weights)."""
    student_snapshot_every: int = 1
    """How often (in steps) to re-snapshot the student weights and create a NEW
    sampling client for on-policy rollouts. 1 = fully on-policy (a fresh sampling
    client every step). Tinker has no API to close a sampling client and counts
    each as an active session, so a long run at 1 accumulates one leaked session
    per step and eventually trips 'Too many active sessions'. Setting N>1 refreshes
    the sampler every N steps and REUSES it in between — sessions grow as steps/N
    instead of steps, at the cost of up to N-1 steps of student staleness (a small,
    bounded off-policy lag). Recommended N=4-8 for runs >~60 steps/stage."""
    max_context_length: int = 2048
    demo_template: str = DEFAULT_DEMO_TEMPLATE
    system_prompt: str | None = None
    skip_first_n_tokens: int = 3
    user_template: str = "{question}"

    # Student rollout generation knobs
    max_tokens: int = 256
    temperature: float = 1.0

    # Hybrid SDFT + SFT anchor (forward_kl only). alpha=0 -> pure SDFT.
    sft_anchor_alpha: float = 0.0
    """Weight of a supervised golden-answer cross_entropy term added alongside
    the distillation loss: total = alpha * CE(golden) + (1 - alpha) * SDFT_KL.
    Lifts peak accuracy (SDFT's ceiling is its demo-conditioned teacher; a direct
    golden signal converges faster) while the SDFT term preserves forgetting
    resistance. Tune in [0, 1]; 0.3-0.5 is a reasonable start."""
    sft_anchor_zero_shot: bool = True
    """If True, the SFT anchor prompt is ZERO-SHOT (system + question, no few-shot
    scaffold) so it teaches the model to emit <answer>SLUG</answer> in the exact
    shape the benchmark evaluates (which is zero-shot). This also fixes the
    scaffold-dependence that makes pure-SDFT rollouts score 0 at zero-shot eval.
    If False, the anchor mirrors the few-shot student rollout prompt."""

    # Multi-teacher continual: frozen sampler URIs from prior stages + replay.
    frozen_teacher_sampler_paths: list[str] = Field(default_factory=list)
    """Tinker sampler-weight URIs (checkpoint-final) from completed prior stages.
    Loaded as additional frozen teachers; combined with the current stage teacher."""
    replay_fraction: float = 0.0
    """Fraction of each batch replayed from prior-stage train data (multi-teacher).
    Replay rows distill from the matching frozen stage teacher only; current rows
    use the full ensemble (all frozen + current). Typical: 0.25."""


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


@register_algorithm("sdft")
class SDFT(BaseAlgorithm):
    name: ClassVar[str] = "sdft"
    Config: ClassVar[type] = SDFTConfig

    # Generic FORMAT exemplars prepended to the STUDENT prompt only (never the
    # teacher). They demonstrate the shape "<answer>SLUG</answer> then stop" on
    # DIFFERENT queries — they never reveal the current query's answer, so the
    # SDFT asymmetry (teacher knows this answer, student doesn't) is preserved.
    # This keeps on-policy rollouts clean/terminated at the SOURCE (generation),
    # which is the only thing that stops the doubled-</think> drift — editing
    # the loss target can't, because SDFT distills the teacher's (permissive,
    # non-EOS) distribution rather than a hard stop label. Validated at 0.6B
    # (local_sdft): well-formed rollout rate 0% -> 95.6%.
    _FEWSHOT: ClassVar[list[tuple[str, str]]] = [
        ("Query: Enumerate all airtable workspaces", "AIRTABLE_LIST_BASES"),
        ("Query: Make a fresh table in an airtable base", "AIRTABLE_CREATE_TABLE"),
        ("Query: Pull up a specific airtable record by id", "AIRTABLE_GET_RECORD"),
    ]

    def _check_inputs(self, ctx: RunContext) -> None:
        rows = ctx.extras.get("train_rows")
        if not rows:
            raise RuntimeError("SDFT.train: ctx.extras['train_rows'] missing/empty")
        self._ctx = ctx
        examples = cast("list[PromptExample]", parse_rows(rows, TargetFormat.PROMPT_DATASET))
        prior_rows = ctx.extras.get("replay_prior_rows") or []
        replay_frac = float(ctx.extras.get("replay_fraction", self.cfg.replay_fraction))
        if prior_rows and replay_frac > 0.0:
            prior_parsed = [
                cast("list[PromptExample]", parse_rows(pr, TargetFormat.PROMPT_DATASET))
                for pr in prior_rows
            ]
            self._dataset = MixedSDFTDataset(
                current_rows=examples,
                prior_rows=prior_parsed,
                batch_size=self.cfg.batch_size,
                replay_fraction=replay_frac,
            )
            print(
                f"SDFT: MixedSDFTDataset current={len(examples)} "
                f"prior={[len(p) for p in prior_parsed]} replay_fraction={replay_frac}",
                flush=True,
            )
        else:
            self._dataset = SimpleSDFTDataset(rows=examples, batch_size=self.cfg.batch_size)
        self._n_rows = len(rows)
        self._teacher_modes: list[str] | None = None

    async def setup(self, ctx: RunContext, backend: TinkerBackend) -> None:
        self._tokenizer = backend.get_tokenizer()
        print("SDFT.setup: tokenizer loaded", flush=True)

        self._backend = backend
        self._snapshot_i = 0
        # Cached student sampling client, refreshed every cfg.student_snapshot_every
        # steps (see SDFTConfig.student_snapshot_every). Tinker exposes no way to
        # release a sampling client, so reusing one across N steps is how we keep
        # the active-session count bounded on long runs.
        self._student_client: TinkerSamplingClient | None = None

        # Teacher = a FROZEN snapshot of THIS stage's STARTING weights (the
        # warm-start checkpoint the student was just initialized from), NOT the
        # raw base model. In continual SDFT the student starts each stage from
        # the prior stage's checkpoint; the teacher must be that same
        # task-competent model — it just additionally SEES the golden demo in
        # its prompt. Using the base model as teacher distills the capable
        # student back toward base behavior => catastrophic forgetting
        # (observed val_stage0 0.80 -> 0.21). Snapshotting the student's initial
        # weights here captures exactly the stage-start weights and is correct
        # for every stage (base for stage0-from-scratch, prev-stage otherwise).
        # Matches local_sdft.py, which loads the teacher from init_from_checkpoint.
        print("SDFT.setup: snapshotting teacher (warm-start) weights...", flush=True)
        teacher_snapshot = await backend.save_for_sampler("teacher_init")
        teacher_client = backend._service.create_sampling_client(  # type: ignore[attr-defined]
            base_model=self._model_name,
            model_path=teacher_snapshot,
        )
        self._teacher = TinkerSamplingClient(teacher_client, name="teacher")
        print(f"SDFT.setup: teacher client done (snapshot={teacher_snapshot})", flush=True)

        # Frozen stage-teacher snapshots (multi-teacher continual).
        paths = list(ctx.extras.get("frozen_teacher_sampler_paths") or self.cfg.frozen_teacher_sampler_paths)
        self._frozen_teachers: list[TinkerSamplingClient] = []
        _loop = asyncio.get_running_loop()
        _svc = backend._service  # type: ignore[attr-defined]
        _mname = self._model_name
        for j, path in enumerate(paths):
            raw = await _loop.run_in_executor(
                None,
                lambda p=path: _svc.create_sampling_client(base_model=_mname, model_path=p),
            )
            self._frozen_teachers.append(TinkerSamplingClient(raw, name=f"frozen_teacher_{j}"))
        if self._frozen_teachers:
            print(
                f"SDFT.setup: loaded {len(self._frozen_teachers)} frozen stage teacher(s)",
                flush=True,
            )

        # Renderer for building chat-templated student prompts. Prefer the
        # model's configured renderer (e.g. qwen3_5_disable_thinking) over the
        # recommended default — otherwise base-model rollouts default to
        # thinking mode, ramble past max_tokens, never emit the stop token, and
        # the teacher scores garbage. Matches base.py's backend renderer pick.
        print("SDFT.setup: loading renderer...", flush=True)
        from tinker_cookbook.model_info import get_recommended_renderer_name  # lazy: optional dep
        from tinker_cookbook.renderers import get_renderer
        handles = ctx.extras.get("backend_handles", {})
        renderer_name = (
            self.cfg.renderer_name
            or handles.get("renderer_name")
            or get_recommended_renderer_name(self._model_name)
        )
        self._renderer = get_renderer(renderer_name, self._tokenizer)
        self._stop_sequences = self._renderer.get_stop_sequences()
        print(f"SDFT.setup: renderer loaded ({renderer_name}), stop={self._stop_sequences}", flush=True)

        self._steps_per_epoch = max(1, len(self._dataset))
        print("SDFT.setup: done", flush=True)

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        batch_out = self._dataset.get_batch(step_idx)
        if len(batch_out) == 3:
            questions, golden, teacher_modes = batch_out
            self._teacher_modes = teacher_modes
        else:
            questions, golden = batch_out
            self._teacher_modes = ["ensemble"] * len(questions)

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

        # 2. On-policy student rollouts — direct SamplingClient on the EXISTING
        #    ServiceClient session (no new ServiceClient created here). tinker has
        #    no API to release a sampling client and counts each as an active
        #    session, so we only mint a NEW one every cfg.student_snapshot_every
        #    steps and reuse it in between (bounded staleness) to keep the active
        #    session count ~ steps/N. The sync create_sampling_client() is run via
        #    run_in_executor so it doesn't block the asyncio event loop.
        _loop = asyncio.get_running_loop()
        _svc = self._backend._service  # type: ignore[attr-defined]
        _mname = self._model_name
        refresh_every = max(1, self.cfg.student_snapshot_every)
        need_new = self._student_client is None or (step_idx % refresh_every == 0)
        if need_new:
            self._snapshot_i += 1
            print(f"SDFT.build_batch: step {step_idx}, saving sampler snap {self._snapshot_i}...", flush=True)
            model_path = await self._backend.save_for_sampler(f"student_snap_{self._snapshot_i}")
            print(f"SDFT.build_batch: snap saved ({model_path}), creating sampling client...", flush=True)
            _mpath = model_path
            raw_student = await _loop.run_in_executor(
                None,
                lambda: _svc.create_sampling_client(base_model=_mname, model_path=_mpath),
            )
            print("SDFT.build_batch: student client created", flush=True)
            self._student_client = TinkerSamplingClient(raw_student, name=f"student_snap_{self._snapshot_i}")
        else:
            print(
                f"SDFT.build_batch: step {step_idx}, reusing student snap {self._snapshot_i} "
                f"(refresh every {refresh_every})",
                flush=True,
            )
        student_client = self._student_client
        model_path = None  # only set when a fresh snapshot was taken this step

        # Periodic teacher re-sync (approx EMA) — reverse_kl only. Re-point the
        # teacher at the CURRENT student snapshot (just saved above) every N
        # steps, so the teacher tracks the improving student (paper uses per-step
        # EMA; tinker's server-side weights only allow this periodic version).
        if (self.cfg.loss_mode == "reverse_kl" and self.cfg.teacher_sync_every
                and step_idx > 0 and step_idx % self.cfg.teacher_sync_every == 0):
            # Need a concrete student snapshot to point the teacher at. If this
            # step reused a cached student client (no fresh snapshot), take one now.
            resync_path = model_path
            if resync_path is None:
                self._snapshot_i += 1
                resync_path = await self._backend.save_for_sampler(f"student_snap_{self._snapshot_i}")
            raw_teacher = await _loop.run_in_executor(
                None, lambda: _svc.create_sampling_client(base_model=_mname, model_path=resync_path),
            )
            self._teacher = TinkerSamplingClient(raw_teacher, name=f"teacher_resync_{step_idx}")
            print(f"SDFT.build_batch: teacher re-synced to student at step {step_idx}", flush=True)

        # Render student prompts (chat template applied via the renderer, same as
        # harbor's TinkerLLM does internally).
        sampling_params = tinker.SamplingParams(
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            stop=self._stop_sequences,
        )
        model_inputs: list[tinker.ModelInput] = []
        for q in questions:
            messages: list[dict[str, str]] = []
            if self.cfg.system_prompt:
                messages.append({"role": "system", "content": self.cfg.system_prompt})
            # Few-shot FORMAT scaffolding (student only) — see SDFT._FEWSHOT.
            for demo_q, demo_a in self._FEWSHOT:
                messages.append({"role": "user", "content": demo_q})
                messages.append({"role": "assistant", "content": f"<answer>{demo_a}</answer>"})
            messages.append({"role": "user", "content": self._student_user_content(q)})
            model_inputs.append(self._renderer.build_generation_prompt(messages))

        student_datums: list[tinker.Datum] = []
        completion_slices: list[CompletionSlice] = []
        teacher_forced_seqs: list[tinker.ModelInput] = []
        valid_teacher_prompts: list[tinker.ModelInput] = []

        _max_rollout_attempts = 10
        for _attempt in range(_max_rollout_attempts):
            # Sample all prompts in parallel; capture exceptions so one failure
            # doesn't abort the rest.
            raw_responses = await asyncio.gather(
                *[
                    student_client.raw.sample_async(
                        prompt=mi,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                    for mi in model_inputs
                ],
                return_exceptions=True,
            )

            # 3. Wrap each successful response as a student Datum.
            student_datums = []
            completion_slices = []
            teacher_forced_seqs = []
            valid_teacher_prompts = []
            rk_records: list[tuple[list[int], list[int], list[float], int]] = []
            n_empty = 0
            _reverse = self.cfg.loss_mode == "reverse_kl"

            for mi, resp, tp in zip(model_inputs, raw_responses, teacher_prompts):
                if isinstance(resp, Exception):
                    logger.warning(
                        "SDFT step %d (attempt %d/%d): sample error: %s",
                        step_idx, _attempt + 1, _max_rollout_attempts, resp,
                    )
                    n_empty += 1
                    continue
                seq = resp.sequences[0] if resp.sequences else None
                completion = list(seq.tokens) if (seq and seq.tokens) else []
                if not completion:
                    n_empty += 1
                    continue

                if _reverse:
                    # Reverse-KL needs the EXACT sampled tokens + their sampling
                    # logprobs (behavior policy), so NO truncate/re-encode here.
                    # The teacher assigns post-answer junk low logprob => negative
                    # advantage => reverse-KL suppresses the doubling on its own.
                    behavior_lp = list(seq.logprobs) if getattr(seq, "logprobs", None) else []
                    avail = self.cfg.max_context_length - tp.length
                    if avail <= 0:
                        n_empty += 1
                        continue
                    if len(completion) > avail:
                        completion = completion[:avail]
                    if len(behavior_lp) < len(completion):
                        behavior_lp = behavior_lp + [0.0] * (len(completion) - len(behavior_lp))
                    else:
                        behavior_lp = behavior_lp[: len(completion)]
                    teacher_forced_seqs.append(build_teacher_forced_sequence(tp, completion))
                    rk_records.append((list(mi.to_ints()), completion, behavior_lp, tp.length))
                    valid_teacher_prompts.append(tp)
                    continue

                # forward_kl: pin EOS after the first </answer> (truncate + EOS)
                # so the on-policy target can't drift into doubled-answer junk.
                completion = self._truncate_at_answer(completion)
                sp = tinker.ModelInput.from_ints(mi.to_ints())
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
                valid_teacher_prompts.append(tp)

            if n_empty:
                logger.warning(
                    "SDFT step %d (attempt %d/%d): %d/%d rollouts empty",
                    step_idx, _attempt + 1, _max_rollout_attempts, n_empty, len(questions),
                )
            if student_datums or rk_records:
                break
            if _attempt < _max_rollout_attempts - 1:
                logger.warning(
                    "SDFT step %d: all rollouts empty, retrying in 30s (attempt %d/%d)",
                    step_idx, _attempt + 1, _max_rollout_attempts,
                )
                await asyncio.sleep(30)

        if not student_datums and not rk_records:
            raise RuntimeError(
                f"SDFT step {step_idx}: all rollouts empty after {_max_rollout_attempts} attempts"
            )

        # Reverse-KL (paper-faithful): teacher per-token logprobs → importance_sampling.
        if _reverse:
            return await self._reverse_kl_batch(step_idx, rk_records, teacher_forced_seqs)

        # 4. Teacher topK at each completion position — single or multi-teacher.
        teacher_topk = await self._query_teacher_topk(teacher_forced_seqs, self._teacher_modes)

        # 5. Build CE Datums with (N, K) soft targets. When the hybrid SFT anchor
        #    is on, down-weight the distillation term to (1 - alpha) so it sums
        #    cleanly with the alpha-weighted golden datums below.
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

        # 6. Hybrid SFT anchor: append supervised golden-answer datums (hard
        #    targets, weight=alpha) so total loss = alpha*CE(golden) + (1-alpha)*KL.
        if alpha > 0.0:
            anchor_datums = self._build_sft_anchors(questions, golden, alpha)
            ce_datums = ce_datums + anchor_datums
            sdft_metrics["sdft/sft_anchor_alpha"] = alpha
            sdft_metrics["sdft/n_sft_anchors"] = float(len(anchor_datums))

        if self._frozen_teachers:
            sdft_metrics["sdft/n_frozen_teachers"] = float(len(self._frozen_teachers))
            n_ens = sum(1 for m in self._teacher_modes if m == "ensemble")
            n_rep = sum(1 for m in self._teacher_modes if m.startswith("frozen:"))
            sdft_metrics["sdft/n_ensemble_rows"] = float(n_ens)
            sdft_metrics["sdft/n_replay_rows"] = float(n_rep)

        return TrainingBatch(
            data=ce_datums,
            loss_fn="cross_entropy",
            metrics=sdft_metrics,
        )

    def _build_sft_anchors(
        self, questions: list[str], golden: list[str], alpha: float,
    ) -> list[tinker.Datum]:
        """Supervised golden-answer datums for the hybrid loss. Prompt shape is
        zero-shot by default (matches the benchmark's eval prompt), completion is
        ``<answer>SLUG</answer>`` + stop token so the student also learns to
        terminate. See :meth:`SDFTConfig.sft_anchor_alpha`."""
        stop_id = self._stop_sequences[0] if self._stop_sequences else None
        anchors: list[tinker.Datum] = []
        for q, g in zip(questions, golden):
            messages: list[dict[str, str]] = []
            if self.cfg.system_prompt:
                messages.append({"role": "system", "content": self.cfg.system_prompt})
            if not self.cfg.sft_anchor_zero_shot:
                for demo_q, demo_a in self._FEWSHOT:
                    messages.append({"role": "user", "content": demo_q})
                    messages.append({"role": "assistant", "content": f"<answer>{demo_a}</answer>"})
            messages.append({"role": "user", "content": self._student_user_content(q)})
            prompt_mi = self._renderer.build_generation_prompt(messages)
            golden_ids = self._tokenizer.encode(f"<answer>{g}</answer>", add_special_tokens=False)
            if stop_id is not None:
                golden_ids = list(golden_ids) + [int(stop_id)]
            anchors.append(build_sft_anchor_datum(
                prompt=prompt_mi, completion_tokens=golden_ids,
                topk=self.cfg.topk, weight=alpha,
            ))
        return anchors

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

    async def _query_teacher_topk(
        self,
        teacher_forced_seqs: list[tinker.ModelInput],
        teacher_modes: list[str],
    ) -> list[list[list[tuple[int, float]] | None] | None]:
        """Query one or many teachers per datum; ensemble modes average top-K probs."""
        async def _topk_for_datum(seq: tinker.ModelInput, mode: str):
            if mode == "ensemble":
                teachers = self._frozen_teachers + [self._teacher]
            elif mode.startswith("frozen:"):
                idx = int(mode.split(":", 1)[1])
                if idx < 0 or idx >= len(self._frozen_teachers):
                    raise RuntimeError(f"invalid frozen teacher index {idx} in mode {mode!r}")
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

    def _student_user_content(self, question: str) -> str:
        """The user turn the student sees (no demo)."""
        return self.cfg.user_template.format(question=question, prompt=question)

    def _truncate_at_answer(self, completion_tokens: list[int]) -> list[int]:
        """Cut a rollout at the first ``</answer>`` and re-append the stop token,
        so the training target is always ``<answer>SLUG</answer><|im_end|>`` —
        EOS pinned right after the answer, matching the clean SFT target. Cleanly
        terminated rollouts are unchanged (idempotent); drifted ones (doubled
        answer / stray ``</think>``) get their post-answer junk stripped. If no
        closing tag is present (rare) the rollout is left as sampled."""
        eos = self._stop_sequences[0] if self._stop_sequences else self._tokenizer.eos_token_id
        text = self._tokenizer.decode(completion_tokens)
        idx = text.find("</answer>")
        if idx == -1:
            return completion_tokens
        clean = text[: idx + len("</answer>")]
        toks = self._tokenizer.encode(clean, add_special_tokens=False)
        if eos is not None:
            toks = toks + [int(eos)]
        return toks

    # --- reverse-KL (paper-faithful) --------------------------------------

    async def _reverse_kl_batch(
        self, step_idx: int, rk_records: list, teacher_forced_seqs: list,
    ) -> TrainingBatch:
        """Paper reverse-KL (arXiv:2601.19897 Eq 1-2) as tinker importance_sampling.

        Per completion token y_t: advantage = log pi_teacher(y_t|x,c) - log
        pi_behavior(y_t|x). The teacher logprob is its per-token prompt_logprob
        over [teacher_prompt(with demo) + y]; the behavior logprob is the
        student sampler's own logprob at y_t. Gradient-ascending advantage*ratio
        (what importance_sampling does) descends the reverse KL."""
        teacher_responses = await asyncio.gather(*[
            self._teacher.sample_async(
                prompt=seq, params=tinker.SamplingParams(max_tokens=1),
                num_samples=1, include_prompt_logprobs=True,
            )
            for seq in teacher_forced_seqs
        ])
        datums: list[tinker.Datum] = []
        adv_sum = 0.0
        adv_n = 0
        tok_total = 0
        for (sp_ints, completion, behavior_lp, tp_len), tresp in zip(rk_records, teacher_responses):
            plp = getattr(tresp, "prompt_logprobs", None)
            if not plp:
                continue
            teacher_lp: list[float] = []
            ok = True
            for t in range(len(completion)):
                pos = tp_len + t
                if pos >= len(plp) or plp[pos] is None:
                    ok = False
                    break
                teacher_lp.append(float(plp[pos]))
            if not ok:
                continue
            datum = self._is_datum(sp_ints, completion, behavior_lp, teacher_lp)
            if datum is None:
                continue
            datums.append(datum)
            tok_total += len(completion)
            for t in range(self.cfg.skip_first_n_tokens, len(completion)):
                adv_sum += teacher_lp[t] - behavior_lp[t]
                adv_n += 1
        if not datums:
            raise RuntimeError(f"SDFT step {step_idx}: no reverse-KL datums built")
        metrics = {
            "sdft/num_datums": float(len(datums)),
            "sdft/total_completion_tokens": float(tok_total),
            "sdft/mean_advantage": float(adv_sum / adv_n) if adv_n else 0.0,
        }
        return TrainingBatch(data=datums, loss_fn="importance_sampling", metrics=metrics)

    def _is_datum(
        self, prompt_ints: list[int], completion: list[int],
        behavior_lp: list[float], teacher_lp: list[float],
    ) -> "tinker.Datum | None":
        """Build one importance_sampling Datum: target_tokens + per-position
        behavior logprobs (IS ratio) + advantages (0 off-completion and on the
        first skip_first_n boilerplate tokens → those positions don't train)."""
        import torch
        full = list(prompt_ints) + list(completion)
        if len(full) < 2:
            return None
        targets = full[1:]
        n = len(targets)
        logprobs = [0.0] * n
        advantages = [0.0] * n
        start = len(prompt_ints) - 1
        for t in range(len(completion)):
            j = start + t
            if j < 0 or j >= n or t < self.cfg.skip_first_n_tokens:
                continue
            logprobs[j] = float(behavior_lp[t])
            advantages[j] = float(teacher_lp[t] - behavior_lp[t])
        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints(full[:-1]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(torch.tensor(targets, dtype=torch.long)),
                "logprobs": tinker.TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
                "advantages": tinker.TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
            },
        )


__all__ = ["SDFT", "SDFTConfig"]

"""LocalSDFT — self-distillation fine-tuning on the `local` (HF + PEFT LoRA)
backend. A from-scratch custom training loop (not TRL's SFTTrainer), because
SDFT needs on-policy student rollouts + a frozen teacher forward pass each
step, which SFTTrainer has no hook for.

Method (Shenfeld et al. 2026, "Self-Distillation Enables Continual Learning";
see the Tinker-coupled reference at algorithms/sdft.py and the backend-agnostic
math helpers at training/sdft_data.py — this module lifts the same math into
plain HF tensors instead of tinker.Datum/ModelInput containers):

  1. Teacher prompt = system + user-with-golden-demo (the frozen teacher SEES
     the answer). Student prompt = system + user, no demo.
  2. Student (current trainable weights) generates an on-policy rollout from
     the student prompt (sampling, temperature, stop on EOS).
  3. Teacher-forced scoring: run the frozen teacher over
     teacher_prompt + completion, take top-K logprobs at each completion
     position.
  4. Loss = forward-KL(teacher top-K || student) at completion positions
     (skipping the first `skip_first_n_tokens`) — the student is scored via
     its OWN forward pass over student_prompt + completion (same completion
     tokens, different prompt prefix; aligned by position-within-completion,
     not absolute token position, since the two prompts differ in length).

Pre-conditions:
  * ctx.backend.name == 'local'
  * ctx.extras['backend_handles']['model'] / ['tokenizer'] are set
  * ctx.extras['train_rows'] contains SDFT rows: {'inputs': {'question': ...},
    'expected': <golden_answer>, 'metadata': {...}}

Hard-won lesson (see experiments/20260701_continual_sdft_paper/LOCAL_0.6B_HANDOFF.md):
SDFT cannot learn the task from a raw base model — base-model rollouts don't
follow the answer format and never stop, so the teacher's top-K distribution
over that rambling is an uninformative distillation signal. Always warm-start
from an SFT checkpoint via `init_from_checkpoint` (handled below the same way
as LocalSFT).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from ..protocols import RunContext, RunResult
from ..registry import register_algorithm

# Raise ImportError if PEFT is missing (keeps the `local` extra's contract:
# torch/transformers/peft/trl/datasets/accelerate all-or-nothing).
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402

logger = logging.getLogger(__name__)


DEFAULT_DEMO_TEMPLATE = (
    "{question}\n\n"
    "Example response:\n"
    "<answer>{golden_answer}</answer>\n\n"
    "Now answer with a response of your own, formatted the same way."
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a tool search engine. Match user queries to the correct API "
    "tool. Give your answer inside <answer></answer> tags."
)


class LocalSDFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 1.0e-4
    batch_size: int = 8
    max_steps: int = 300
    warmup_steps: int = 10
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42

    # SDFT knobs (mirrors evsys_sdk.algorithms.sdft.SDFTConfig)
    topk: int = 20
    skip_first_n_tokens: int = 3
    max_context_length: int = 1024
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    demo_template: str = DEFAULT_DEMO_TEMPLATE

    # Student rollout generation knobs
    max_tokens: int = 64
    temperature: float = 0.7
    """Lower than the Tinker reference's 1.0 — MPS/float32 + a 0.6B model
    drifts to rambling rollouts faster at temp=1.0 (see handoff lesson #4).
    Watch avg rollout length / EOS-termination rate as a health metric."""

    # --- from-base (no warm-start) support -------------------------------
    # By default SDFT REQUIRES init_from_checkpoint (a task-competent
    # warm-start): raw base rollouts are off-format, so the teacher's top-K
    # over them is an uninformative distillation signal (handoff §1). Set
    # allow_from_base=True ONLY when the student rollouts are made
    # well-formatted by other means (e.g. a few-shot format-inducing prompt,
    # student_fewshot below) so context-distillation can bake the task in
    # without SFT. When from-base, a fresh LoRA adapter is attached.
    allow_from_base: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Few-shot format-inducing examples prepended to the STUDENT rollout
    # prompt (list of [query, answer_slug] pairs). Purely shared format
    # scaffolding — the teacher keeps its per-query golden-answer demo, so its
    # information advantage is preserved; only the student's format prior
    # changes. Empty = current zero-shot student prompt (warm-start path).
    student_fewshot: list[tuple[str, str]] = Field(default_factory=list)


@register_algorithm("local_sdft")
class LocalSDFT:
    name: ClassVar[str] = "local_sdft"
    Config: ClassVar[type] = LocalSDFTConfig

    def __init__(self, **kwargs) -> None:
        self.cfg = LocalSDFTConfig.model_validate(kwargs)

    # -- data ---------------------------------------------------------------

    @staticmethod
    def _questions_and_answers(rows: list[dict]) -> tuple[list[str], list[str]]:
        questions, answers = [], []
        for r in rows:
            q = (r.get("inputs") or {}).get("question")
            a = r.get("expected")
            if not q or a is None:
                raise ValueError(
                    "LocalSDFT: rows must be SDFT-shaped "
                    "{'inputs': {'question': ...}, 'expected': ...} "
                    f"(got keys {list(r.keys())})"
                )
            questions.append(q)
            answers.append(str(a))
        return questions, answers

    def _build_prompts(self, tokenizer, question: str, golden_answer: str) -> tuple[str, str]:
        """Return (teacher_prompt_text, student_prompt_text), both chat-templated
        with add_generation_prompt=True and thinking disabled."""
        demo = self.cfg.demo_template.format(question=question, golden_answer=golden_answer)
        teacher_messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": demo},
        ]
        # Student prompt: optional few-shot format scaffolding (shared, generic
        # examples that DON'T reveal this query's answer) + the query. The
        # teacher still gets THIS query's golden answer via `demo`, so its
        # information advantage is preserved — the few-shot only fixes the
        # student's output FORMAT so its rollouts are well-formed and the
        # teacher's top-K lands on informative answer-token positions.
        student_messages = [{"role": "system", "content": self.cfg.system_prompt}]
        for fs_q, fs_a in self.cfg.student_fewshot:
            student_messages.append({"role": "user", "content": fs_q})
            student_messages.append({"role": "assistant", "content": f"<answer>{fs_a}</answer>"})
        student_messages.append({"role": "user", "content": question})
        teacher_text = tokenizer.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        student_text = tokenizer.apply_chat_template(
            student_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return teacher_text, student_text

    # -- train ----------------------------------------------------------------

    def train(self, ctx: RunContext) -> RunResult:
        if ctx.backend.name != "local":
            raise RuntimeError(f"LocalSDFT requires backend=local (got '{ctx.backend.name}')")

        handles = ctx.extras.get("backend_handles", {})
        base_model = handles.get("model")
        tokenizer = handles.get("tokenizer")
        device = handles.get("device", "cpu")
        if base_model is None or tokenizer is None:
            raise RuntimeError("LocalSDFT.train: backend_handles missing 'model' or 'tokenizer'")
        rows = ctx.extras.get("train_rows")
        if not rows:
            raise RuntimeError("LocalSDFT.train: ctx.extras['train_rows'] missing/empty")

        questions, answers = self._questions_and_answers(rows)
        n_rows = len(questions)

        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ctx.log_store.log_hyperparams({"algorithm": self.name, **self.cfg.model_dump()})

        init_from_checkpoint = handles.get("init_from_checkpoint")
        if not init_from_checkpoint and not self.cfg.allow_from_base:
            # Hard rule from the handoff (default): SDFT cannot learn the task
            # from a raw base model (uninformative teacher signal over
            # off-format rollouts). Fail loudly instead of silently training a
            # no-op. Set allow_from_base=True (+ a format-inducing
            # student_fewshot prompt) to intentionally test from-base.
            raise RuntimeError(
                "LocalSDFT.train: no init_from_checkpoint set. SDFT must be "
                "warm-started from an SFT checkpoint that already follows the "
                "<answer> format — training from a raw base model produces an "
                "uninformative distillation signal (see LOCAL_0.6B_HANDOFF.md §1). "
                "Set allow_from_base=True to override (only meaningful with a "
                "format-inducing student_fewshot prompt)."
            )

        # Lazy import: transformers is an optional `local` extra (mirrors the
        # lazy import in backends/local.py::LocalBackend.prepare).
        from transformers import AutoModelForCausalLM  # noqa: E402
        model_dtype = base_model.dtype if hasattr(base_model, "dtype") else torch.float32

        if init_from_checkpoint:
            # Warm-start path: student resumes the adapter (trainable); teacher
            # is a separate frozen copy of the SAME warm-started weights.
            student = PeftModel.from_pretrained(base_model, init_from_checkpoint, is_trainable=True)
            teacher_base = AutoModelForCausalLM.from_pretrained(
                handles.get("model_name"), dtype=model_dtype,
            )
            teacher = PeftModel.from_pretrained(teacher_base, init_from_checkpoint, is_trainable=False)
        else:
            # From-base path (allow_from_base): attach a FRESH LoRA adapter to
            # the base model for the student; the teacher is the raw base model
            # (frozen), which — critically — still SEES the golden answer via
            # its in-context demo (build_teacher_prompt), so context
            # distillation can transfer the task into the student's weights.
            lora_config = LoraConfig(
                r=self.cfg.lora_rank,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=self.cfg.lora_target_modules,
                task_type="CAUSAL_LM",
            )
            student = get_peft_model(base_model, lora_config)
            teacher = AutoModelForCausalLM.from_pretrained(
                handles.get("model_name"), dtype=model_dtype,
            )

        student.to(device)
        student.train()
        teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=self.cfg.learning_rate,
        )

        rng = random.Random(self.cfg.seed)
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

        loss_history: list[float] = []
        rollout_len_history: list[float] = []
        eos_term_history: list[float] = []

        for step in range(self.cfg.max_steps):
            batch_idx = [rng.randrange(n_rows) for _ in range(self.cfg.batch_size)]
            step_metrics = self._train_step(
                student=student,
                teacher=teacher,
                tokenizer=tokenizer,
                device=device,
                questions=[questions[i] for i in batch_idx],
                answers=[answers[i] for i in batch_idx],
                optimizer=optimizer,
                eos_id=eos_id,
                pad_id=pad_id,
            )
            if step_metrics is None:
                continue  # empty batch (all rollouts degenerate) — skip, don't crash the run
            loss_history.append(step_metrics["train/loss"])
            rollout_len_history.append(step_metrics["train/avg_rollout_len"])
            eos_term_history.append(step_metrics["train/eos_terminated_frac"])

            if (step + 1) % self.cfg.logging_steps == 0 or step == 0:
                ctx.log_store.log_metrics(step_metrics, step=step + 1)
                logger.info(
                    "LocalSDFT step %d/%d: loss=%.4f avg_rollout_len=%.1f eos_frac=%.2f",
                    step + 1, self.cfg.max_steps,
                    step_metrics["train/loss"], step_metrics["train/avg_rollout_len"],
                    step_metrics["train/eos_terminated_frac"],
                )

        final = out / "final"
        student.save_pretrained(str(final))
        tokenizer.save_pretrained(str(final))

        # Free the teacher's memory before returning (student/backend teardown
        # happens in the runner; the teacher is local to this call).
        del teacher
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        final_loss = loss_history[-1] if loss_history else 0.0
        metrics = {"train/final_loss": final_loss}
        if rollout_len_history:
            metrics["train/avg_rollout_len_overall"] = sum(rollout_len_history) / len(rollout_len_history)
        if eos_term_history:
            metrics["train/eos_terminated_frac_overall"] = sum(eos_term_history) / len(eos_term_history)

        # "state-final" is what Experiment._run_continual._final_state_checkpoint
        # looks for to chain this stage's adapter into the next.
        artifacts = {"final_checkpoint": str(final), "state-final": str(final)}
        for k, v in artifacts.items():
            ctx.log_store.log_artifact(k, v, kind="checkpoint")

        return RunResult(
            run_id=ctx.run_id,
            status="completed",
            metrics=metrics,
            artifacts=artifacts,
        )

    # -- one step -------------------------------------------------------------

    def _train_step(
        self,
        *,
        student,
        teacher,
        tokenizer,
        device: str,
        questions: list[str],
        answers: list[str],
        optimizer: torch.optim.Optimizer,
        eos_id: int | None,
        pad_id: int,
    ) -> dict[str, float] | None:
        cfg = self.cfg

        teacher_prompt_texts: list[str] = []
        student_prompt_texts: list[str] = []
        for q, a in zip(questions, answers):
            tp, sp = self._build_prompts(tokenizer, q, a)
            teacher_prompt_texts.append(tp)
            student_prompt_texts.append(sp)

        # --- 1. On-policy student rollouts (batched generate, left-padded). ---
        tokenizer.padding_side = "left"
        student_prompt_enc = tokenizer(
            student_prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(device)

        student.eval()
        with torch.no_grad():
            gen_out = student.generate(
                **student_prompt_enc,
                max_new_tokens=cfg.max_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=1.0,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        student.train()

        prompt_len = student_prompt_enc["input_ids"].shape[1]
        completions_padded = gen_out[:, prompt_len:]

        # Per-example: strip trailing pad, keep up to (and including) the first
        # EOS. Empty completions are dropped from the batch.
        completion_token_lists: list[list[int]] = []
        eos_terminated: list[bool] = []
        for row in completions_padded:
            toks = row.tolist()
            if eos_id is not None and eos_id in toks:
                cut = toks.index(eos_id) + 1  # keep the EOS token itself
                toks = toks[:cut]
                eos_terminated.append(True)
            else:
                eos_terminated.append(False)
            # Drop any leftover pad tokens (can appear pre-EOS-cut only if
            # pad==eos and index() already handled it; this is a defensive trim).
            if pad_id is not None and pad_id != eos_id:
                toks = [t for t in toks if t != pad_id]
            completion_token_lists.append(toks)

        keep = [i for i, t in enumerate(completion_token_lists) if len(t) > cfg.skip_first_n_tokens]
        if not keep:
            return None

        avg_rollout_len = sum(len(completion_token_lists[i]) for i in keep) / len(keep)
        eos_frac = sum(1 for i in keep if eos_terminated[i]) / len(keep)

        # --- 2. Teacher-forced forward pass: teacher_prompt + completion. ---
        teacher_seqs, teacher_prompt_lens = [], []
        for i in keep:
            t_ids = tokenizer(teacher_prompt_texts[i], add_special_tokens=False)["input_ids"]
            teacher_prompt_lens.append(len(t_ids))
            teacher_seqs.append(t_ids + completion_token_lists[i])

        student_seqs = []
        for i in keep:
            s_ids = tokenizer(student_prompt_texts[i], add_special_tokens=False)["input_ids"]
            student_seqs.append(s_ids + completion_token_lists[i])

        teacher_batch, teacher_attn = _pad_right(teacher_seqs, pad_id)
        student_batch, student_attn = _pad_right(student_seqs, pad_id)
        teacher_batch = teacher_batch.to(device)
        teacher_attn = teacher_attn.to(device)
        student_batch = student_batch.to(device)
        student_attn = student_attn.to(device)

        with torch.no_grad():
            teacher_out = teacher(input_ids=teacher_batch, attention_mask=teacher_attn)
        teacher_logits = teacher_out.logits  # (B, T_teacher, V)

        student_out = student(input_ids=student_batch, attention_mask=student_attn)
        student_logits = student_out.logits  # (B, T_student, V)

        # --- 3. Forward-KL(teacher topK || student) at completion positions. ---
        total_loss = torch.zeros((), device=device, dtype=student_logits.dtype)
        n_loss_positions = 0

        for b, i in enumerate(keep):
            completion_len = len(completion_token_lists[i])
            t_prompt_len = teacher_prompt_lens[b]
            s_prompt_len = len(student_seqs[b]) - completion_len

            for t in range(cfg.skip_first_n_tokens, completion_len):
                # Position in the sequence whose logits PREDICT completion
                # token t: logits at index (prompt_len + t - 1) predict the
                # token at index (prompt_len + t).
                t_pos = t_prompt_len + t - 1
                s_pos = s_prompt_len + t - 1
                if t_pos < 0 or t_pos >= teacher_logits.shape[1]:
                    continue
                if s_pos < 0 or s_pos >= student_logits.shape[1]:
                    continue

                t_logits_row = teacher_logits[b, t_pos]
                topk_vals, topk_idx = torch.topk(t_logits_row, k=min(cfg.topk, t_logits_row.shape[-1]))
                teacher_logprobs = F.log_softmax(topk_vals.float(), dim=-1)
                teacher_probs = teacher_logprobs.exp()

                s_logits_row = student_logits[b, s_pos].float()
                student_logprobs_full = F.log_softmax(s_logits_row, dim=-1)
                student_logprobs_at_topk = student_logprobs_full[topk_idx]

                # Forward-KL(teacher || student) restricted to the teacher's
                # top-K support == cross-entropy of student under the
                # (renormalized) teacher distribution over those K tokens.
                total_loss = total_loss - (teacher_probs * student_logprobs_at_topk).sum()
                n_loss_positions += 1

        if n_loss_positions == 0:
            return None

        loss = total_loss / n_loss_positions
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], max_norm=1.0,
        )
        optimizer.step()

        return {
            "train/loss": float(loss.detach().item()),
            "train/avg_rollout_len": float(avg_rollout_len),
            "train/eos_terminated_frac": float(eos_frac),
            "train/n_loss_positions": float(n_loss_positions),
            "train/batch_kept": float(len(keep)),
        }


def _pad_right(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
    return input_ids, attn


__all__ = ["LocalSDFT", "LocalSDFTConfig"]

"""LocalSDFTPaper — FAITHFUL reproduction of the SDFT paper's method
(Shenfeld et al. 2026, "Self-Distillation Enables Continual Learning",
arXiv:2601.19897), local `local` backend (HF + PEFT LoRA), keeping LoRA
instead of the paper's full fine-tuning.

This is DELIBERATELY SEPARATE from `local_sdft` (which is a simplified variant
that prior experiments depend on). It differs from local_sdft on three core
axes to match the paper:

  1. TEACHER = EMA of the student parameters (exponential moving average),
     updated every step: teacher <- decay*teacher + (1-decay)*student. NOT a
     frozen snapshot. The teacher is the student's own (slowly-tracking)
     weights, conditioned on the demonstration.
  2. LOSS = REVERSE KL, on-policy REINFORCE form (paper Eq 1-2):
     objective  min_theta  E_{y~pi_student(.|x)} [ sum_t log( pi_student(y_t|.) /
                                                              pi_teacher(y_t|.,c) ) ]
     Uses the FULL-vocab sampled-token log-ratio (NOT top-K, NOT the forward-KL
     soft-target form of local_sdft). Gradient estimator (score-function /
     REINFORCE) documented inline in `_train_step`.
  3. TEACHER PROMPT = the paper's exact template; student is conditioned ONLY on
     the query (pi(.|x)), no demo / no few-shot scaffold. CoT is elicited via
     the paper's prompt text ("...including the thinking process") + large
     max_tokens.

No SFT warm-start, no few-shot prompt trick (an Instruct model is assumed
competent enough — that's the point of the paper's design).

Pre-conditions:
  * ctx.backend.name == 'local'
  * ctx.extras['backend_handles']['model'] / ['tokenizer'] set
  * ctx.extras['train_rows'] are SDFT rows:
    {'inputs': {'question': ...}, 'expected': <golden_answer>, ...}
"""

from __future__ import annotations

import copy
import logging
import random
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from ..protocols import RunContext, RunResult
from ..registry import register_algorithm

from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402

logger = logging.getLogger(__name__)


# Paper's exact teacher-prompt body (page 3-5): Question + demonstration + the
# "including the thinking process" CoT instruction.
PAPER_DEMO_TEMPLATE = (
    "{question}\n"
    "This is an example for a response to the question:\n"
    "{golden_answer}\n"
    "Now answer with a response of your own, including the thinking process:"
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a tool search engine. Match user queries to the correct API "
    "tool. Give your answer inside <answer></answer> tags."
)


class LocalSDFTPaperConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 1.0e-4
    batch_size: int = 4
    max_steps: int = 300
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42

    # --- faithful-paper knobs ---
    ema_decay: float = 0.99
    """Teacher EMA decay: teacher <- decay*teacher + (1-decay)*student."""
    demo_template: str = PAPER_DEMO_TEMPLATE
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    enable_thinking: bool = True
    """Paper elicits CoT. Note: Qwen2.5 templates have no <think> block, so CoT
    is driven by the prompt text; the flag is forwarded to apply_chat_template
    only when the tokenizer accepts it (Qwen3), else ignored."""

    # Student rollout generation
    max_tokens: int = 512
    """Large: CoT needs room. Watch avg rollout length / eos_frac — a 0.5B model
    with thinking may ramble to max_tokens (a key finding to report, not hide)."""
    temperature: float = 1.0

    # LoRA (fresh adapter — no warm-start in the paper method)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@register_algorithm("local_sdft_paper")
class LocalSDFTPaper:
    name: ClassVar[str] = "local_sdft_paper"
    Config: ClassVar[type] = LocalSDFTPaperConfig

    def __init__(self, **kwargs) -> None:
        self.cfg = LocalSDFTPaperConfig.model_validate(kwargs)

    # -- data ---------------------------------------------------------------

    @staticmethod
    def _questions_and_answers(rows: list[dict]) -> tuple[list[str], list[str]]:
        qs, ans = [], []
        for r in rows:
            q = (r.get("inputs") or {}).get("question")
            a = r.get("expected")
            if not q or a is None:
                raise ValueError(
                    "LocalSDFTPaper: rows must be SDFT-shaped "
                    "{'inputs':{'question':...},'expected':...} "
                    f"(got {list(r.keys())})"
                )
            qs.append(q)
            ans.append(str(a))
        return qs, ans

    def _apply_template(self, tokenizer, messages, add_generation_prompt=True):
        kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
        # enable_thinking only for tokenizers that accept it (Qwen3); Qwen2.5
        # ignores it (no <think> block) so we don't pass it there.
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=self.cfg.enable_thinking, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    def _prompts(self, tokenizer, question: str, golden: str) -> tuple[str, str]:
        """(teacher_prompt_with_demo, student_prompt_query_only)."""
        demo = self.cfg.demo_template.format(question=question, golden_answer=golden)
        teacher_msgs = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": demo},
        ]
        student_msgs = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": question},
        ]
        return (self._apply_template(tokenizer, teacher_msgs),
                self._apply_template(tokenizer, student_msgs))

    # -- train --------------------------------------------------------------

    def train(self, ctx: RunContext) -> RunResult:
        if ctx.backend.name != "local":
            raise RuntimeError(f"LocalSDFTPaper requires backend=local (got '{ctx.backend.name}')")

        handles = ctx.extras.get("backend_handles", {})
        base_model = handles.get("model")
        tokenizer = handles.get("tokenizer")
        device = handles.get("device", "cpu")
        if base_model is None or tokenizer is None:
            raise RuntimeError("LocalSDFTPaper.train: backend_handles missing model/tokenizer")
        rows = ctx.extras.get("train_rows")
        if not rows:
            raise RuntimeError("LocalSDFTPaper.train: ctx.extras['train_rows'] missing/empty")

        questions, answers = self._questions_and_answers(rows)
        n_rows = len(questions)
        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ctx.log_store.log_hyperparams({"algorithm": self.name, **self.cfg.model_dump()})

        from transformers import AutoModelForCausalLM  # noqa: E402  optional `local` extra
        model_dtype = base_model.dtype if hasattr(base_model, "dtype") else torch.float32

        init_from_checkpoint = handles.get("init_from_checkpoint")
        if init_from_checkpoint:
            # Continual chaining: resume the prior stage's adapter (trainable).
            student = PeftModel.from_pretrained(base_model, init_from_checkpoint, is_trainable=True)
        else:
            lora = LoraConfig(
                r=self.cfg.lora_rank, lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout, target_modules=self.cfg.lora_target_modules,
                task_type="CAUSAL_LM",
            )
            student = get_peft_model(base_model, lora)
        student.to(device)
        student.train()

        # EMA teacher: a SEPARATE full model instance whose weights track the
        # student's via EMA. Start it equal to the student's current weights.
        # We keep only ONE merged-style teacher module and EMA its trainable
        # LoRA-affected params. Simplest robust approach on LoRA: keep a second
        # PeftModel with the SAME structure and EMA its LoRA parameters.
        teacher_base = AutoModelForCausalLM.from_pretrained(handles.get("model_name"), dtype=model_dtype)
        if init_from_checkpoint:
            teacher = PeftModel.from_pretrained(teacher_base, init_from_checkpoint, is_trainable=False)
        else:
            # give the teacher an identically-configured fresh adapter, then copy
            # the student's current LoRA weights into it as the EMA starting point.
            teacher = get_peft_model(teacher_base, copy.deepcopy(student.peft_config["default"]))
        teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # Map student<->teacher trainable (LoRA) params by name for EMA updates.
        student_lora = {n: p for n, p in student.named_parameters() if p.requires_grad}
        teacher_named = dict(teacher.named_parameters())
        ema_pairs = [(teacher_named[n], student_lora[n]) for n in student_lora if n in teacher_named]
        if not ema_pairs:
            raise RuntimeError("LocalSDFTPaper: could not align student/teacher LoRA params for EMA")
        # Initialize teacher LoRA = student LoRA (EMA seed).
        with torch.no_grad():
            for tp, sp in ema_pairs:
                tp.copy_(sp.data)

        optimizer = torch.optim.AdamW([p for _, p in student_lora.items()], lr=self.cfg.learning_rate)
        rng = random.Random(self.cfg.seed)
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

        loss_hist, rl_hist, eos_hist = [], [], []
        decay = self.cfg.ema_decay

        for step in range(self.cfg.max_steps):
            idx = [rng.randrange(n_rows) for _ in range(self.cfg.batch_size)]
            m = self._train_step(
                student=student, teacher=teacher, tokenizer=tokenizer, device=device,
                questions=[questions[i] for i in idx], answers=[answers[i] for i in idx],
                optimizer=optimizer, eos_id=eos_id, pad_id=pad_id,
            )
            # EMA teacher update AFTER the student step.
            with torch.no_grad():
                for tp, sp in ema_pairs:
                    tp.mul_(decay).add_(sp.data, alpha=1.0 - decay)

            if m is None:
                continue
            loss_hist.append(m["train/loss"]); rl_hist.append(m["train/avg_rollout_len"]); eos_hist.append(m["train/eos_terminated_frac"])
            if (step + 1) % self.cfg.logging_steps == 0 or step == 0:
                ctx.log_store.log_metrics(m, step=step + 1)
                logger.info(
                    "LocalSDFTPaper step %d/%d: loss=%.4f avg_rollout_len=%.1f eos_frac=%.2f",
                    step + 1, self.cfg.max_steps, m["train/loss"], m["train/avg_rollout_len"], m["train/eos_terminated_frac"],
                )

        final = out / "final"
        student.save_pretrained(str(final))
        tokenizer.save_pretrained(str(final))
        del teacher, teacher_base
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        metrics = {"train/final_loss": loss_hist[-1] if loss_hist else 0.0}
        if rl_hist:
            metrics["train/avg_rollout_len_overall"] = sum(rl_hist) / len(rl_hist)
        if eos_hist:
            metrics["train/eos_terminated_frac_overall"] = sum(eos_hist) / len(eos_hist)
        artifacts = {"final_checkpoint": str(final), "state-final": str(final)}
        for k, v in artifacts.items():
            ctx.log_store.log_artifact(k, v, kind="checkpoint")
        return RunResult(run_id=ctx.run_id, status="completed", metrics=metrics, artifacts=artifacts)

    # -- one step: on-policy reverse-KL REINFORCE ---------------------------

    def _train_step(self, *, student, teacher, tokenizer, device, questions, answers,
                    optimizer, eos_id, pad_id) -> dict[str, float] | None:
        cfg = self.cfg
        teacher_prompts, student_prompts = [], []
        for q, a in zip(questions, answers):
            tp, sp = self._prompts(tokenizer, q, a)
            teacher_prompts.append(tp); student_prompts.append(sp)

        # 1. Sample y ~ pi_student(.|x)  (student prompt = query only).
        tokenizer.padding_side = "left"
        enc = tokenizer(student_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        student.eval()
        with torch.no_grad():
            gen = student.generate(
                **enc, max_new_tokens=cfg.max_tokens, do_sample=True,
                temperature=cfg.temperature, top_p=1.0,
                pad_token_id=pad_id, eos_token_id=eos_id,
            )
        student.train()
        plen = enc["input_ids"].shape[1]
        comps = []
        eos_term = []
        for row in gen[:, plen:]:
            toks = row.tolist()
            if eos_id is not None and eos_id in toks:
                toks = toks[: toks.index(eos_id) + 1]; eos_term.append(True)
            else:
                eos_term.append(False)
            if pad_id is not None and pad_id != eos_id:
                toks = [t for t in toks if t != pad_id]
            comps.append(toks)
        keep = [i for i, t in enumerate(comps) if len(t) >= 1]
        if not keep:
            return None
        avg_len = sum(len(comps[i]) for i in keep) / len(keep)
        eos_frac = sum(1 for i in keep if eos_term[i]) / len(keep)

        # 2. Build teacher-forced sequences: student = [student_prompt + y],
        #    teacher = [teacher_prompt(with demo) + y]. Same y; different prefix,
        #    aligned by completion position (t-th completion token).
        s_seqs, s_plens, t_seqs, t_plens = [], [], [], []
        for i in keep:
            s_ids = tokenizer(student_prompts[i], add_special_tokens=False)["input_ids"]
            t_ids = tokenizer(teacher_prompts[i], add_special_tokens=False)["input_ids"]
            s_plens.append(len(s_ids)); t_plens.append(len(t_ids))
            s_seqs.append(s_ids + comps[i]); t_seqs.append(t_ids + comps[i])
        s_batch, s_attn = _pad_right(s_seqs, pad_id)
        t_batch, t_attn = _pad_right(t_seqs, pad_id)
        s_batch, s_attn = s_batch.to(device), s_attn.to(device)
        t_batch, t_attn = t_batch.to(device), t_attn.to(device)

        # student forward WITH grad; teacher forward NO grad.
        s_out = student(input_ids=s_batch, attention_mask=s_attn).logits
        with torch.no_grad():
            t_out = teacher(input_ids=t_batch, attention_mask=t_attn).logits

        # 3. Reverse-KL REINFORCE loss (paper Eq 1-2). For each sampled token y_t:
        #      logp_s = log pi_student(y_t | x, y_<t)          (grad flows)
        #      logp_t = log pi_teacher(y_t | x, c, y_<t)       (detached)
        #      per-token log-ratio  r_t = logp_s.detach() - logp_t   (the "reward"
        #        being minimized; = reverse-KL integrand under the sample)
        #    Objective  min E_y[sum_t r_t].  Score-function gradient of
        #    E_{y~pi_s}[R(y)] with R = sum_t (logp_s - logp_t) is
        #      grad = E[ (sum_t logp_s) * R.detach() + grad(sum_t (logp_s - logp_t)) ].
        #    We use the standard per-token surrogate:
        #      loss = mean_t [ logp_s * ratio_t.detach()  +  (logp_s - logp_t) ]
        #    whose gradient is the REINFORCE estimator of the reverse-KL gradient
        #    (first term = score-function on the sampled token weighted by the
        #    log-ratio advantage; second = the pathwise term). logp_t is fully
        #    detached (teacher = EMA, no grad).
        total = torch.zeros((), device=device, dtype=s_out.dtype)
        ntok = 0
        for b, i in enumerate(keep):
            clen = len(comps[i])
            sp0, tp0 = s_plens[b], t_plens[b]
            for tpos in range(clen):
                s_logit_pos = sp0 + tpos - 1
                t_logit_pos = tp0 + tpos - 1
                if s_logit_pos < 0 or s_logit_pos >= s_out.shape[1]:
                    continue
                if t_logit_pos < 0 or t_logit_pos >= t_out.shape[1]:
                    continue
                y_t = comps[i][tpos]
                s_logp_full = F.log_softmax(s_out[b, s_logit_pos].float(), dim=-1)
                t_logp_full = F.log_softmax(t_out[b, t_logit_pos].float(), dim=-1)
                logp_s = s_logp_full[y_t]
                logp_t = t_logp_full[y_t]
                ratio = (logp_s.detach() - logp_t.detach())
                total = total + logp_s * ratio + (logp_s - logp_t.detach())
                ntok += 1
        if ntok == 0:
            return None
        loss = total / ntok
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in student.named_parameters() if p.requires_grad], 1.0)
        optimizer.step()
        return {
            "train/loss": float(loss.detach().item()),
            "train/avg_rollout_len": float(avg_len),
            "train/eos_terminated_frac": float(eos_frac),
            "train/n_loss_tokens": float(ntok),
            "train/batch_kept": float(len(keep)),
        }


def _pad_right(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    ml = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), ml), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), ml), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
    return ids, attn


__all__ = ["LocalSDFTPaper", "LocalSDFTPaperConfig"]

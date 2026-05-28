"""SDPO (Self-Distillation Policy Optimization) — Tinker variant.

Background. The reference SDPO (lasgroup/SDPO, built on verl) is *token-level
full-logit* self-distillation: a feedback-conditioned "teacher" forward and a
no-feedback "student" forward are run over the same response tokens, and the
policy is trained on the per-token KL/JSD between their full-vocab
distributions (`actor.self_distillation.full_logit_distillation`).

Why this is the *surrogate* variant. Tinker is a managed training API: it
exposes `forward_backward_custom` whose custom loss receives only the
**log-probs of the target tokens** (`loss_type_input="logprobs"`), never full
or top-k vocab distributions with gradients (verified against tinker 0.22.1).
So full-logit / top-k distillation is not trainable on Tinker. What *is*
expressible is SDPO's selected-token branch from its `core_algos.py`:

    per_token_loss = (student_logp - teacher_logp).detach() * student_logp

i.e. a per-token, REINFORCE-shaped surrogate using only response-token
log-probs. The teacher term is detached (consistent with the reference), so we
can obtain it from a forward-only feedback-conditioned pass.

Data. Plain multi-turn chat (`messages`), no target labels: for each assistant
turn that is followed by a user turn, the assistant turn is the *response*, and
that next user turn is the *feedback* the teacher conditions on (the student
does not see it). See `slice_chat_for_sdpo`.

The Tinker training loop needs a live `TINKER_API_KEY` and is not exercised in
CI; the testable cores (`slice_chat_for_sdpo`, `sdpo_surrogate_per_token_loss`)
are pure and unit-tested.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..protocols import RunContext, RunResult
from ..registry import register_algorithm

logger = logging.getLogger(__name__)


class SDPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = 1e-5
    num_steps: int = 100
    batch_size: int = 16
    """Number of (student, teacher, response) records per optimizer step."""
    lora_rank: int = 8
    save_every: int = 50
    max_tokens: int = 512
    feedback_prefix: str = "Feedback to incorporate: "
    """Prepended to the next-user-turn feedback in the teacher context."""
    min_feedback_chars: int = 1
    """Skip records whose feedback turn is shorter than this (noise filter)."""
    renderer_name: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None


# ---------------------------------------------------------------------------
# Testable cores (pure — no Tinker, no network).
# ---------------------------------------------------------------------------


def slice_chat_for_sdpo(
    messages: list[dict[str, Any]],
    *,
    feedback_prefix: str = "Feedback to incorporate: ",
    min_feedback_chars: int = 1,
) -> list[dict[str, Any]]:
    """Turn one conversation into SDPO records.

    For each assistant turn at index ``i`` that is immediately followed by a
    user turn at ``i+1``, emit a record:

      * ``response``          — the assistant turn's content (tokens trained on)
      * ``student_messages``  — history up to (not including) the assistant turn
      * ``teacher_messages``  — same history plus the *next user turn* injected
        as feedback (this is what the teacher conditions on; the student does not)

    The feedback is the next user message — no labels/rewards required.
    """
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if i + 1 >= len(messages) or messages[i + 1].get("role") != "user":
            continue
        feedback = str(messages[i + 1].get("content", "") or "")
        if len(feedback.strip()) < min_feedback_chars:
            continue
        history = messages[:i]
        teacher_messages = [
            *history,
            {"role": "user", "content": f"{feedback_prefix}{feedback}"},
        ]
        out.append({
            "response": str(msg.get("content", "") or ""),
            "student_messages": history,
            "teacher_messages": teacher_messages,
        })
    return out


def sdpo_surrogate_per_token_loss(student_logprobs, teacher_logprobs, mask=None):
    """SDPO selected-token surrogate loss (the only Tinker-expressible variant).

    ``per_token = (student_logp - teacher_logp).detach() * student_logp`` summed
    (masked) over response tokens. ``student_logprobs`` carries gradient;
    ``teacher_logprobs`` is treated as a constant (detached), matching SDPO's
    reference selected-token branch.

    Args are 1-D tensors of equal length over the response tokens. Returns
    ``(loss, metrics)``.
    """
    import torch

    student = student_logprobs
    teacher = teacher_logprobs.detach() if hasattr(teacher_logprobs, "detach") else torch.as_tensor(teacher_logprobs)
    log_ratio = (student - teacher).detach()
    per_token = log_ratio * student
    if mask is not None:
        m = mask.to(per_token.dtype)
        denom = torch.clamp_min(m.sum(), 1.0)
        loss = (per_token * m).sum() / denom
        mean_log_ratio = ((student - teacher) * m).sum().detach() / denom
    else:
        loss = per_token.mean()
        mean_log_ratio = (student - teacher).mean().detach()
    metrics = {"sdpo/mean_log_ratio": float(mean_log_ratio)}
    return loss, metrics


# ---------------------------------------------------------------------------
# Algorithm (Tinker loop — needs TINKER_API_KEY; not covered by CI).
# ---------------------------------------------------------------------------


@register_algorithm("sdpo")
class SDPO:
    name: ClassVar[str] = "sdpo"
    Config: ClassVar[type] = SDPOConfig

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = SDPOConfig.model_validate(kwargs)

    def _records(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        for row in rows:
            msgs = row.get("messages")
            if isinstance(msgs, list):
                recs.extend(slice_chat_for_sdpo(
                    msgs,
                    feedback_prefix=self.cfg.feedback_prefix,
                    min_feedback_chars=self.cfg.min_feedback_chars,
                ))
        return recs

    def train(self, ctx: RunContext) -> RunResult:
        if ctx.backend.name != "tinker":
            raise RuntimeError(f"SDPO requires backend=tinker (got '{ctx.backend.name}')")
        rows = ctx.extras.get("train_rows") or []
        records = self._records(rows)
        if not records:
            return RunResult(
                run_id=ctx.run_id, status="failed",
                error="SDPO found no (assistant→user) feedback pairs in the chat data.",
            )

        try:
            import tinker  # noqa: F401
            from tinker_cookbook.tokenizer_utils import get_tokenizer
        except ImportError as e:
            return RunResult(run_id=ctx.run_id, status="failed", error=f"tinker not installed: {e}")

        handles = ctx.extras.get("backend_handles", {})
        model_name = handles.get("model_name") or ctx.extras.get("model_name")
        service_client = handles.get("service_client")
        if service_client is None or not model_name:
            return RunResult(run_id=ctx.run_id, status="failed", error="tinker handles missing (service_client/model_name)")

        ctx.log_store.log_hyperparams({
            "algorithm": self.name, "model_name": model_name,
            "n_records": len(records), **self.cfg.model_dump(),
        })

        tokenizer = get_tokenizer(model_name)
        training_client = service_client.create_lora_training_client(
            base_model=model_name, rank=self.cfg.lora_rank,
        )
        sampling_client = training_client.save_weights_and_get_sampling_client(name=f"{ctx.run_id}-init")

        try:
            self._run_loop(ctx, training_client, sampling_client, tokenizer, records)
        except Exception as e:
            logger.exception("SDPO.train loop failed")
            return RunResult(run_id=ctx.run_id, status="failed", error=str(e))

        artifacts: dict[str, str] = {}
        try:
            final = training_client.save_state(name=f"{ctx.run_id}-final").result()
            path = getattr(final, "path", None) or (final.get("path") if isinstance(final, dict) else None)
            if path:
                artifacts["final_checkpoint"] = path
                ctx.log_store.log_artifact("final_checkpoint", path, kind="checkpoint")
        except Exception as e:
            logger.warning("SDPO: final save_state failed: %s", e)
        return RunResult(run_id=ctx.run_id, status="completed", artifacts=artifacts)

    def _run_loop(self, ctx, training_client, sampling_client, tokenizer, records):
        """One optimizer step per batch of records.

        For each record: teacher log-probs of the response tokens under the
        feedback-conditioned context come from a forward-only pass (detached);
        the student's response-token log-probs are produced (with gradient) by
        `forward_backward_custom`, whose loss applies
        `sdpo_surrogate_per_token_loss`.

        NB: the exact datum packing / log-prob alignment for
        `forward_backward_custom` depends on the live Tinker API and is
        validated against a real TINKER_API_KEY, not in CI.
        """
        import tinker

        def render(messages: list[dict[str, Any]]) -> list[int]:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return tokenizer.encode(text)

        adam = tinker.AdamParams(learning_rate=self.cfg.learning_rate)
        bs = self.cfg.batch_size
        for step in range(self.cfg.num_steps):
            batch = records[(step * bs) % len(records):][:bs] or records[:bs]
            data, teacher_lp = [], []
            for rec in batch:
                resp_ids = tokenizer.encode(rec["response"])[: self.cfg.max_tokens]
                student_ids = render(rec["student_messages"])
                teacher_ids = render(rec["teacher_messages"])
                # Teacher: forward-only log-probs of the response tokens (detached).
                t_lp = sampling_client.compute_logprobs(
                    tinker.ModelInput.from_ints(teacher_ids + resp_ids)
                ).result()
                teacher_lp.append([x for x in t_lp[-len(resp_ids):] if x is not None])
                full_ids = student_ids + resp_ids
                data.append(tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(full_ids),
                    loss_fn_inputs={"target_tokens": [*full_ids[1:], resp_ids[-1]]},
                ))

            def loss_fn(_data, logprobs_list, _teacher=teacher_lp):
                import torch
                total, n = None, 0
                for student_lp, t in zip(logprobs_list, _teacher, strict=False):
                    r = len(t)
                    if r == 0 or len(student_lp) == 0:
                        continue
                    # student_lp is per-position over the full sequence; the
                    # response logprobs are the last r positions.
                    s = student_lp[-r:]
                    k = min(len(s), r)
                    tt = torch.as_tensor(t[-k:], dtype=torch.float32)
                    loss, _ = sdpo_surrogate_per_token_loss(s[-k:], tt)
                    total = loss if total is None else total + loss
                    n += 1
                total = total / max(n, 1)
                return total, {"sdpo/batch_records": float(n)}

            fb = training_client.forward_backward_custom(data, loss_fn).result()
            training_client.optim_step(adam).result()
            metrics = {"sdpo/step": float(step), **{k: float(v) for k, v in getattr(fb, "metrics", {}).items()}}
            ctx.log_store.log_metrics(metrics, step=step)
            if self.cfg.save_every and step and step % self.cfg.save_every == 0:
                try:
                    training_client.save_state(name=f"{ctx.run_id}-step{step}").result()
                except Exception as e:
                    logger.warning("SDPO: save_state @%d failed: %s", step, e)

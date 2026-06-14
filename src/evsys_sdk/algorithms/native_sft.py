"""NativeSFT — supervised fine-tuning on the native training loop.

Replaces :class:`~evsys_sdk.algorithms.tinker_sft.TinkerSFT`. Where the
older wrapper delegated the entire loop to ``tinker_cookbook``, this
composer wires the three pieces from :mod:`evsys_sdk.training`:

* :class:`~evsys_sdk.training.tinker_backend.TinkerBackend` — talks to tinker
* :class:`~evsys_sdk.training.step_builder.SFTStepBuilder` — slices a
  pre-tokenized list of :class:`tinker.Datum` per step, computes
  ``train_mean_nll``
* :class:`~evsys_sdk.training.loop.TrainingLoop` — drives the for-loop,
  pipelining, checkpoint cadence, in-loop eval cadence, metric writes

…and that's the whole algorithm. Researchers who want a one-line tweak
(focal loss, custom loss, extra metrics) subclass ``SFTStepBuilder`` and
hand it to a ``NativeSFT`` re-register — no SDK code change needed.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import tinker
from pydantic import BaseModel, ConfigDict, Field

from ..data_types import ChatMessagesRow, TargetFormat, parse_rows
from ..protocols import RunContext, RunResult
from ..registry import register_algorithm
from ..training.evaluators import build_in_loop_evaluators
from ..training.loop import TrainingLoop
from ..training.sft_data import sft_tokenize
from ..training.step_builder import SFTStepBuilder
from ..training.tinker_backend import TinkerBackend


# ---------------------------------------------------------------------------
# Config — same shape as TinkerSFTConfig modulo cookbook-only knobs
# ---------------------------------------------------------------------------


class NativeSFTConfig(BaseModel):
    """Pydantic config for :class:`NativeSFT`.

    ``extra="forbid"`` so a typo in the YAML config fails loudly. Field
    semantics match ``TinkerSFTConfig`` so existing experiment configs
    flip ``algorithm.kind: tinker_sft`` → ``native_sft`` without touching
    anything else.
    """

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 1.0e-4
    num_epochs: int = 1
    batch_size: int = 4
    max_steps: int | None = None
    lora_rank: int = 8
    max_seq_len: int = 2048

    # Which assistant turns the loss is computed on. Lives here (algorithm
    # config), NOT on the dataset — ChatMessagesRow carries only the
    # conversation; the algorithm decides what to supervise.
    supervise: Literal["all_assistant", "last_assistant"] = "all_assistant"

    # checkpoint cadence — same resolution rules as TinkerSFT
    save_every: int = 0
    """If 0, computed from save_at_fractions."""
    save_at_fractions: list[float] = Field(default_factory=lambda: [1.0])

    # in-loop eval cadence (separate from post-training Benchmark eval that
    # Experiment runs); 0 disables.
    eval_every: int = 0

    # renderer (HF chat-template variant); enable_thinking gated separately so
    # non-Qwen tokenizers keep working — see training/templates.py.
    renderer_name: str | None = None
    enable_thinking: bool | None = None

    # Adam knobs are passthrough.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8

    # Optional wandb hookup (off by default; the SDK doesn't pull wandb).
    wandb_project: str | None = None
    wandb_name: str | None = None


# ---------------------------------------------------------------------------
# Algorithm composer
# ---------------------------------------------------------------------------


@register_algorithm("native_sft")
class NativeSFT:
    name: ClassVar[str] = "native_sft"
    Config: ClassVar[type] = NativeSFTConfig

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = NativeSFTConfig.model_validate(kwargs)

    # --- public entry point (sync signature; runs the async loop inside) ----

    def train(self, ctx: RunContext) -> RunResult:
        if ctx.backend.name != "tinker":
            raise RuntimeError(
                f"NativeSFT requires backend=tinker (got '{ctx.backend.name}'). "
                "Use mock_sft for mock backends."
            )
        return asyncio.run(self._train_async(ctx))

    async def _train_async(self, ctx: RunContext) -> RunResult:
        rows = ctx.extras.get("train_rows")
        if not rows:
            raise RuntimeError("NativeSFT.train: ctx.extras['train_rows'] missing/empty")

        handles = ctx.extras.get("backend_handles", {})
        model_name = handles.get("model_name") or ctx.extras.get("model_name")
        if not model_name:
            raise RuntimeError("model_name not set in backend handles")

        # 1. backend (async factory; allocates the LoRA training client)
        backend = await TinkerBackend.create(
            model_name=model_name,
            lora_rank=self.cfg.lora_rank,
            renderer_name=self.cfg.renderer_name or handles.get("renderer_name"),
            resume_state_path=handles.get("load_checkpoint_path"),
        )

        # 2. standardize raw rows → typed ChatMessagesRow (strict), then
        #    tokenize → list of Datum with assistant-span loss masks. The
        #    supervise decision is the algorithm's, not the dataset's.
        chat_rows = cast("list[ChatMessagesRow]", parse_rows(rows, TargetFormat.CHAT_MESSAGES))
        datums = sft_tokenize(
            chat_rows, backend.get_tokenizer(),
            max_seq_len=self.cfg.max_seq_len,
            enable_thinking=self.cfg.enable_thinking,
            supervise=self.cfg.supervise,
        )

        # 3. total step count + save cadence
        n_rows = len(rows)
        steps_per_epoch = max(1, n_rows // self.cfg.batch_size)
        total_steps = (
            self.cfg.max_steps
            if self.cfg.max_steps is not None
            else steps_per_epoch * self.cfg.num_epochs
        )
        save_every = self._resolve_save_every(total_steps)

        # 4. log hyperparams once so the experiment record carries them
        ctx.log_store.log_hyperparams({
            "algorithm": self.name,
            **self.cfg.model_dump(),
            "model_name": model_name,
            "n_train_rows": n_rows,
            "n_train_datums": len(datums),
            "total_steps": total_steps,
            "save_every": save_every,
        })

        # 5. compose the loop and run
        evaluators = build_in_loop_evaluators(
            ctx.config.metadata if hasattr(ctx, "config") else None,
            tokenizer=backend.get_tokenizer(),
            store=getattr(ctx, "store", None) or ctx.extras.get("store"),
        )
        loop = TrainingLoop(
            backend=backend,
            step_builder=SFTStepBuilder(
                datums=datums, batch_size=self.cfg.batch_size,
            ),
            log_store=ctx.log_store,
            output_dir=Path(ctx.output_dir),
            adam_params=tinker.AdamParams(
                learning_rate=self.cfg.learning_rate,
                beta1=self.cfg.adam_beta1,
                beta2=self.cfg.adam_beta2,
                eps=self.cfg.adam_eps,
            ),
            save_every=save_every,
            eval_every=self.cfg.eval_every,
            evaluators=evaluators,
        )
        artifacts = await loop.run(num_steps=total_steps)

        # 6. record the run_dir + per-checkpoint sampler URIs as artifacts
        # so downstream consumers (TinkerInference.from_run_result,
        # Experiment._eval_arm) keep working unchanged.
        artifacts_dict = artifacts.as_dict()
        for key, value in artifacts_dict.items():
            ctx.log_store.log_artifact(key, value, kind="checkpoint")

        return RunResult(
            run_id=ctx.run_id,
            status="completed",
            metrics={},
            artifacts=artifacts_dict,
        )

    # --- save cadence helper (same shape as TinkerSFT) ---------------------

    def _resolve_save_every(self, total_steps: int) -> int:
        if self.cfg.save_every:
            return self.cfg.save_every
        marks = sorted({
            max(1, int(round(f * total_steps)))
            for f in self.cfg.save_at_fractions
        })
        if not marks:
            return total_steps
        gcd = marks[0]
        for m in marks[1:]:
            gcd = math.gcd(gcd, m)
        # If the GCD is "too small" (< 5% of total) fall back to total/10 so
        # we don't pathologically save every other step.
        min_acceptable = max(1, total_steps // 20)
        if gcd >= min_acceptable:
            return gcd
        return max(1, total_steps // 10)


__all__ = ["NativeSFT", "NativeSFTConfig"]

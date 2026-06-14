"""NativeSDFT — self-distillation fine-tuning on the native training loop.

Replaces :class:`~evsys_sdk.algorithms.tinker_sdft.TinkerSDFT`. Same composer
shape as :class:`~evsys_sdk.algorithms.native_sft.NativeSFT`: build the
backend, build the dataset, build the StepBuilder, hand them to a
:class:`~evsys_sdk.training.loop.TrainingLoop`, run it.

The interesting bit is the closure that hands the StepBuilder the latest
student sampler each step — that's the on-policy bit. Cookbook parity
without monkey-patching :func:`sdft.train_step` for loss logging
(``train/mean_loss`` is the StepBuilder's
:meth:`~evsys_sdk.training.step_builder.SDFTStepBuilder.step_metrics`
output now).
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any, ClassVar, cast

import tinker
from pydantic import BaseModel, ConfigDict, Field

from ..data_types import PromptExample, TargetFormat, parse_rows
from ..protocols import RunContext, RunResult
from ..registry import register_algorithm
from ..training.evaluators import build_in_loop_evaluators
from ..training.loop import TrainingLoop
from ..training.sdft_data import DEFAULT_DEMO_TEMPLATE
from ..training.step_builder import SDFTStepBuilder, SimpleSDFTDataset
from ..training.tinker_backend import TinkerBackend


class NativeSDFTConfig(BaseModel):
    """Config for :class:`NativeSDFT`. ``extra="forbid"`` so YAML typos fail."""

    model_config = ConfigDict(extra="forbid")

    # Training cadence
    learning_rate: float = 1e-4
    num_epochs: int = 1
    batch_size: int = 4
    max_steps: int | None = None

    # Model / LoRA
    lora_rank: int = 8
    renderer_name: str | None = None
    enable_thinking: bool | None = None

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

    # Checkpoint / eval cadence
    save_every: int = 0
    """If 0, computed from save_at_fractions."""
    save_at_fractions: list[float] = Field(default_factory=lambda: [1.0])
    eval_every: int = 0

    # Optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8

    wandb_project: str | None = None
    wandb_name: str | None = None


@register_algorithm("native_sdft")
class NativeSDFT:
    name: ClassVar[str] = "native_sdft"
    Config: ClassVar[type] = NativeSDFTConfig

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = NativeSDFTConfig.model_validate(kwargs)

    def train(self, ctx: RunContext) -> RunResult:
        if ctx.backend.name != "tinker":
            raise RuntimeError(
                f"NativeSDFT requires backend=tinker (got '{ctx.backend.name}')."
            )
        return asyncio.run(self._train_async(ctx))

    async def _train_async(self, ctx: RunContext) -> RunResult:
        rows = ctx.extras.get("train_rows")
        if not rows:
            raise RuntimeError("NativeSDFT.train: ctx.extras['train_rows'] missing/empty")
        # Standardize raw rows → typed PromptExample (strict): inputs['question']
        # is the prompt, expected is the gold answer. Build the dataset up front
        # so row-shape errors surface before we allocate a backend.
        examples = cast("list[PromptExample]", parse_rows(rows, TargetFormat.PROMPT_DATASET))
        dataset = SimpleSDFTDataset(rows=examples, batch_size=self.cfg.batch_size)

        handles = ctx.extras.get("backend_handles", {})
        model_name = handles.get("model_name") or ctx.extras.get("model_name")
        if not model_name:
            raise RuntimeError("model_name not set in backend handles")

        # 1. backend (LoRA training client over the student model)
        backend = await TinkerBackend.create(
            model_name=model_name,
            lora_rank=self.cfg.lora_rank,
            renderer_name=self.cfg.renderer_name or handles.get("renderer_name"),
            resume_state_path=handles.get("load_checkpoint_path"),
        )

        # 2. teacher sampling client (frozen; same base model).
        # TinkerBackend.snapshot_sampling_client returns a *student* sampler
        # bound to the current weights — we need a separate teacher over the
        # untrained base. Construct it via the underlying service client.
        teacher_client = backend._service.create_sampling_client(  # type: ignore[attr-defined]
            base_model=model_name,
        )
        from ..training.tinker_backend import TinkerSamplingClient
        teacher_wrapped = TinkerSamplingClient(teacher_client, name="teacher")

        # 3. compute total steps + save cadence
        steps_per_epoch = max(1, len(dataset))
        total_steps = (
            self.cfg.max_steps
            if self.cfg.max_steps is not None
            else steps_per_epoch * self.cfg.num_epochs
        )
        save_every = self._resolve_save_every(total_steps)

        # 4. log hyperparams
        ctx.log_store.log_hyperparams({
            "algorithm": self.name,
            **self.cfg.model_dump(),
            "model_name": model_name,
            "n_train_rows": len(rows),
            "total_steps": total_steps,
            "save_every": save_every,
        })

        # 5. closure providing the latest student sampler per step.
        # The cookbook does this via save_checkpoint_and_get_sampling_client
        # after each optim; same shape, exposed as an injectable seam.
        snapshot_counter = {"i": 0}

        async def _latest_student_sampler():
            snapshot_counter["i"] += 1
            return await backend.snapshot_sampling_client(
                name=f"student_snap_{snapshot_counter['i']}"
            )

        step_builder = SDFTStepBuilder(
            dataset=dataset,
            tokenizer=backend.get_tokenizer(),
            teacher_client=teacher_wrapped,
            student_sampler_provider=_latest_student_sampler,
            system_prompt=self.cfg.system_prompt,
            demo_template=self.cfg.demo_template,
            enable_thinking=self.cfg.enable_thinking,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            topk=self.cfg.topk,
            max_context_length=self.cfg.max_context_length,
            user_template=self.cfg.user_template,
            skip_first_n=self.cfg.skip_first_n_tokens,
        )

        evaluators = build_in_loop_evaluators(
            ctx.config.metadata if hasattr(ctx, "config") else None,
            tokenizer=backend.get_tokenizer(),
            store=getattr(ctx, "store", None) or ctx.extras.get("store"),
        )
        loop = TrainingLoop(
            backend=backend, step_builder=step_builder,
            log_store=ctx.log_store, output_dir=Path(ctx.output_dir),
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

        artifacts_dict = artifacts.as_dict()
        for key, value in artifacts_dict.items():
            ctx.log_store.log_artifact(key, value, kind="checkpoint")

        return RunResult(
            run_id=ctx.run_id,
            status="completed",
            metrics={},
            artifacts=artifacts_dict,
        )

    # --- helpers -----------------------------------------------------------

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
        min_acceptable = max(1, total_steps // 20)
        if gcd >= min_acceptable:
            return gcd
        return max(1, total_steps // 10)


__all__ = ["NativeSDFT", "NativeSDFTConfig"]

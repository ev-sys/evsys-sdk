"""LocalSFT — TRL SFTTrainer wrapper.

Pre-conditions:
  * ctx.backend.name == 'local'
  * ctx.extras['backend_handles']['model'] / ['tokenizer'] are set
  * ctx.extras['train_rows'] contains rows with 'messages' (chat format)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..protocols import RunContext, RunResult
from ..registry import register_algorithm

# Raise ImportError if TRL is missing
from trl import SFTConfig, SFTTrainer  # noqa: E402
from peft import LoraConfig  # noqa: E402
from datasets import Dataset  # noqa: E402

logger = logging.getLogger(__name__)


class LocalSFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_steps: int = 20
    max_seq_len: int = 512
    max_steps: int | None = None
    """If set, overrides num_epochs — training stops after this many steps."""
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 5
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@register_algorithm("local_sft")
class LocalSFT:
    name: ClassVar[str] = "local_sft"
    Config: ClassVar[type] = LocalSFTConfig

    def __init__(self, **kwargs) -> None:
        self.cfg = LocalSFTConfig.model_validate(kwargs)

    def train(self, ctx: RunContext) -> RunResult:
        if ctx.backend.name != "local":
            raise RuntimeError(f"LocalSFT requires backend=local (got '{ctx.backend.name}')")

        handles = ctx.extras.get("backend_handles", {})
        model = handles.get("model")
        tokenizer = handles.get("tokenizer")
        if model is None or tokenizer is None:
            raise RuntimeError("LocalSFT.train: backend_handles missing 'model' or 'tokenizer'")
        rows = ctx.extras.get("train_rows")
        if not rows:
            raise RuntimeError("LocalSFT.train: ctx.extras['train_rows'] missing/empty")

        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ctx.log_store.log_hyperparams({"algorithm": self.name, **self.cfg.model_dump()})

        train_ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
        lora_config = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.lora_target_modules,
            task_type="CAUSAL_LM",
        )

        sft_kwargs: dict = dict(
            output_dir=str(out),
            num_train_epochs=self.cfg.num_epochs,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            warmup_steps=self.cfg.warmup_steps,
            max_length=self.cfg.max_seq_len,
            logging_steps=self.cfg.logging_steps,
            save_strategy="steps",
            save_steps=self.cfg.save_steps,
            save_total_limit=self.cfg.save_total_limit,
            bf16=self.cfg.bf16,
            fp16=self.cfg.fp16,
            report_to="none",
            seed=self.cfg.seed,
        )
        if self.cfg.max_steps is not None:
            sft_kwargs["max_steps"] = self.cfg.max_steps
        args = SFTConfig(**sft_kwargs)

        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            peft_config=lora_config,
            processing_class=tokenizer,
        )

        try:
            trainer.train()
        except Exception as e:
            logger.exception("LocalSFT.train failed")
            return RunResult(run_id=ctx.run_id, status="failed", error=str(e))

        final = out / "final"
        trainer.save_model(str(final))
        tokenizer.save_pretrained(str(final))

        # Drain TRL log history into our log store.
        for entry in getattr(trainer.state, "log_history", []):
            step = int(entry.get("step", 0) or 0)
            metrics = {k: float(v) for k, v in entry.items() if isinstance(v, (int, float)) and k != "step"}
            if metrics:
                ctx.log_store.log_metrics(metrics, step=step)

        artifacts = {"final_checkpoint": str(final)}
        for ckpt in sorted(out.glob("checkpoint-*")):
            artifacts[ckpt.name] = str(ckpt)
        for k, v in artifacts.items():
            ctx.log_store.log_artifact(k, v, kind="checkpoint")

        loss_entries = [e for e in getattr(trainer.state, "log_history", []) if "loss" in e]
        final_loss = float(loss_entries[-1]["loss"]) if loss_entries else 0.0
        return RunResult(
            run_id=ctx.run_id,
            status="completed",
            metrics={"train/final_loss": final_loss},
            artifacts=artifacts,
        )

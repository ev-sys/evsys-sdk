"""LocalRL — TRL GRPOTrainer wrapper, with verifier-driven reward."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..protocols import RunContext, RunResult
from ..registry import get_verifier, register_algorithm

from trl import GRPOConfig, GRPOTrainer  # noqa: E402
from peft import LoraConfig  # noqa: E402
from datasets import Dataset  # noqa: E402

logger = logging.getLogger(__name__)


class LocalRLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = 5e-5
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_completion_length: int = 128
    num_generations: int = 4
    warmup_steps: int = 30
    logging_steps: int = 10
    save_steps: int = 50
    save_total_limit: int = 5
    bf16: bool = True
    beta: float = 0.04
    seed: int = 42
    lora_rank: int = 4
    lora_alpha: int = 8
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    verifier_kind: str = "composio_tool_match"
    verifier_params: dict[str, Any] = Field(default_factory=dict)


@register_algorithm("local_rl")
class LocalRL:
    name: ClassVar[str] = "local_rl"
    Config: ClassVar[type] = LocalRLConfig

    def __init__(self, **kwargs) -> None:
        self.cfg = LocalRLConfig.model_validate(kwargs)

    def train(self, ctx: RunContext) -> RunResult:
        if ctx.backend.name != "local":
            raise RuntimeError(f"LocalRL requires backend=local (got '{ctx.backend.name}')")
        handles = ctx.extras.get("backend_handles", {})
        model = handles.get("model")
        tokenizer = handles.get("tokenizer")
        if model is None or tokenizer is None:
            raise RuntimeError("LocalRL.train: backend_handles missing 'model' or 'tokenizer'")
        rows = ctx.extras.get("train_rows") or []
        if not rows:
            raise RuntimeError("LocalRL.train: ctx.extras['train_rows'] missing/empty")

        verifier_cls = get_verifier(self.cfg.verifier_kind)
        verifier = verifier_cls(**self.cfg.verifier_params)

        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ctx.log_store.log_hyperparams({"algorithm": self.name, **self.cfg.model_dump()})

        # Build dataset of prompts + extra columns kwargs the reward fn needs.
        ds_rows = []
        for r in rows:
            ds_rows.append({
                "prompt": r.get("prompt", ""),
                "tool_slug": r.get("tool_slug", ""),
                "toolkit": r.get("toolkit", ""),
            })
        train_ds = Dataset.from_list(ds_rows)

        def reward_fn(completions: list[str], **kwargs) -> list[float]:
            tool_slugs = kwargs.get("tool_slug", [])
            toolkits = kwargs.get("toolkit", [])
            out_rewards: list[float] = []
            for i, c in enumerate(completions):
                target = {
                    "tool_slug": tool_slugs[i] if i < len(tool_slugs) else "",
                    "toolkit": toolkits[i] if i < len(toolkits) else "",
                }
                out_rewards.append(verifier.verify(prompt="", completion=c, target=target).reward)
            return out_rewards

        args = GRPOConfig(
            output_dir=str(out),
            num_train_epochs=self.cfg.num_epochs,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            warmup_steps=self.cfg.warmup_steps,
            max_completion_length=self.cfg.max_completion_length,
            num_generations=self.cfg.num_generations,
            logging_steps=self.cfg.logging_steps,
            save_strategy="steps",
            save_steps=self.cfg.save_steps,
            save_total_limit=self.cfg.save_total_limit,
            bf16=self.cfg.bf16,
            beta=self.cfg.beta,
            seed=self.cfg.seed,
            log_completions=False,
            report_to="none",
        )

        lora_config = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=0.05,
            target_modules=self.cfg.lora_target_modules,
            task_type="CAUSAL_LM",
        )

        trainer = GRPOTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            reward_funcs=reward_fn,
            peft_config=lora_config,
            processing_class=tokenizer,
        )

        try:
            trainer.train()
        except Exception as e:
            logger.exception("LocalRL.train failed")
            return RunResult(run_id=ctx.run_id, status="failed", error=str(e))

        final = out / "final"
        trainer.save_model(str(final))
        tokenizer.save_pretrained(str(final))

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

        return RunResult(
            run_id=ctx.run_id,
            status="completed",
            metrics={},
            artifacts=artifacts,
        )

"""MockSFT — fake SFT for tests. Logs deterministic loss curve, saves stub ckpts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from ..protocols import RunContext, RunResult
from ..registry import register_algorithm


class MockSFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 1
    max_steps: int | None = None
    lora_rank: int = 8
    save_at_fractions: list[float] = [1.0]
    """Save checkpoints at these fractions of total steps (e.g. [0.5, 1.0])."""
    seed: int = 42


@register_algorithm("mock_sft")
class MockSFT:
    name: ClassVar[str] = "mock_sft"
    Config: ClassVar[type] = MockSFTConfig

    def __init__(self, **kwargs) -> None:
        self.cfg = MockSFTConfig.model_validate(kwargs)

    def train(self, ctx: RunContext) -> RunResult:
        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ctx.log_store.log_hyperparams({"algorithm": self.name, **self.cfg.model_dump()})

        # Pull dataset to set step count.
        # Try the data spec to estimate; otherwise default 100.
        n_rows = ctx.extras.get("n_train_rows", 100)
        steps_per_epoch = max(1, n_rows // self.cfg.batch_size)
        total_steps = (
            self.cfg.max_steps
            if self.cfg.max_steps is not None
            else steps_per_epoch * self.cfg.num_epochs
        )

        save_steps = sorted({max(1, round(f * total_steps)) for f in self.cfg.save_at_fractions})

        artifacts: dict[str, str] = {}
        for step in range(1, total_steps + 1):
            # Loss decreases like 2.0 * exp(-step/total*3) + small noise; deterministic.
            loss = 2.0 * math.exp(-3.0 * step / total_steps) + 0.05 * math.sin(step * 0.3)
            ctx.log_store.log_metrics({"train/loss": loss}, step=step)
            if step in save_steps:
                ckpt_dir = out / f"checkpoint-{step}"
                ckpt_dir.mkdir(exist_ok=True)
                (ckpt_dir / "metadata.json").write_text(
                    json.dumps({"step": step, "loss": loss, "algorithm": self.name})
                )
                key = f"ckpt_step_{step}"
                ctx.log_store.log_artifact(key, str(ckpt_dir), kind="checkpoint")
                artifacts[key] = str(ckpt_dir)

        final_dir = out / "final"
        final_dir.mkdir(exist_ok=True)
        (final_dir / "metadata.json").write_text(
            json.dumps({"final": True, "step": total_steps, "algorithm": self.name})
        )
        artifacts["final_checkpoint"] = str(final_dir)
        ctx.log_store.log_artifact("final_checkpoint", str(final_dir), kind="checkpoint")

        return RunResult(
            run_id=ctx.run_id,
            status="completed",
            metrics={"train/final_loss": float(loss), "total_steps": float(total_steps)},
            artifacts=artifacts,
        )

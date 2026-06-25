"""MockRL — fake GRPO-style RL. Deterministic reward curve."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..protocols import RunContext, RunResult
from ..registry import register_algorithm, get_verifier
from ..training.callbacks import dispatch, make_loop_state


class MockRLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = 5e-5
    num_steps: int = 100
    num_generations: int = 4
    kl_penalty_coef: float = 0.0
    verifier_kind: str = "format_only"
    verifier_params: dict[str, Any] = {}
    save_every: int = 50
    lora_rank: int = 4
    seed: int = 42


@register_algorithm("mock_rl")
class MockRL:
    name: ClassVar[str] = "mock_rl"
    Config: ClassVar[type] = MockRLConfig

    def __init__(self, **kwargs) -> None:
        self.cfg = MockRLConfig.model_validate(kwargs)

    def train(self, ctx: RunContext) -> RunResult:
        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Instantiate verifier just to prove the wiring works.
        v_cls = get_verifier(self.cfg.verifier_kind)
        verifier = v_cls(**self.cfg.verifier_params)

        cbs, state = make_loop_state(ctx, num_steps=self.cfg.num_steps)
        artifacts: dict[str, str] = {}
        reward = 0.1
        for step in range(1, self.cfg.num_steps + 1):
            # Deterministic upward curve toward ~0.9, with plateau.
            reward = 0.9 - 0.8 * math.exp(-step / max(1, self.cfg.num_steps / 4))
            state.step = step
            dispatch(cbs, "on_step_end", state, step, None,
                     {"train/reward": reward, "train/kl": 0.0})
            if step % max(1, self.cfg.save_every) == 0:
                ckpt = out / f"checkpoint-{step}"
                ckpt.mkdir(exist_ok=True)
                (ckpt / "metadata.json").write_text(
                    json.dumps({"step": step, "reward": reward})
                )
                artifacts[f"ckpt_step_{step}"] = str(ckpt)

        final_dir = out / "final"
        final_dir.mkdir(exist_ok=True)
        (final_dir / "metadata.json").write_text(
            json.dumps({"final": True, "step": self.cfg.num_steps, "reward": reward})
        )
        artifacts["final_checkpoint"] = str(final_dir)

        # Probe the verifier so it's actually exercised.
        probe = verifier.verify(prompt="x", completion="<think>t</think>\n<answer>X</answer>", target={})
        return RunResult(
            run_id=ctx.run_id,
            status="completed",
            metrics={
                "train/final_reward": float(reward),
                "verifier/probe_reward": float(probe.reward),
            },
            artifacts=artifacts,
        )

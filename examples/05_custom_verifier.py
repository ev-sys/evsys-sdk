"""05 — register a custom verifier (reward function) for RL.

Verifiers are the most common extension point: a new domain (browser actions,
SQL queries, math problems) usually means a new reward function. Here we
register one and use it inside MockRL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from evsys_sdk import (
    AlgorithmConfig,
    BackendConfig,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    RunConfig,
    register_verifier,
    run_experiment,
)
from evsys_sdk.protocols import VerificationResult

HERE = Path(__file__).parent


class LengthRewardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_length: int = 100
    sigma: float = 30.0


@register_verifier("length_reward")
class LengthReward:
    """Bell-curve reward peaking at `target_length` characters."""

    name: ClassVar[str] = "length_reward"
    Config: ClassVar[type] = LengthRewardConfig

    def __init__(self, *, target_length: int = 100, sigma: float = 30.0) -> None:
        self.target_length = target_length
        self.sigma = sigma

    def verify(self, *, prompt: str, completion: str, target: dict[str, Any]) -> VerificationResult:
        delta = abs(len(completion) - self.target_length)
        # Gaussian-shaped reward.
        import math
        r = math.exp(-(delta**2) / (2 * self.sigma**2))
        return VerificationResult(reward=r, info={"length": len(completion)})


def main():
    cfg = ExperimentConfig(
        name="example_05_custom_verifier",
        output_dir=str(HERE / "outputs" / "05"),
        run=RunConfig(
            name="rl_with_length",
            data=DataConfig(source_kind="in_memory", rows=[{"x": 1, "tool_slug": "X", "toolkit": "Y"}]),
            model=ModelConfig(name="tiny/fake"),
            algorithm=AlgorithmConfig(
                kind="mock_rl",
                params={
                    "num_steps": 30,
                    "save_every": 30,
                    "verifier_kind": "length_reward",
                    "verifier_params": {"target_length": 80, "sigma": 20.0},
                },
            ),
            backend=BackendConfig(kind="mock"),
            eval=EvalConfig(enabled=False),
        ),
    )
    [result] = run_experiment(cfg)
    print(f"Status: {result.status}, verifier probe reward: {result.metrics.get('verifier/probe_reward'):.3f}")


if __name__ == "__main__":
    main()

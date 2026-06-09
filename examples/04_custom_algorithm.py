"""04 — register a custom algorithm without forking the library.

Decorate a class with @register_algorithm("my_name"), declare a Pydantic
Config, and YAML files can use kind: my_name immediately.

In a third-party package, you'd also wire this up via entry points so
installation alone makes it available — see docs/cookbook.md.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from evsys_sdk import (
    AlgorithmConfig,
    BackendConfig,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    RunConfig,
    register_algorithm,
    run_experiment,
)
from evsys_sdk.protocols import RunContext, RunResult

HERE = Path(__file__).parent


class CosineToyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_steps: int = 50
    period: float = 10.0


@register_algorithm("cosine_toy")
class CosineToy:
    """Logs a cosine-shaped 'reward' curve. Pure demonstration."""

    name: ClassVar[str] = "cosine_toy"
    Config: ClassVar[type] = CosineToyConfig

    def __init__(self, **kwargs) -> None:
        self.cfg = CosineToyConfig.model_validate(kwargs)

    def train(self, ctx: RunContext) -> RunResult:
        for step in range(1, self.cfg.num_steps + 1):
            v = 0.5 + 0.5 * math.cos(step / self.cfg.period)
            ctx.log_store.log_metrics({"train/reward": v}, step=step)
        return RunResult(run_id=ctx.run_id, status="completed", metrics={"train/last_reward": v})


def main():
    cfg = ExperimentConfig(
        name="example_04_custom_algo",
        output_dir=str(HERE / "outputs" / "04"),
        run=RunConfig(
            name="cosine",
            data=DataConfig(source_kind="in_memory", rows=[{"x": 1}]),
            model=ModelConfig(name="tiny/fake"),
            algorithm=AlgorithmConfig(kind="cosine_toy", params={"num_steps": 30, "period": 5.0}),
            backend=BackendConfig(kind="mock"),
            eval=EvalConfig(enabled=False),
        ),
    )
    [result] = run_experiment(cfg)
    print(f"Status: {result.status}, last reward: {result.metrics['train/last_reward']:.3f}")


if __name__ == "__main__":
    main()

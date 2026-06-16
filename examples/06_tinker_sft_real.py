"""06 — real Tinker SFT (small).

Requires:
  pip install -e .[tinker]
  TINKER_API_KEY env var set

This will charge a small amount against your Tinker quota. ~$0.01.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from evsys_sdk import (
    AlgorithmConfig,
    BackendConfig,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    RunConfig,
    TransformSpec,
    run_experiment,
)

HERE = Path(__file__).parent


def main():
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY missing — see docs/cookbook.md")

    rows = [
        {"query": "save a contact from email", "tool_slug": "OUTLOOK_CREATE_CONTACT", "toolkit": "OUTLOOK", "description": "Create contact"},
        {"query": "edit slack message", "tool_slug": "SLACK_UPDATES_A_SLACK_MESSAGE", "toolkit": "SLACK", "description": "Update message"},
    ]

    cfg = ExperimentConfig(
        name="example_06_tinker_real",
        output_dir=str(HERE / "outputs" / "06"),
        run=RunConfig(
            name="sft_smoke",
            data=DataConfig(
                source_kind="in_memory",
                rows=rows,
                transforms=[TransformSpec(kind="jsonl_to_chat", params={"user_template": "Query: {query}", "assistant_template": "<answer>{tool_slug}</answer>"})],
            ),
            model=ModelConfig(name="Qwen/Qwen3.5-4B"),
            backend=BackendConfig(kind="tinker"),
            algorithm=AlgorithmConfig(
                kind="sft",
                params={
                    "learning_rate": 1.0e-4,
                    "max_steps": 2,
                    "batch_size": 1,
                    "lora_rank": 1,
                    "save_at_fractions": [1.0],
                },
            ),
            eval=EvalConfig(enabled=False),
        ),
    )

    [result] = run_experiment(cfg)
    print(f"Status: {result.status}")
    print(f"Final metrics: {result.metrics}")


if __name__ == "__main__":
    main()

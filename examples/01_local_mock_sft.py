"""01 — minimal local mock SFT.

Run:
    python examples/01_local_mock_sft.py

What this shows:
  * The mock backend + mock_sft algorithm produce a deterministic loss curve and
    saved checkpoints — useful for testing the wiring before any real training.
  * Local FS data store + JSONL log store — no network, no Tinker required.
"""

from __future__ import annotations

from pathlib import Path

from evsys_sdk import (
    AlgorithmConfig,
    BackendConfig,
    DataConfig,
    ExperimentConfig,
    LogStoreSpec,
    ModelConfig,
    RunConfig,
    TransformSpec,
    run_experiment,
)

HERE = Path(__file__).parent


def main():
    cfg = ExperimentConfig(
        name="example_01_mock_sft",
        output_dir=str(HERE / "outputs" / "01"),
        log_store=LogStoreSpec(kind="jsonl"),
        run=RunConfig(
            name="mock_sft_run",
            data=DataConfig(
                source_kind="in_memory",
                rows=[
                    {"query": "save a contact", "tool_slug": "OUTLOOK_CREATE_CONTACT", "toolkit": "OUTLOOK", "description": "..."},
                    {"query": "edit a slack message", "tool_slug": "SLACK_UPDATES_A_SLACK_MESSAGE", "toolkit": "SLACK", "description": "..."},
                ],
                transforms=[TransformSpec(kind="jsonl_to_chat", params={"user_template": "Query: {query}", "assistant_template": "<answer>{tool_slug}</answer>"})],
            ),
            model=ModelConfig(name="tiny/fake"),
            algorithm=AlgorithmConfig(
                kind="mock_sft",
                params={"num_epochs": 1, "batch_size": 1, "save_at_fractions": [0.5, 1.0]},
            ),
            backend=BackendConfig(kind="mock"),
        ),
    )

    [result] = run_experiment(cfg)
    print(f"Status: {result.status}")
    print(f"Artifacts: {list(result.artifacts.keys())}")
    print(f"Metrics: {result.metrics}")


if __name__ == "__main__":
    main()

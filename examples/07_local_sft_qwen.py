"""07 — real local SFT on Qwen/Qwen3-0.6B.

Trains a LoRA adapter on a tiny synthetic tool-query dataset using the
local backend (transformers + TRL + PEFT). No GPU required — runs on CPU
or MPS in a few minutes for 5 steps.

Run:
    pip install -e ".[local]"
    python examples/07_local_sft_qwen.py

Expected output:
    Status: completed
    Final loss: ~1.x
    Checkpoint: examples/outputs/07/local_sft_qwen/final/
"""

from __future__ import annotations

import time
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

# 10 synthetic tool-query rows covering different toolkits.
ROWS = [
    {"query": "save a contact from an email I received", "tool_slug": "OUTLOOK_CREATE_CONTACT", "toolkit": "OUTLOOK", "description": "Creates a new contact in Outlook."},
    {"query": "edit a slack message by timestamp", "tool_slug": "SLACK_UPDATES_A_SLACK_MESSAGE", "toolkit": "SLACK", "description": "Updates a Slack message identified by channel and ts."},
    {"query": "set the topic of a Slack conversation", "tool_slug": "SLACK_SET_THE_TOPIC_OF_A_CONVERSATION", "toolkit": "SLACK", "description": "Sets the topic for a Slack channel."},
    {"query": "create a new issue on GitHub", "tool_slug": "GITHUB_CREATE_AN_ISSUE", "toolkit": "GITHUB", "description": "Opens a new issue in a GitHub repository."},
    {"query": "send an email via Gmail", "tool_slug": "GMAIL_SEND_EMAIL", "toolkit": "GMAIL", "description": "Sends an email through Gmail."},
    {"query": "list my upcoming Google Calendar events", "tool_slug": "GOOGLE_CALENDAR_LIST_EVENTS", "toolkit": "GOOGLE_CALENDAR", "description": "Lists upcoming events from Google Calendar."},
    {"query": "create a new Notion page", "tool_slug": "NOTION_CREATE_PAGE", "toolkit": "NOTION", "description": "Creates a new page in Notion."},
    {"query": "post a message to a Slack channel", "tool_slug": "SLACK_POST_MESSAGE", "toolkit": "SLACK", "description": "Posts a message to a Slack channel."},
    {"query": "star a GitHub repository", "tool_slug": "GITHUB_STAR_A_REPO_FOR_THE_AUTHENTICATED_USER", "toolkit": "GITHUB", "description": "Stars a GitHub repo."},
    {"query": "reply to an Outlook email", "tool_slug": "OUTLOOK_REPLY_TO_EMAIL", "toolkit": "OUTLOOK", "description": "Sends a reply to an existing Outlook email."},
]


def main() -> None:
    cfg = ExperimentConfig(
        name="example_07_local_sft_qwen",
        output_dir=str(HERE / "outputs" / "07"),
        log_store=LogStoreSpec(kind="jsonl"),
        run=RunConfig(
            name="local_sft_qwen",
            data=DataConfig(
                source_kind="in_memory",
                rows=ROWS,
                transforms=[TransformSpec(kind="jsonl_to_chat", params={"user_template": "Query: {query}", "assistant_template": "<answer>{tool_slug}</answer>"})],
            ),
            model=ModelConfig(name="Qwen/Qwen3-0.6B"),
            backend=BackendConfig(
                kind="local",
                params={
                    "dtype": "float32",  # safe for CPU and MPS
                    "device": "cpu",     # change to "cuda" or "mps" if available
                },
            ),
            algorithm=AlgorithmConfig(
                kind="local_sft",
                params={
                    "num_epochs": 1,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "max_steps": 5,          # keep small for a smoke test
                    "max_seq_len": 256,
                    "lora_rank": 4,
                    "lora_alpha": 8,
                    "bf16": False,           # set True on CUDA for faster training
                    "fp16": False,
                    "logging_steps": 1,
                    "save_steps": 5,
                    "save_total_limit": 1,
                    "warmup_steps": 2,
                },
            ),
        ),
    )

    print(f"Model : Qwen/Qwen3-0.6B")
    print(f"Steps : 5  (change max_steps to train longer)")
    print(f"Output: {HERE / 'outputs' / '07'}")
    print("Starting training — first run will download the model (~1.2 GB)...\n")

    t0 = time.time()
    [result] = run_experiment(cfg)
    elapsed = time.time() - t0

    print(f"\nStatus  : {result.status}")
    if result.status == "completed":
        loss = result.metrics.get("train/final_loss", "n/a")
        ckpt = result.artifacts.get("final_checkpoint", "n/a")
        print(f"Loss    : {loss}")
        print(f"Checkpoint: {ckpt}")
        print(f"Logs    : {HERE / 'outputs' / '07' / 'local_sft_qwen' / 'logs' / 'metrics.jsonl'}")
    else:
        print(f"Error   : {result.error}")
    print(f"Time    : {elapsed:.1f}s")


if __name__ == "__main__":
    main()

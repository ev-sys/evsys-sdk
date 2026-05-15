"""Pytest fixtures."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure src is importable.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture()
def tmp_run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    return d


@pytest.fixture()
def composio_rows() -> list[dict]:
    return [
        {
            "query": "I want to save a contact from an email I received",
            "tool_slug": "OUTLOOK_CREATE_CONTACT",
            "toolkit": "OUTLOOK",
            "description": "Creates a new contact in a Microsoft Outlook user's contacts folder.",
        },
        {
            "query": "Edit Slack message by timestamp and channel ID",
            "tool_slug": "SLACK_UPDATES_A_SLACK_MESSAGE",
            "toolkit": "SLACK",
            "description": "Updates a Slack message identified by channel and ts.",
        },
        {
            "query": "Modify the channel topic to reflect our new project",
            "tool_slug": "SLACK_SET_THE_TOPIC_OF_A_CONVERSATION",
            "toolkit": "SLACK",
            "description": "Sets or updates the topic for a specified Slack conversation.",
        },
    ]


@pytest.fixture()
def composio_jsonl(tmp_path: Path, composio_rows: list[dict]) -> Path:
    p = tmp_path / "composio.jsonl"
    with p.open("w") as f:
        for r in composio_rows:
            f.write(json.dumps(r) + "\n")
    return p

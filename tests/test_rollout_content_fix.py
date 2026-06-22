"""Regression tests for the rollout-completion write bug.

`BasicLoopAgent.run` writes the model's completion to disk for the host-side
verifier. Harbor's `Chat` returns `content` as a plain str for simple replies
but as a **list of content blocks** when the model emits structured output (a
thinking model finishing its `<think>` block). `Path.write_text` raises
`TypeError: data must be str, not list` on a non-str, which killed the trial →
0 completions → SDFT (and latently RL) crashed. `_content_to_str` flattens it.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from evsys_sdk.training.harbor_agents import _content_to_str


@pytest.mark.parametrize(
    "content, expected",
    [
        ("plain string", "plain string"),
        (None, ""),
        ([{"type": "text", "text": "<answer>4</answer>"}], "<answer>4</answer>"),
        (
            [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "ab",
        ),
        (["x", "y"], "xy"),  # blocks that are already strings
    ],
)
def test_content_to_str(content, expected):
    assert _content_to_str(content) == expected


def test_write_text_no_longer_raises_on_list_content():
    # The exact failure mode: write_text on the coerced list-content must not
    # raise (it used to raise TypeError: data must be str, not list).
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "completion.txt"
        p.write_text(_content_to_str([{"type": "text", "text": "hi"}]))
        assert p.read_text() == "hi"

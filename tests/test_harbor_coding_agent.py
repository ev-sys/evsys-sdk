"""CodingLoopAgent — the multi-turn harness that executes the model's fenced
bash blocks inside the trial environment.

The content-shape tests are the important ones: a chat response's ``content``
is a plain string for some providers and a LIST of content blocks for others,
and the agent parsed it with a regex. Every rollout raised
``TypeError: expected string or bytes-like object, got 'list'`` on the first
turn, so no trajectory was ever harvested and RL reward was structurally zero —
which reads as "the model is bad", not "the harness is broken".
"""

from __future__ import annotations

import pytest

pytest.importorskip("harbor")
pytest.importorskip("tinker")

from evsys_sdk.training.harbor_coding_agent import (
    _BASH_BLOCK,
    _as_text,
)


class TestContentCoercion:
    def test_plain_string_passes_through(self):
        assert _as_text("ls /workspace") == "ls /workspace"

    def test_block_list_is_joined(self):
        assert _as_text([{"type": "text", "text": "a"},
                         {"type": "text", "text": "b"}]) == "a\nb"

    def test_blocks_without_text_are_dropped_not_stringified(self):
        """A tool_use block has no `text`; it must not leak `{'type': ...}` into
        the transcript the next turn is built from."""
        out = _as_text([{"type": "text", "text": "keep"},
                        {"type": "tool_use", "id": "t1"}])
        assert out == "keep"

    def test_string_list_is_joined(self):
        assert _as_text(["one", "two"]) == "one\ntwo"

    def test_empty_shapes_are_empty_strings(self):
        assert _as_text(None) == ""
        assert _as_text([]) == ""

    def test_unknown_shape_degrades_instead_of_raising(self):
        assert _as_text(42) == "42"


class TestBashBlockExtraction:
    def test_finds_a_block_in_block_style_content(self):
        """The regression: this is the exact path that used to raise."""
        blocks = _BASH_BLOCK.findall(_as_text(
            [{"type": "text", "text": "let me look\n```bash\nls /workspace\n```"}]))
        assert blocks == ["ls /workspace\n"]

    def test_finds_multiple_blocks_in_order(self):
        blocks = _BASH_BLOCK.findall(_as_text(
            "first\n```bash\ncd /workspace\n```\nthen\n```bash\nls -1\n```"))
        assert blocks == ["cd /workspace\n", "ls -1\n"]

    def test_no_blocks_is_empty_not_an_error(self):
        assert _BASH_BLOCK.findall(_as_text("just talking, no commands")) == []


class TestExecBlocks:
    """`_exec_blocks` returns None when there is nothing to run — the agent
    treats that as "done talking" and stops the turn loop."""

    def _agent(self):
        from evsys_sdk.training.harbor_coding_agent import CodingLoopAgent

        return CodingLoopAgent.__new__(CodingLoopAgent)

    @pytest.mark.asyncio
    async def test_returns_none_when_no_blocks(self):
        agent = self._agent()
        assert await agent._exec_blocks(object(), "no commands here") is None

    @pytest.mark.asyncio
    async def test_accepts_block_style_content_without_raising(self):
        """Before the fix this raised TypeError and killed the whole trial."""
        agent = self._agent()
        agent._exec_timeout_s = 5

        class _Env:
            async def exec(self, script, timeout_sec=None):
                return type("R", (), {"stdout": f"ran:{script.strip()}",
                                      "stderr": "", "return_code": 0})()

        out = await agent._exec_blocks(
            _Env(), [{"type": "text", "text": "```bash\necho hi\n```"}])
        assert out is not None and "ran:echo hi" in out

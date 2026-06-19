"""Test-only harbor agent: a MULTI-TURN, TOOL-USING agent.

Demonstrates the agent-plugin contract end to end — a researcher subclasses harbor's
``BaseAgent``, runs a real tool loop inside ``run()`` (call a tool -> observe ->
repeat), records a multi-turn rollout on the ``AgentContext``, and registers the
class with ``@register_agent``. No model/credentials: the "policy" is scripted so the
test is deterministic, but the harness (multi-turn loop + tool execution + per-turn
harvest) is exactly what a real agent would do.

Harbor loads it by import path ``tests.test_harbor_tool_agent:ToolLoopAgent`` at
runtime. It defines no test functions — it's a registered agent used by
``test_harbor_rollout_smoke.py`` — so the ``importorskip`` below also makes pytest
skip it cleanly in CI (where harbor isn't installed) instead of erroring on import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("harbor")  # subclasses harbor BaseAgent; skip collection without harbor

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import BaseModel, ConfigDict

from evsys_sdk.registry import register_agent


def lookup_tool(query: str) -> str:
    """A trivial, deterministic 'tool': returns a fact about the query. Stands in
    for any real tool (search, code-exec, an MCP server, environment.exec, ...)."""
    return f"<tool_result>len={len(query)}</tool_result>"


@register_agent("tool_loop")
class ToolLoopAgent(BaseAgent):
    """Multi-turn tool agent: each turn calls ``lookup_tool`` with the instruction,
    feeds the result back, and after ``max_turns`` turns emits a final answer that
    embeds the marker ``TOOLS_OK``. Records one rollout turn per loop iteration."""

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        max_turns: int = 2

    def __init__(
        self,
        *,
        max_turns: int = 2,
        # model/sampling kwargs injected by run_harbor_rollouts — ignored (no model here)
        model_name: str | None = None,
        model_path: str | None = None,
        renderer_name: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model_client: str | None = None,
        system_prompt: str | None = None,
        api_base: str | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)  # harbor's own kwargs: logs_dir, logger, mcp_servers, ...
        self._max_turns = max_turns

    @staticmethod
    def name() -> str:
        return "tool_loop"

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        prompt_turns: list[list[int]] = []
        completion_turns: list[list[int]] = []
        tool_calls = 0
        observation = instruction
        for turn in range(self._max_turns):
            # "policy" decides to call the tool (scripted/deterministic for the test)
            result = lookup_tool(observation)   # <-- the TOOL executes here
            tool_calls += 1
            observation = result
            # record this turn (prompt = what the model saw, completion = its action)
            prompt_turns.append(list(range(turn * 3, turn * 3 + 3)))
            completion_turns.append([100 + turn, 101 + turn])
        answer = f"ANSWER TOOLS_OK calls={tool_calls} last={observation}"

        context.n_input_tokens = sum(len(p) for p in prompt_turns)
        context.n_output_tokens = sum(len(c) for c in completion_turns)
        context.rollout_details = [{
            "prompt_token_ids": prompt_turns,
            "completion_token_ids": completion_turns,   # one entry per turn → multi-turn harvest
            "logprobs": [[-0.1] * len(c) for c in completion_turns],
        }]
        context.metadata = {"tool_calls": tool_calls}
        logs_dir = getattr(self, "logs_dir", None)
        if logs_dir is not None:
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            (Path(logs_dir) / "completion.txt").write_text(answer)

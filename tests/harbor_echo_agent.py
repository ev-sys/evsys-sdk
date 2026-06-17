"""Test-only harbor agent: a no-model agent that echoes the instruction.

Lives in tests/ (NOT shipped in the SDK) — used solely by
``test_harbor_rollout_smoke.py`` to drive a real harbor ``Job`` end to end
without tinker/a real model. Harbor loads it by the string import path
``tests.harbor_echo_agent:EchoAgent`` at trial runtime. It populates the
``AgentContext`` exactly like a real agent (rollout_details + completion +
token counts) so the harvest + Python scoring can be exercised.
"""

from __future__ import annotations

from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class EchoAgent(BaseAgent):
    def __init__(self, *, prefix: str = "ECHO:", **kw: Any) -> None:
        super().__init__(**kw)
        self._prefix = prefix

    @staticmethod
    def name() -> str:
        return "evsys-echo-test"

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
        text = f"{self._prefix}{instruction}"
        toks = list(range(1, len(text.split()) + 2))
        context.n_input_tokens = len(instruction.split())
        context.n_output_tokens = len(toks)
        context.rollout_details = [{
            "prompt_token_ids": [list(range(context.n_input_tokens))],
            "completion_token_ids": [toks],
            "logprobs": [[-0.1] * len(toks)],
        }]
        context.metadata = {**(context.metadata or {}), "completion": text}

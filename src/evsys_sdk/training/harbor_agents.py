"""Harbor glue classes — agent + environment (harbor 0.13.2).

These subclass harbor base classes, so ``harbor`` is imported at module top.
Our own code **never imports this module directly** — harbor loads these by the
string ``import_path`` recorded in the job/agent config at runtime (see
:mod:`evsys_sdk.training.harbor_engine`). That keeps the optional ``[harbor]``
extra out of the base import path.

* :class:`NoOpEnvironment` — sandbox-free ``BaseEnvironment`` (no container).
* :class:`BasicLoopAgent` — drives ``Chat(TinkerLLM(model_path))`` on-policy and
  records token-level ``rollout_details`` + the completion text onto the
  ``AgentContext`` (``context.metadata['completion']``). The default agent.
* :class:`EchoAgent` — a no-model agent that echoes the instruction; used to
  smoke-test the rollout wiring without tinker / a real model.

Scoring is **not** done by a harbor verifier here. Harbor 0.13.2's verifier runs
host-side against files synced from a container; our agents run in-process with
no container, so we disable the harbor verifier and score completions in Python
(:func:`evsys_sdk.training.harbor_engine.run_harbor_rollouts`).
"""

from __future__ import annotations

from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.llms.chat import Chat
from harbor.llms.tinker import TinkerLLM
from harbor.models.agent.context import AgentContext


class NoOpEnvironment(BaseEnvironment):
    """Sandbox-free environment: every operation is a no-op (no container).

    Agents using this run in-process; there's nothing to upload/download, so the
    log-sync steps harbor logs a warning for are harmless (the agent writes
    directly onto the ``AgentContext``, which is harvested regardless)."""

    @staticmethod
    def type() -> str:
        return "noop"

    def _validate_definition(self) -> None:
        return None

    async def start(self, force_build: bool = False) -> None:
        return None

    async def stop(self, delete: bool = True) -> None:
        return None

    async def upload_file(self, source_path, target_path) -> None:
        return None

    async def upload_dir(self, source_dir, target_dir) -> None:
        return None

    async def download_file(self, source_path, target_path) -> None:
        return None

    async def download_dir(self, source_dir, target_dir) -> None:
        return None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None) -> ExecResult:
        return ExecResult(stdout="", stderr="", return_code=0)


class BasicLoopAgent(BaseAgent):
    """Drive ``Chat(TinkerLLM(model_path))`` and record the rollout onto the
    ``AgentContext``: token-level ``rollout_details`` + the completion text in
    ``context.metadata['completion']`` (the host side scores it in Python)."""

    def __init__(
        self,
        *,
        model_name: str,
        model_path: str | None = None,
        renderer_name: str | None = None,
        max_tokens: int = 512,
        temperature: float = 1.0,
        max_turns: int = 1,
        system_prompt: str | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self._model_name = model_name
        self._model_path = model_path
        self._renderer_name = renderer_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_turns = max_turns
        self._system_prompt = system_prompt

    @staticmethod
    def name() -> str:
        return "evsys-basic-loop"

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
        llm = TinkerLLM(
            model_name=self._model_name,
            model_path=self._model_path,
            renderer_name=self._renderer_name,
            collect_rollout_details=True,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        chat = Chat(llm)
        if self._system_prompt:
            chat.messages.append({"role": "system", "content": self._system_prompt})
        resp = await chat.chat(instruction)
        context.rollout_details = chat.rollout_details
        context.metadata = {**(context.metadata or {}), "completion": resp.content or ""}


class EchoAgent(BaseAgent):
    """A no-model agent that echoes the instruction — for smoke-testing the
    rollout wiring (job → harvest → Python scoring) without tinker/a real model.
    Populates ``context`` exactly like a real agent does."""

    def __init__(self, *, prefix: str = "ECHO:", **kw: Any) -> None:
        super().__init__(**kw)
        self._prefix = prefix

    @staticmethod
    def name() -> str:
        return "evsys-echo"

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


__all__ = ["BasicLoopAgent", "EchoAgent", "NoOpEnvironment"]

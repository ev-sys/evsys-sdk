"""Harbor glue classes — agent / environment / verifier.

These subclass harbor base classes, so ``harbor`` is imported at module top.
Our own code **never imports this module directly** — harbor loads these by the
string ``import_path`` recorded in the trial/task config at runtime (see
:mod:`evsys_sdk.training.harbor_engine`). That keeps the optional ``[harbor]``
extra out of the base import path.

* :class:`NoOpEnvironment` — sandbox-free ``BaseEnvironment`` (all no-ops).
* :class:`BasicLoopAgent` — drives ``Chat(TinkerLLM(model_path))`` (on-policy)
  and records token-level ``rollout_details``. The default; users can plug any
  ``BaseAgent`` via ``agent_import_path``.
* :class:`EvsysVerifier` — scores the completion in Python via a registered
  verifier fn (no ``test.sh``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.llms.chat import Chat
from harbor.llms.tinker import TinkerLLM
from harbor.models.agent.context import AgentContext
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from .harbor_engine import _COMPLETION_FILE


class NoOpEnvironment(BaseEnvironment):
    """Sandbox-free environment: every operation is a no-op (no container)."""

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
    """Drive ``Chat(TinkerLLM(model_path))`` and record ``rollout_details``."""

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
        # Persist the completion so EvsysVerifier (host-side) can score it.
        agent_dir = getattr(self, "agent_dir", None)
        if agent_dir is not None:
            Path(agent_dir).mkdir(parents=True, exist_ok=True)
            (Path(agent_dir) / _COMPLETION_FILE).write_text(resp.content or "")
        context.rollout_details = chat.rollout_details


class EvsysVerifier(BaseVerifier):
    """Score the agent's completion in Python via a registered verifier fn.

    kwargs (from ``[verifier.kwargs]`` in task.toml): ``fn_name``, ``expected``,
    ``params``. Reads the completion the agent wrote to its agent dir — no
    ``test.sh``, no container exec.
    """

    def __init__(self, *, fn_name: str, expected: Any = None, params: dict | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        from ..verifiers import get_verifier_fn

        self._fn = get_verifier_fn(fn_name)
        self._expected = expected
        self._params = dict(params or {})

    async def verify(self) -> VerifierResult:
        completion = ""
        try:
            path = Path(self.trial_paths.agent_dir) / _COMPLETION_FILE
            if path.exists():
                completion = path.read_text()
        except Exception:  # pragma: no cover - defensive
            completion = ""
        reward = float(self._fn(completion, self._expected, self._params))
        return VerifierResult(rewards={"reward": reward})


__all__ = ["BasicLoopAgent", "NoOpEnvironment", "EvsysVerifier"]

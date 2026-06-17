"""Harbor glue classes — agent + environment (harbor 0.13.2).

These subclass harbor base classes, so ``harbor`` is imported at module top.
Our own code **never imports this module directly** — harbor loads these by the
string ``import_path`` recorded in the job/agent config at runtime (see
:mod:`evsys_sdk.training.harbor_engine`). That keeps the optional ``[harbor]``
extra out of the base import path.

* :class:`NoOpEnvironment` — sandbox-free ``BaseEnvironment`` (no container).
* :class:`BasicLoopAgent` — drives ``Chat(TinkerLLM(model_path))`` on-policy,
  records token-level ``rollout_details`` onto the ``AgentContext``, and writes
  the completion to the agent dir so the verifier can read it. The default agent.
* :class:`EvsysVerifier` — a harbor ``BaseVerifier`` that wraps our **registered
  verifier fns**. It runs host-side (no container exec): reads the completion the
  agent wrote + the per-task spec (``fn_name``/``expected``/``params``)
  materialized into the task dir, and returns the reward as a harbor
  ``VerifierResult``. Used in SHARED mode (a dummy ``tests/test.sh``, written by
  the materializer, satisfies harbor's task-load check and is never executed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.llms.chat import Chat
from harbor.llms.tinker import TinkerLLM
from harbor.models.agent.context import AgentContext
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from .harbor_engine import _COMPLETION_FILE, _VERIFIER_SPEC_FILE


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
    """Drive ``Chat(TinkerLLM(model_path))`` and record the rollout: token-level
    ``rollout_details`` + the completion text on the ``AgentContext``, and the
    completion written to the agent dir for the host-side verifier to score."""

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
        context.rollout_details = chat.rollout_details   # tokens/logprobs → harvested into the Trajectory
        # Write the completion to the agent dir so the host-side EvsysVerifier
        # (run by harbor) can read it (self.logs_dir == trial_paths.agent_dir).
        logs_dir = getattr(self, "logs_dir", None)
        if logs_dir is not None:
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            (Path(logs_dir) / _COMPLETION_FILE).write_text(resp.content or "")


class EvsysVerifier(BaseVerifier):
    """Score the agent's completion with a registered evsys verifier fn —
    host-side, no container exec. Reads the completion the agent wrote to its
    agent dir and the per-task spec (``fn_name``/``expected``/``params``)
    materialized into the task dir, and returns the reward as harbor's
    ``VerifierResult``. Set as the job-level verifier (SHARED mode)."""

    async def verify(self) -> VerifierResult:
        from ..verifiers import get_verifier_fn

        completion = _read_text(Path(self.trial_paths.agent_dir) / _COMPLETION_FILE)
        spec = _read_json(Path(self.task.paths.task_dir) / _VERIFIER_SPEC_FILE)
        reward = 0.0
        fn_name = spec.get("fn_name")
        if fn_name:
            try:
                fn = get_verifier_fn(fn_name)
                reward = float(fn(completion, spec.get("expected"), dict(spec.get("params") or {})))
            except Exception:  # pragma: no cover - a bad fn shouldn't crash the trial
                reward = 0.0
        return VerifierResult(rewards={"reward": reward})


def _read_text(path: Path) -> str:
    try:
        return path.read_text() if path.exists() else ""
    except Exception:  # pragma: no cover - defensive
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:  # pragma: no cover - defensive
        return {}


__all__ = ["BasicLoopAgent", "EvsysVerifier", "NoOpEnvironment"]

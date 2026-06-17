"""Harbor glue classes — agent + environment (harbor 0.13.2).

These subclass harbor base classes, so ``harbor`` is imported at module top.
Our own code **never imports this module directly** — harbor loads these by the
string ``import_path`` recorded in the job/agent config at runtime (see
:mod:`evsys_sdk.training.harbor_engine`). That keeps the optional ``[harbor]``
extra out of the base import path.

* :class:`NoOpEnvironment` — sandbox-free ``BaseEnvironment`` (no container).
* :class:`BasicLoopAgent` — drives ``Chat(<llm>)`` on-policy, records token-level
  ``rollout_details`` onto the ``AgentContext``, and writes the completion to the
  agent dir so the verifier can read it. The default agent. ``model_client`` picks
  the sampler: ``"tinker"`` (on-policy ``TinkerLLM``) or ``"litellm"`` (any provider
  litellm supports, for benchmarking closed/API models through the same path).
  Users can also plug any ``BaseAgent`` via ``agent_import_path``.
* :class:`EvsysVerifier` — a harbor ``BaseVerifier`` that wraps our **registered
  verifier fns**. It runs host-side (no container exec): reads the completion the
  agent wrote + the per-task spec (``fn_name``/``expected``/``params``)
  materialized into the task dir, and returns the reward as a harbor
  ``VerifierResult``. Used in SHARED mode (a dummy ``tests/test.sh``, written by
  the materializer, satisfies harbor's task-load check and is never executed).
"""

from __future__ import annotations

import asyncio
import json
import logging
import weakref
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared LLM cache
#
# harbor builds a fresh agent per trial (AgentFactory in trial.py), so an LLM
# built in run()/__init__ is re-created for every task — and for TinkerLLM that
# means a new sampling client (a server round-trip) per task. The agent is the
# wrong scope to share from (it holds per-trial state like logs_dir); the LLM
# client holds none, so we cache it ABOVE the agent, at module scope.
#
# Keyed by the running event loop (weakref → auto-evicted when the loop is
# GC'd): shared across all trials of one harbor job (one ``asyncio.run``), never
# reused across loops since a tinker client holds loop-bound httpx sessions.
# First creation is warmed under a per-loop lock so concurrent trials can't race
# into multiple sampling clients.
# ---------------------------------------------------------------------------

_LLM_CACHE: "weakref.WeakKeyDictionary[Any, dict[tuple, Any]]" = weakref.WeakKeyDictionary()
_LLM_LOCKS: "weakref.WeakKeyDictionary[Any, asyncio.Lock]" = weakref.WeakKeyDictionary()


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
        model_client: str = "tinker",
        api_base: str | None = None,
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
        self._model_client = model_client
        self._api_base = api_base

    @staticmethod
    def name() -> str:
        return "evsys-basic-loop"

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    def _build_llm(self) -> Any:
        """The harbor sampler for this rollout. ``model_client`` picks it:
        ``"tinker"`` → on-policy ``TinkerLLM`` (needs ``model_path``);
        ``"litellm"`` → harbor's litellm LLM for any provider (``model_name`` a
        litellm string, e.g. ``"anthropic/claude-opus-4-1"``; keys from the
        provider env vars). Both collect rollout details so usage/cost is
        captured — and for API models the cost is real."""
        if self._model_client == "litellm":
            from harbor.llms.lite_llm import LiteLLM  # lazy: tinker rollouts skip litellm

            return LiteLLM(
                model_name=self._model_name,
                temperature=self._temperature,
                api_base=self._api_base,
                collect_rollout_details=True,
            )
        return TinkerLLM(
            model_name=self._model_name,
            model_path=self._model_path,
            renderer_name=self._renderer_name,
            collect_rollout_details=True,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

    def _cache_key(self) -> tuple:
        return (
            self._model_client,
            self._model_name,
            self._model_path,
            self._renderer_name,
            self._max_tokens,
            self._temperature,
            self._api_base,
        )

    async def _shared_llm(self) -> Any:
        """The cached LLM for this rollout's (loop, config) — built once per
        harbor job and reused by every trial (see the module cache note). The
        first build is warmed under a per-loop lock so concurrent trials share
        ONE sampling client instead of each creating their own."""
        loop = asyncio.get_running_loop()
        per_loop = _LLM_CACHE.setdefault(loop, {})
        lock = _LLM_LOCKS.setdefault(loop, asyncio.Lock())
        key = self._cache_key()
        async with lock:
            llm = per_loop.get(key)
            if llm is None:
                logger.info("[harbor] LLM cache MISS — building %s client for %s (1 per job)",
                            self._model_client, self._model_name)
                llm = self._build_llm()
                ensure = getattr(llm, "_ensure_client", None)
                if ensure is not None:
                    await ensure()  # create the sampling client once, under lock
                per_loop[key] = llm
            else:
                logger.debug("[harbor] LLM cache HIT — reusing shared %s client", self._model_client)
            return llm

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        chat = Chat(await self._shared_llm())
        if self._system_prompt:
            chat.messages.append({"role": "system", "content": self._system_prompt})
        logger.info("[harbor] agent.run instruction=%r", _trunc(instruction))
        resp = await chat.chat(instruction)
        context.rollout_details = chat.rollout_details   # tokens/logprobs → harvested into the Trajectory
        details = chat.rollout_details or []              # list of per-turn dicts (harvest reads [0])
        rd = details[0] if details else {}
        logger.info(
            "[harbor] agent.run completion=%r (prompt_tokens=%s completion_tokens=%s)",
            _trunc(resp.content or ""),
            len(rd.get("prompt_token_ids") or []) or None,
            len(rd.get("completion_token_ids") or []) or None,
        )
        # Write the completion to the agent dir so the host-side EvsysVerifier
        # (run by harbor) can read it (self.logs_dir == trial_paths.agent_dir).
        logs_dir = getattr(self, "logs_dir", None)
        if logs_dir is not None:
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            (Path(logs_dir) / _COMPLETION_FILE).write_text(resp.content or "")
            logger.debug("[harbor] wrote completion → %s", Path(logs_dir) / _COMPLETION_FILE)


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
                logger.exception("[harbor] verifier fn %r raised — reward=0", fn_name)
                reward = 0.0
        logger.info(
            "[harbor] verify fn=%s expected=%r completion=%r → reward=%.3f",
            fn_name, _trunc(str(spec.get("expected")), 60), _trunc(completion, 80), reward,
        )
        return VerifierResult(rewards={"reward": reward})


def _trunc(s: str, n: int = 200) -> str:
    """Single-line, length-capped string for debug logs."""
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n} chars)"


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

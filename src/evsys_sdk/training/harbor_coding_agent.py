"""Multi-turn coding agent + sandbox verifier for harbor rollouts.

``BasicLoopAgent`` is chat-only — harbor's ``Chat`` has no tools parameter and
never touches the environment. ``CodingLoopAgent`` closes that gap for coding
tasks: each turn it samples the model, executes any fenced ```bash blocks
inside the trial's environment (``environment.exec`` — a Modal sandbox when
the job's environment is ``type: modal``), and feeds the combined output back
as the next user message, until the model emits ``<final>`` or ``max_turns``
runs out. ``Chat`` accumulates ``rollout_details`` across turns, so on-policy
token-level trajectories come through the normal harvest for training.

``SandboxTestVerifier`` scores a trial by running the task's
``sandbox_verifier.json`` ``test_command`` inside that same environment
(exit 0 → reward 1.0) — the objective "did the tests pass" signal for
distilled coding benchmarks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from ..logger import get_logger
from .harbor_agents import BasicLoopAgent

log = get_logger(__name__)

_BASH_BLOCK = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)


def _as_text(content: Any) -> str:
    """Message content as a string.

    A chat response's ``content`` is a plain string for some providers and a
    LIST of content blocks for others. Every rollout of this agent crashed with
    ``TypeError: expected string or bytes-like object, got 'list'`` the moment a
    block-style response arrived, so no trajectory was ever harvested and reward
    was structurally 0."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                parts.append(str(b.get("text") or b.get("content") or ""))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)
_FINAL_MARKER = "<final>"
_COMPLETION_FILE = "completion.txt"
_VERIFIER_SPEC_FILE = "sandbox_verifier.json"
_OUTPUT_CAP = 8_000  # chars of tool output fed back per turn

_CODING_SYSTEM_PROMPT = """You are a coding agent working inside a sandboxed checkout of a \
repository at /workspace. To run commands, emit one or more fenced blocks:

```bash
<commands>
```

After each of your messages, every bash block is executed in order and you get the combined \
output back. Small file edits: use heredocs (cat > path <<'EOF' ... EOF). When the task is \
done, write {final} on its own line followed by a short summary of what you changed and why. \
Work incrementally; run the relevant tests before declaring {final}."""


class CodingLoopAgent(BasicLoopAgent):
    """Sample → execute ```bash blocks in the environment → feed output back.

    Inherits the LLM plumbing (tinker/litellm client cache) from
    :class:`BasicLoopAgent`; only the turn loop differs.
    """

    def __init__(self, *, final_marker: str = _FINAL_MARKER, exec_timeout_s: int = 300,
                 **kw: Any) -> None:
        super().__init__(**kw)
        self._final_marker = final_marker
        self._exec_timeout_s = exec_timeout_s

    @staticmethod
    def name() -> str:
        return "evsys-coding-loop"

    async def _exec_blocks(self, environment: BaseEnvironment, text: Any) -> str | None:
        """Run each fenced bash block; return combined output (None = no blocks)."""
        blocks = _BASH_BLOCK.findall(_as_text(text))
        if not blocks:
            return None
        chunks: list[str] = []
        for script in blocks:
            try:
                result = await environment.exec(script, timeout_sec=self._exec_timeout_s)
                out = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
                chunks.append(f"$ (block)\nexit={result.return_code}\n{out.strip()}")
            except Exception as e:  # env hiccup is feedback, not a crash
                chunks.append(f"$ (block)\nexecution error: {e}")
        combined = "\n\n".join(chunks)
        return combined[:_OUTPUT_CAP] + ("\n[output truncated]" if len(combined) > _OUTPUT_CAP else "")

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        chat = Chat(await self._shared_llm())
        system = self._system_prompt or _CODING_SYSTEM_PROMPT.format(final=self._final_marker)
        chat.messages.append({"role": "system", "content": system})

        message = instruction
        final_text = ""
        for _ in range(max(1, self._max_turns)):
            resp = await chat.chat(message)
            final_text = _as_text(resp.content)
            if self._final_marker in final_text:
                break
            tool_output = await self._exec_blocks(environment, final_text)
            if tool_output is None:
                break  # no commands and no final marker — the agent is done talking
            message = tool_output

        context.rollout_details = chat.rollout_details
        context.n_input_tokens = chat.total_input_tokens
        context.n_output_tokens = chat.total_output_tokens
        context.n_cache_tokens = chat.total_cache_tokens
        context.cost_usd = chat.total_cost

        diff = await self._exec_blocks(environment, "```bash\ngit -C /workspace diff\n```") or ""
        logs_dir = getattr(self, "logs_dir", None)
        if logs_dir is not None:
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            (Path(logs_dir) / _COMPLETION_FILE).write_text(
                final_text + ("\n\n--- workspace diff ---\n" + diff if diff.strip() else "")
            )


class SandboxTestVerifier(BaseVerifier):
    """Reward = the task's test command exiting 0 inside the trial environment.

    Reads ``sandbox_verifier.json`` (``{"test_command": ...}``) from the task
    dir; no spec or no command → reward 0 with a log, never a crash.
    """

    async def verify(self) -> VerifierResult:
        spec_path = Path(self.task.paths.task_dir) / _VERIFIER_SPEC_FILE
        try:
            spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
        except json.JSONDecodeError:
            spec = {}
        cmd = spec.get("test_command")
        if not cmd:
            log.warning("[sandbox-verifier] no test_command in %s", spec_path)
            return VerifierResult(rewards={"reward": 0.0})
        try:
            result = await self.environment.exec(cmd, timeout_sec=int(spec.get("timeout_s", 600)))
            return VerifierResult(rewards={"reward": 1.0 if result.return_code == 0 else 0.0})
        except Exception as e:
            log.warning("[sandbox-verifier] exec failed: %s", e)
            return VerifierResult(rewards={"reward": 0.0})


__all__ = ["CodingLoopAgent", "SandboxTestVerifier"]

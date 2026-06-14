"""Generic on-policy rollout helper — multi-turn by default.

One function, :func:`generate_rollouts`, drives a policy through a batch of
:class:`RolloutTask`\\s and returns token-level :class:`Trajectory`\\s (per-turn
tokens + logprobs) plus a scalar reward. Single-turn is the degenerate case
(``env`` ends the episode after one turn, or ``env is None`` for pure
generation, e.g. SDFT's student rollout).

The policy is harbor's ``TinkerLLM`` pointed at a tinker ``model_path``
(a saved-weights checkpoint URI) — so re-pointing it at the latest checkpoint
each step keeps rollouts on-policy. harbor is imported lazily so the base SDK
and the SFT/mock paths never pull it in.

No sandbox: an :data:`RolloutEnv` is a plain in-process async callable
``(messages) -> EnvStep``. A verifier (single-turn) or a tool loop
(multi-turn) lives inside it; swapping in a real sandbox later only changes the
env body, not this helper's signature.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

# ---------------------------------------------------------------------------
# Trajectory data types (formerly in training/env.py)
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One assistant turn of a rollout.

    ``prompt_tokens`` is the full rendered context the policy saw for this turn
    (system + prior turns + the latest observation); ``completion_tokens`` /
    ``logprobs`` are the sampled response. For a single-turn rollout there is
    exactly one ``Turn``.
    """

    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    text: str = ""


@dataclass
class Trajectory:
    """One rollout: an ordered list of :class:`Turn`\\s + a scalar reward."""

    turns: list[Turn]
    reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryGroup:
    """All rollouts sampled from one :class:`RolloutTask` (``num_samples`` of
    them). Group-relative advantage baselines subtract the within-group mean."""

    trajectories: list[Trajectory]
    tags: list[str] = field(default_factory=list)

    @property
    def rewards(self) -> list[float]:
        return [t.reward for t in self.trajectories]


# ---------------------------------------------------------------------------
# Environment contract — in-process, no sandbox
# ---------------------------------------------------------------------------


@dataclass
class EnvStep:
    """What an :data:`RolloutEnv` returns after an assistant turn."""

    observation: str | None = None
    """Next user message fed back to the policy. ``None`` ends the episode."""
    reward: float = 0.0
    """Reward so far (the last non-default value is the trajectory reward)."""
    done: bool = False
    """Terminate the episode after this turn."""


RolloutEnv = Callable[[list[dict]], Awaitable[EnvStep]]
"""``async (messages) -> EnvStep``. Receives the running chat messages (each
``{"role", "content"}``) and returns the next observation + reward + done.
Single-turn verifier: return ``EnvStep(done=True, reward=...)`` on turn 0.
Multi-turn tool loop: parse the last assistant message, run an in-process tool,
return its output as ``observation``. All pure Python — no container."""


@dataclass
class RolloutTask:
    """One unit of rollout work: a prompt + its (optional) per-task env.

    Algorithms map their typed rows onto this: ``native_rl`` turns a
    ``HarborTask`` into ``RolloutTask(prompt=instruction, env=verifier_env)``;
    ``native_sdft`` turns a ``PromptExample`` into
    ``RolloutTask(prompt=question, env=None)`` (pure generation).
    """

    prompt: str
    env: RolloutEnv | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


def _import_tinker_llm():
    """Lazy import of harbor's TinkerLLM (optional ``[harbor]`` extra)."""
    try:
        from harbor.llms.tinker import TinkerLLM
    except ImportError as e:  # pragma: no cover - exercised via the error path
        raise RuntimeError(
            "generate_rollouts requires the 'harbor' package. "
            "Install it with: pip install 'evsys-sdk[harbor]'"
        ) from e
    return TinkerLLM


async def generate_rollouts(
    tasks: Sequence[RolloutTask],
    *,
    model_name: str,
    model_path: str | None,
    renderer_name: str | None = None,
    num_samples: int = 1,
    max_turns: int = 8,
    max_tokens: int = 512,
    temperature: float = 1.0,
    system_prompt: str | None = None,
    _llm: Any | None = None,
) -> list[list[Trajectory]]:
    """Roll out every task ``num_samples`` times against ``model_path``.

    Returns ``trajectories[task][sample]``. The policy is a single harbor
    ``TinkerLLM`` (built from ``model_path`` → on-policy) reused across all
    rollouts. Each rollout runs the multi-turn loop:

      1. policy emits an assistant turn (tokens + logprobs recorded),
      2. ``task.env(messages)`` returns the next observation / reward / done,
      3. repeat until ``done`` / ``observation is None`` / ``max_turns`` —
         or, when ``task.env is None``, stop after one turn (pure generation).

    ``_llm`` is an injection seam for tests (a stand-in with an async
    ``call(prompt, message_history)`` returning ``prompt_token_ids`` /
    ``completion_token_ids`` / ``logprobs`` / ``content``).
    """
    llm = _llm if _llm is not None else _build_llm(
        model_name=model_name, model_path=model_path, renderer_name=renderer_name,
        max_tokens=max_tokens, temperature=temperature,
    )

    async def _one(task: RolloutTask) -> Trajectory:
        return await _rollout_once(
            llm, task, max_turns=max_turns, system_prompt=system_prompt,
        )

    # Flatten (task, sample) → run concurrently → regroup.
    coros = [_one(t) for t in tasks for _ in range(num_samples)]
    flat = await asyncio.gather(*coros)
    return [
        list(flat[i * num_samples:(i + 1) * num_samples])
        for i in range(len(tasks))
    ]


def _build_llm(*, model_name, model_path, renderer_name, max_tokens, temperature):
    TinkerLLM = _import_tinker_llm()
    return TinkerLLM(
        model_name=model_name,
        model_path=model_path,
        renderer_name=renderer_name,
        collect_rollout_details=True,
        max_tokens=max_tokens,
        temperature=temperature,
    )


async def _rollout_once(
    llm: Any,
    task: RolloutTask,
    *,
    max_turns: int,
    system_prompt: str | None,
) -> Trajectory:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    turns: list[Turn] = []
    reward = 0.0
    observation = task.prompt

    for _ in range(max(1, max_turns)):
        resp = await llm.call(prompt=observation, message_history=messages)
        turns.append(Turn(
            prompt_tokens=list(getattr(resp, "prompt_token_ids", None) or []),
            completion_tokens=list(getattr(resp, "completion_token_ids", None) or []),
            logprobs=[float(x) for x in (getattr(resp, "logprobs", None) or [])],
            text=getattr(resp, "content", "") or "",
        ))
        messages.append({"role": "user", "content": observation})
        messages.append({"role": "assistant", "content": turns[-1].text})

        if task.env is None:
            break  # pure generation (single turn)
        step = await task.env(messages)
        reward = step.reward
        if step.done or step.observation is None:
            break
        observation = step.observation

    return Trajectory(turns=turns, reward=float(reward), metadata=dict(task.metadata))


__all__ = [
    "Turn",
    "Trajectory",
    "TrajectoryGroup",
    "EnvStep",
    "RolloutEnv",
    "RolloutTask",
    "generate_rollouts",
]

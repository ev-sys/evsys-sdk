"""NativeRL — on-policy RL on the native training loop.

Single-turn out of the box (the path most evsys projects start with —
prompt → completion → verifier reward); multi-turn slots in via the
:class:`~evsys_sdk.training.env.EnvGroupBuilder` Protocol when a project
needs it.

Researchers supply:

* ``train_rows`` — list of ``{"prompt": str, "expected": ..., "tags": [...]}``
  rows, or pre-built :class:`EnvGroupBuilder` instances via
  ``ctx.extras["env_builders"]`` for non-text envs.
* a verifier name registered under :mod:`evsys_sdk.verifiers` — or a callable
  passed at construction time for unusual cases.

The composer wires the rest: a SimpleRLDataset over the rows, an
:class:`~evsys_sdk.training.step_builder.RLStepBuilder` over the dataset, a
:class:`~evsys_sdk.training.loop.TrainingLoop` over the step builder.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any, Callable, ClassVar, cast

import tinker
from pydantic import BaseModel, ConfigDict, Field

from ..data_types import HarborTask, InProcessVerifier, TargetFormat, parse_rows
from ..protocols import RunContext, RunResult
from ..registry import register_algorithm
from ..training.evaluators import build_in_loop_evaluators
from ..training.loop import TrainingLoop
from ..training.rollout import EnvStep, RolloutTask
from ..training.step_builder import RLStepBuilder
from ..training.tinker_backend import TinkerBackend


class NativeRLConfig(BaseModel):
    """Config for :class:`NativeRL`."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 1.0e-5
    """RL needs a lower LR than SFT — IS gradients can be large."""
    num_epochs: int = 1
    batch_size: int = 4
    num_samples: int = 1
    """Rollouts per builder — `group_size` in the cookbook."""
    max_steps: int | None = None

    lora_rank: int = 8
    renderer_name: str | None = None
    enable_thinking: bool | None = None

    verifier_name: str | None = None
    """Fallback verifier-fn name used only when a HarborTask's InProcessVerifier
    leaves ``fn_name`` empty. Normally the verifier is carried per-row by the
    HarborTask itself (looked up via :func:`evsys_sdk.verifiers.get_verifier_fn`)."""

    max_tokens: int = 256
    temperature: float = 1.0
    max_turns: int = 8
    """Max agent turns per rollout. With a single-turn verifier env this is
    effectively 1; >1 enables multi-turn envs (the env decides when to stop)."""
    drop_constant_reward: bool = True

    system_prompt: str | None = None
    user_template: str = "{prompt}"

    save_every: int = 0
    save_at_fractions: list[float] = Field(default_factory=lambda: [1.0])
    eval_every: int = 0

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8


@register_algorithm("native_rl")
class NativeRL:
    name: ClassVar[str] = "native_rl"
    Config: ClassVar[type] = NativeRLConfig

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = NativeRLConfig.model_validate(kwargs)

    def train(self, ctx: RunContext) -> RunResult:
        if ctx.backend.name != "tinker":
            raise RuntimeError(
                f"NativeRL requires backend=tinker (got '{ctx.backend.name}')."
            )
        return asyncio.run(self._train_async(ctx))

    async def _train_async(self, ctx: RunContext) -> RunResult:
        handles = ctx.extras.get("backend_handles", {})
        model_name = handles.get("model_name") or ctx.extras.get("model_name")
        if not model_name:
            raise RuntimeError("model_name not set in backend handles")

        backend = await TinkerBackend.create(
            model_name=model_name,
            lora_rank=self.cfg.lora_rank,
            renderer_name=self.cfg.renderer_name or handles.get("renderer_name"),
            resume_state_path=handles.get("load_checkpoint_path"),
        )

        rows = ctx.extras.get("train_rows")
        if not rows:
            raise RuntimeError(
                "NativeRL.train: ctx.extras['train_rows'] missing/empty "
                "(HarborTask rows: task_id + instruction + verifier)."
            )
        tasks = self._tasks_from_rows(rows)

        steps_per_epoch = max(1, len(tasks) // self.cfg.batch_size)
        total_steps = (
            self.cfg.max_steps
            if self.cfg.max_steps is not None
            else steps_per_epoch * self.cfg.num_epochs
        )
        save_every = self._resolve_save_every(total_steps)

        ctx.log_store.log_hyperparams({
            "algorithm": self.name,
            **self.cfg.model_dump(),
            "model_name": model_name,
            "n_tasks": len(tasks),
            "total_steps": total_steps,
            "save_every": save_every,
        })

        # Per-step checkpoint provider: snapshot the current weights and hand
        # the rollout helper its tinker:// path (keeps rollouts on-policy).
        snapshot_counter = {"i": 0}

        async def _checkpoint() -> str:
            snapshot_counter["i"] += 1
            return await backend.save_for_sampler(f"rl_snap_{snapshot_counter['i']}")

        step_builder = RLStepBuilder(
            tasks=tasks,
            checkpoint_provider=_checkpoint,
            model_name=model_name,
            batch_size=self.cfg.batch_size,
            renderer_name=self.cfg.renderer_name or handles.get("renderer_name"),
            num_samples=self.cfg.num_samples,
            max_turns=self.cfg.max_turns,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system_prompt=self.cfg.system_prompt,
            drop_constant_reward=self.cfg.drop_constant_reward,
        )

        evaluators = build_in_loop_evaluators(
            ctx.config.metadata if hasattr(ctx, "config") else None,
            tokenizer=backend.get_tokenizer(),
            store=getattr(ctx, "store", None) or ctx.extras.get("store"),
        )
        loop = TrainingLoop(
            backend=backend, step_builder=step_builder,
            log_store=ctx.log_store, output_dir=Path(ctx.output_dir),
            adam_params=tinker.AdamParams(
                learning_rate=self.cfg.learning_rate,
                beta1=self.cfg.adam_beta1,
                beta2=self.cfg.adam_beta2,
                eps=self.cfg.adam_eps,
            ),
            save_every=save_every, eval_every=self.cfg.eval_every,
            evaluators=evaluators,
        )
        artifacts = await loop.run(num_steps=total_steps)

        d = artifacts.as_dict()
        for k, v in d.items():
            ctx.log_store.log_artifact(k, v, kind="checkpoint")

        return RunResult(
            run_id=ctx.run_id, status="completed", metrics={}, artifacts=d,
        )

    def _tasks_from_rows(self, rows: list[dict[str, Any]]) -> list[RolloutTask]:
        from ..verifiers import get_verifier_fn

        # Standardize raw rows → typed HarborTask (strict): instruction is the
        # prompt, the in-process verifier spec rides on each task.
        harbor_tasks = cast("list[HarborTask]", parse_rows(rows, TargetFormat.HARBOR_TASK))

        out: list[RolloutTask] = []
        for t in harbor_tasks:
            v = t.verifier
            if not isinstance(v, InProcessVerifier):
                raise RuntimeError(
                    f"NativeRL: task {t.task_id!r} uses a {v.kind!r} verifier; "
                    "the rollout path executes only 'in_process' verifiers today "
                    "(e2b / llm_judge are not yet wired)."
                )
            fn_name = v.fn_name or self.cfg.verifier_name
            if not fn_name:
                raise RuntimeError(
                    f"NativeRL: task {t.task_id!r} has no verifier fn_name and "
                    "cfg.verifier_name is unset."
                )
            try:
                verifier_fn = get_verifier_fn(fn_name)
            except ValueError as e:
                raise RuntimeError(f"NativeRL: unknown verifier_name {fn_name!r}") from e

            metadata = {k: val for k, val in t.metadata.items()}
            metadata.setdefault("task_id", t.task_id)
            out.append(RolloutTask(
                prompt=self.cfg.user_template.format(prompt=t.instruction),
                env=_verifier_env(verifier_fn, expected=v.expected, params=v.params),
                metadata=metadata,
            ))
        if not out:
            raise RuntimeError("NativeRL: no usable HarborTask rows")
        return out

    def _resolve_save_every(self, total_steps: int) -> int:
        if self.cfg.save_every:
            return self.cfg.save_every
        marks = sorted({
            max(1, int(round(f * total_steps)))
            for f in self.cfg.save_at_fractions
        })
        if not marks:
            return total_steps
        gcd = marks[0]
        for m in marks[1:]:
            gcd = math.gcd(gcd, m)
        min_acceptable = max(1, total_steps // 20)
        if gcd >= min_acceptable:
            return gcd
        return max(1, total_steps // 10)


def _verifier_env(fn: Callable[..., Any], *, expected: Any, params: dict | None = None):
    """Build a single-turn in-process :data:`RolloutEnv` from a registered
    verifier fn ``(output, expected, params) -> reward``. It scores the last
    assistant message and ends the episode (no sandbox)."""
    p = dict(params or {})

    async def env(messages: list[dict]) -> EnvStep:
        last = messages[-1].get("content", "") if messages else ""
        reward = float(fn(last, expected, p))
        return EnvStep(done=True, reward=reward)

    return env


__all__ = ["NativeRL", "NativeRLConfig"]

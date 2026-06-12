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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar

import tinker
from pydantic import BaseModel, ConfigDict, Field

from ..protocols import RunContext, RunResult
from ..registry import register_algorithm
from ..training.env import EnvGroupBuilder, SingleTurnEnv, VerifierFn
from ..training.loop import TrainingLoop
from ..training.step_builder import RLStepBuilder
from ..training.templates import messages_to_model_input
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
    """Looks up :func:`evsys_sdk.registry.get_verifier_fn` at runtime. Required
    unless ``ctx.extras['env_builders']`` is set by the runner."""

    max_tokens: int = 256
    temperature: float = 1.0
    drop_constant_reward: bool = True

    system_prompt: str | None = None
    user_template: str = "{prompt}"

    save_every: int = 0
    save_at_fractions: list[float] = Field(default_factory=lambda: [1.0])
    eval_every: int = 0

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8


@dataclass
class _SimpleRLDataset:
    builders: list[EnvGroupBuilder]
    batch_size: int

    def __post_init__(self) -> None:
        if not self.builders:
            raise ValueError("_SimpleRLDataset: builders is empty")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0 (got {self.batch_size})")

    def __len__(self) -> int:
        return max(1, len(self.builders) // self.batch_size)

    def get_batch(self, step_idx: int) -> list[EnvGroupBuilder]:
        n = len(self.builders)
        start = (step_idx * self.batch_size) % n
        end = start + self.batch_size
        if end <= n:
            return self.builders[start:end]
        return self.builders[start:] + self.builders[: end - n]


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

        builders = ctx.extras.get("env_builders")
        if not builders:
            rows = ctx.extras.get("train_rows")
            if not rows:
                raise RuntimeError(
                    "NativeRL.train: provide either ctx.extras['env_builders'] "
                    "or ctx.extras['train_rows'] with 'prompt' + 'expected'."
                )
            builders = self._builders_from_rows(rows, backend.get_tokenizer())

        dataset = _SimpleRLDataset(
            builders=list(builders), batch_size=self.cfg.batch_size,
        )

        steps_per_epoch = max(1, len(dataset))
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
            "n_builders": len(dataset.builders),
            "total_steps": total_steps,
            "save_every": save_every,
        })

        snapshot_counter = {"i": 0}

        async def _latest_sampler():
            snapshot_counter["i"] += 1
            return await backend.snapshot_sampling_client(
                name=f"rl_snap_{snapshot_counter['i']}"
            )

        step_builder = RLStepBuilder(
            dataset=dataset,
            student_sampler_provider=_latest_sampler,
            num_samples=self.cfg.num_samples,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            drop_constant_reward=self.cfg.drop_constant_reward,
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
            evaluators=[],
        )
        artifacts = await loop.run(num_steps=total_steps)

        d = artifacts.as_dict()
        for k, v in d.items():
            ctx.log_store.log_artifact(k, v, kind="checkpoint")

        return RunResult(
            run_id=ctx.run_id, status="completed", metrics={}, artifacts=d,
        )

    def _builders_from_rows(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
    ) -> list[EnvGroupBuilder]:
        from ..verifiers import get_verifier_fn
        if not self.cfg.verifier_name:
            raise RuntimeError(
                "NativeRL: cfg.verifier_name unset and no env_builders in extras"
            )
        try:
            verifier_fn = get_verifier_fn(self.cfg.verifier_name)
        except ValueError as e:
            raise RuntimeError(
                f"NativeRL: unknown verifier_name {self.cfg.verifier_name!r}"
            ) from e

        verifier = _wrap_verifier(verifier_fn)
        builders: list[EnvGroupBuilder] = []
        for r in rows:
            prompt_str = r.get("prompt") or r.get("question") or r.get("instruction")
            expected = r.get("expected") or r.get("golden_answer") or r.get("answer")
            if not prompt_str or expected is None:
                continue
            messages = []
            if self.cfg.system_prompt:
                messages.append({"role": "system", "content": self.cfg.system_prompt})
            messages.append({
                "role": "user",
                "content": self.cfg.user_template.format(prompt=prompt_str),
            })
            prompt_mi = messages_to_model_input(
                tokenizer, messages,
                add_generation_prompt=True,
                enable_thinking=self.cfg.enable_thinking,
            )
            tags = list(r.get("tags") or [])
            metadata = {k: v for k, v in r.items()
                        if k not in {"prompt", "question", "instruction",
                                     "expected", "golden_answer", "answer", "tags"}}
            builders.append(SingleTurnEnv(
                prompt=prompt_mi, expected=expected, tokenizer=tokenizer,
                verifier=verifier, tags=tags, metadata=metadata,
            ))
        if not builders:
            raise RuntimeError("NativeRL: no usable rows (need prompt + expected)")
        return builders

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


def _wrap_verifier(fn: Callable[..., Any]) -> VerifierFn:
    """Adapt a registered ``(output, expected, params)`` verifier to the
    :class:`VerifierFn` shape (``(output, expected)``)."""
    def _call(output: str, expected: Any) -> float:
        return float(fn(output, expected, {}))
    return _call


__all__ = ["NativeRL", "NativeRLConfig"]

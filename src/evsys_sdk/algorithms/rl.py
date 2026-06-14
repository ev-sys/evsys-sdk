"""RL — on-policy reinforcement learning on the SDK training loop.

Single-turn out of the box (the path most evsys projects start with —
prompt → completion → verifier reward); multi-turn slots in via the
:class:`~evsys_sdk.training.env.EnvGroupBuilder` Protocol when a project
needs it.

All the composer plumbing lives in
:class:`~evsys_sdk.algorithms.base.BaseAlgorithm`; RL supplies:

* :meth:`setup` — build per-row single-turn envs (from HarborTask rows or
  pre-built ``ctx.extras['env_builders']``) and a per-step student sampler
  provider (snapshots the current weights each step → on-policy).
* :meth:`build_batch` — rollout the batch's envs, group-normalize advantages,
  emit importance-sampling-loss Datums.

Researchers supply ``train_rows`` (HarborTask shape: ``task_id`` +
``instruction`` + a per-row ``in_process`` verifier), or pre-built
``EnvGroupBuilder`` instances via ``ctx.extras['env_builders']`` for non-text
envs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, cast

from ..data_types import HarborTask, InProcessVerifier, TargetFormat, parse_rows
from ..protocols import RunContext
from ..registry import register_algorithm
from ..training.env import EnvGroupBuilder, SingleTurnEnv, VerifierFn
from ..training.loop import TrainingBatch
from ..training.templates import messages_to_model_input
from ..training.tinker_backend import TinkerBackend
from .base import BaseAlgorithm, BaseAlgorithmConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class RLConfig(BaseAlgorithmConfig):
    """Config for :class:`RL`. Inherits the shared training/save/eval knobs
    from :class:`BaseAlgorithmConfig`; adds RL-only fields."""

    learning_rate: float = 1.0e-5
    """RL needs a lower LR than SFT — IS gradients can be large."""

    num_samples: int = 1
    """Rollouts per builder — ``group_size`` in the cookbook."""

    verifier_name: str | None = None
    """Fallback verifier-fn name used only when a HarborTask's InProcessVerifier
    leaves ``fn_name`` empty. Normally the verifier is carried per-row by the
    HarborTask itself (looked up via :func:`evsys_sdk.verifiers.get_verifier_fn`)."""

    max_tokens: int = 256
    temperature: float = 1.0
    drop_constant_reward: bool = True

    system_prompt: str | None = None
    user_template: str = "{prompt}"


# ---------------------------------------------------------------------------
# Per-step dataset over env builders
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


@register_algorithm("rl")
class RL(BaseAlgorithm):
    name: ClassVar[str] = "rl"
    Config: ClassVar[type] = RLConfig

    def _check_inputs(self, ctx: RunContext) -> None:
        if not ctx.extras.get("env_builders") and not ctx.extras.get("train_rows"):
            raise RuntimeError(
                "RL.train: provide either ctx.extras['env_builders'] or "
                "ctx.extras['train_rows'] of HarborTask rows "
                "(task_id + instruction + verifier)."
            )

    async def setup(self, ctx: RunContext, backend: TinkerBackend) -> None:
        builders = ctx.extras.get("env_builders")
        if not builders:
            builders = self._builders_from_rows(
                ctx.extras["train_rows"], backend.get_tokenizer()
            )
        self._dataset = _SimpleRLDataset(
            builders=list(builders), batch_size=self.cfg.batch_size,
        )
        self._n_builders = len(self._dataset.builders)
        self._backend = backend
        self._snapshot_i = 0
        self._steps_per_epoch = max(1, len(self._dataset))

    async def _latest_sampler(self) -> Any:
        self._snapshot_i += 1
        return await self._backend.snapshot_sampling_client(
            name=f"rl_snap_{self._snapshot_i}"
        )

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        from ..training.data_processing import (
            assemble_training_data,
            compute_advantages,
            compute_trajectory_metrics,
        )
        from ..training.rollouts import do_group_rollouts

        builders = list(self._dataset.get_batch(step_idx))
        sampler = await self._latest_sampler()
        groups = await do_group_rollouts(
            sampler=sampler, builders=builders,
            num_samples=self.cfg.num_samples,
            max_tokens=self.cfg.max_tokens, temperature=self.cfg.temperature,
            drop_constant_reward=self.cfg.drop_constant_reward,
        )
        if not groups:
            # No usable groups — emit an empty batch + a zeroed metric row so
            # the loop's step counter advances cleanly.
            return TrainingBatch(
                data=[], loss_fn="importance_sampling",
                metrics={"reward/n_trajectories": 0.0},
            )

        advantages = compute_advantages(groups)
        datums, _meta = assemble_training_data(groups, advantages)
        metrics = compute_trajectory_metrics(groups)
        return TrainingBatch(
            data=datums, loss_fn="importance_sampling", metrics=metrics,
        )

    def _hyperparams_extra(self) -> dict[str, Any]:
        return {"n_builders": self._n_builders}

    # --- internals ---------------------------------------------------------

    def _builders_from_rows(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
    ) -> list[EnvGroupBuilder]:
        from ..verifiers import get_verifier_fn

        # Standardize raw rows → typed HarborTask (strict): instruction is the
        # prompt, the verifier spec rides on each task.
        tasks = cast("list[HarborTask]", parse_rows(rows, TargetFormat.HARBOR_TASK))

        builders: list[EnvGroupBuilder] = []
        for t in tasks:
            v = t.verifier
            if not isinstance(v, InProcessVerifier):
                raise RuntimeError(
                    f"RL: task {t.task_id!r} uses a {v.kind!r} verifier; "
                    "the training rollout path executes only 'in_process' "
                    "verifiers today (e2b / llm_judge are not yet wired)."
                )
            fn_name = v.fn_name or self.cfg.verifier_name
            if not fn_name:
                raise RuntimeError(
                    f"RL: task {t.task_id!r} has no verifier fn_name and "
                    "cfg.verifier_name is unset."
                )
            try:
                verifier_fn = get_verifier_fn(fn_name)
            except ValueError as e:
                raise RuntimeError(f"RL: unknown verifier_name {fn_name!r}") from e
            verifier = _wrap_verifier(verifier_fn, params=v.params)

            messages = []
            if self.cfg.system_prompt:
                messages.append({"role": "system", "content": self.cfg.system_prompt})
            messages.append({
                "role": "user",
                "content": self.cfg.user_template.format(prompt=t.instruction),
            })
            prompt_mi = messages_to_model_input(
                tokenizer, messages,
                add_generation_prompt=True,
                enable_thinking=self.cfg.enable_thinking,
            )
            tags = list(t.metadata.get("tags") or [])
            metadata = {k: val for k, val in t.metadata.items() if k != "tags"}
            metadata.setdefault("task_id", t.task_id)
            builders.append(SingleTurnEnv(
                prompt=prompt_mi, expected=v.expected, tokenizer=tokenizer,
                verifier=verifier, tags=tags, metadata=metadata,
            ))
        if not builders:
            raise RuntimeError("RL: no usable HarborTask rows")
        return builders


def _wrap_verifier(fn: Callable[..., Any], params: dict | None = None) -> VerifierFn:
    """Adapt a registered ``(output, expected, params)`` verifier to the
    :class:`VerifierFn` shape (``(output, expected)``). ``params`` comes from
    the HarborTask's InProcessVerifier spec."""
    params = dict(params or {})

    def _call(output: str, expected: Any) -> float:
        return float(fn(output, expected, params))
    return _call


__all__ = ["RL", "RLConfig"]

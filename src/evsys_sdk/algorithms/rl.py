"""RL — on-policy reinforcement learning, rollouts run by harbor's engine.

Rollouts are handed to **harbor's ``Job`` engine** (retries, bounded
concurrency, timeouts, persistence) via
:func:`evsys_sdk.training.harbor_engine.run_harbor_rollouts`; this algorithm
just turns ``HarborTask`` rows into the engine's inputs and assembles the
returned trajectories into importance-sampling-loss Datums.

Composer plumbing lives in :class:`~evsys_sdk.algorithms.base.BaseAlgorithm`;
RL supplies:

* :meth:`setup` — parse ``HarborTask`` rows; stash the backend + the
  ``.evsys`` rollout workspace.
* :meth:`build_batch` — save a sampler checkpoint (on-policy), roll out the
  batch via harbor, group-normalize advantages, emit IS-loss Datums.

The agent harness is pluggable: by default harbor runs our ``BasicLoopAgent``
(``Chat(TinkerLLM)``); set ``agent_import_path`` to any harbor ``BaseAgent``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, cast

from ..data_types import HarborTask, InProcessVerifier, TargetFormat, parse_rows
from ..protocols import RunContext
from ..registry import register_algorithm
from ..training.loop import TrainingBatch
from ..training.tinker_backend import TinkerBackend
from .base import BaseAlgorithm, BaseAlgorithmConfig


class RLConfig(BaseAlgorithmConfig):
    """Config for :class:`RL` — shared knobs from :class:`BaseAlgorithmConfig`
    plus RL/rollout-engine fields."""

    learning_rate: float = 1.0e-5
    """RL needs a lower LR than SFT — IS gradients can be large."""

    num_samples: int = 1
    """Rollouts per task (``group_size``); >= 2 enables the advantage baseline."""

    verifier_name: str | None = None
    """Fallback verifier-fn name when a HarborTask's InProcessVerifier leaves
    ``fn_name`` empty (normally the verifier rides per-row on the task)."""

    max_tokens: int = 256
    temperature: float = 1.0
    max_turns: int = 1
    drop_constant_reward: bool = True

    system_prompt: str | None = None
    user_template: str = "{prompt}"

    # Harbor engine knobs.
    agent_import_path: str | None = None
    """Override the rollout agent harness (any harbor ``BaseAgent`` import path).
    Default: our ``BasicLoopAgent``."""
    n_concurrent: int = 4
    max_retries: int = 2


@register_algorithm("rl")
class RL(BaseAlgorithm):
    name: ClassVar[str] = "rl"
    Config: ClassVar[type] = RLConfig

    def _check_inputs(self, ctx: RunContext) -> None:
        if not ctx.extras.get("train_rows"):
            raise RuntimeError(
                "RL.train: ctx.extras['train_rows'] missing/empty "
                "(HarborTask rows: task_id + instruction + verifier)."
            )

    async def setup(self, ctx: RunContext, backend: TinkerBackend) -> None:
        rows = ctx.extras["train_rows"]
        tasks = cast("list[HarborTask]", parse_rows(rows, TargetFormat.HARBOR_TASK))
        self._tasks = [self._prep_task(t) for t in tasks]
        self._backend = backend
        self._snapshot_i = 0
        # Rollouts are materialized + persisted under the run's workspace on
        # disk; training rollouts are NOT uploaded to the dashboard (only eval
        # rollouts are — see harbor_eval). When a RunLog is present, route
        # harbor's jobs dir into the agent track so the dense rollout store is
        # harbor-native (referenced, not copied); the human track gets a curated
        # readable view via log_training_rollouts below.
        self._run_log = ctx.extras.get("run_log")
        self._workspace = (
            self._run_log.harbor_dir("train") if self._run_log is not None
            else Path(ctx.output_dir) / "harbor_rollouts"
        )
        self._steps_per_epoch = max(1, len(self._tasks) // self.cfg.batch_size)

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        from ..training.data_processing import (
            assemble_training_data,
            compute_advantages,
            compute_trajectory_metrics,
        )
        from ..training.harbor_engine import run_harbor_rollouts

        batch = self._slice(step_idx)
        self._snapshot_i += 1
        model_path = await self._backend.save_for_sampler(f"rl_snap_{self._snapshot_i}")

        groups = await run_harbor_rollouts(
            batch,                       # HarborTasks → outcome_reward=True (default), scored
            model_name=self._model_name,
            model_path=model_path,
            workspace_dir=self._workspace,
            renderer_name=self.cfg.renderer_name,
            num_samples=self.cfg.num_samples,
            max_turns=self.cfg.max_turns,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system_prompt=self.cfg.system_prompt,
            agent_import_path=self.cfg.agent_import_path,
            n_concurrent=self.cfg.n_concurrent,
            max_retries=self.cfg.max_retries,
        )
        # Readable training rollouts (predictions + per-token logprobs + reward
        # & advantage per sample/group). Full per-trial store is harbor's own,
        # under harbor/train. Bounded cadence so the folder stays clean.
        if self._run_log is not None and (step_idx == 0 or step_idx % 10 == 0):
            from ..training.data_processing import compute_advantages
            tok = getattr(self._backend, "get_tokenizer", lambda: None)()
            self._run_log.log_training_rollouts(
                step_idx, groups, advantages=compute_advantages(groups),
                tokenizer=tok, tasks=batch,
            )
        if self.cfg.drop_constant_reward:
            groups = [g for g in groups if not _all_equal(g.rewards)]
        if not groups:
            return TrainingBatch(
                data=[], loss_fn="importance_sampling",
                metrics={"reward/n_trajectories": 0.0},
            )

        advantages = compute_advantages(groups)
        datums, _meta = assemble_training_data(groups, advantages)
        metrics = compute_trajectory_metrics(groups)
        return TrainingBatch(data=datums, loss_fn="importance_sampling", metrics=metrics)

    def _hyperparams_extra(self) -> dict[str, Any]:
        return {"n_tasks": len(self._tasks)}

    # --- internals ---------------------------------------------------------

    def _prep_task(self, t: HarborTask) -> HarborTask:
        """Template the instruction + fill the verifier fn_name fallback, so the
        materialized harbor task is self-contained."""
        v = t.verifier
        if not isinstance(v, InProcessVerifier):
            raise RuntimeError(
                f"RL: task {t.task_id!r} uses a {v.kind!r} verifier; only "
                "'in_process' is supported in the rollout path today."
            )
        fn_name = v.fn_name or self.cfg.verifier_name
        if not fn_name:
            raise RuntimeError(
                f"RL: task {t.task_id!r} has no verifier fn_name and "
                "cfg.verifier_name is unset."
            )
        # Fail fast on an unknown verifier fn (EvsysVerifier re-resolves it at
        # harbor runtime, but surfacing it here avoids a costly rollout).
        from ..verifiers import get_verifier_fn

        try:
            get_verifier_fn(fn_name)
        except ValueError as e:
            raise RuntimeError(f"RL: unknown verifier_name {fn_name!r}") from e
        return HarborTask(
            task_id=t.task_id,
            instruction=self.cfg.user_template.format(prompt=t.instruction),
            verifier=InProcessVerifier(fn_name=fn_name, expected=v.expected, params=v.params),
            metadata=t.metadata,
        )

    def _slice(self, step_idx: int) -> list[HarborTask]:
        n = len(self._tasks)
        start = (step_idx * self.cfg.batch_size) % n
        end = start + self.cfg.batch_size
        if end <= n:
            return self._tasks[start:end]
        return self._tasks[start:] + self._tasks[: end - n]


def _all_equal(xs: list[float]) -> bool:
    return len(xs) > 0 and all(x == xs[0] for x in xs)


__all__ = ["RL", "RLConfig"]

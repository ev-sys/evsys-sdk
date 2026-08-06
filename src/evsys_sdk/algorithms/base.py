"""BaseAlgorithm — shared base for the SDK's training algorithms.

Every training algorithm (``sft`` / ``sdft`` / ``rl``) used to copy-paste the
same composer body: allocate the tinker backend,
resolve the step count + save cadence, log hyperparams, build the in-loop
evaluators, wire a :class:`~evsys_sdk.training.loop.TrainingLoop`, run it, and
record the checkpoint artifacts. The only thing that genuinely differed was
*how each step's batch is built*. This base owns all of the shared plumbing;
a concrete algorithm overrides just the per-algorithm pieces.

The base **is itself a StepBuilder** — it implements ``build_batch`` and
``step_metrics`` and hands ``self`` to the loop, so there is no separate
StepBuilder object to construct. A subclass overrides:

* :meth:`setup` — one-time prep before the loop. Stash per-algorithm state on
  ``self`` (tokenized datums for SFT; dataset + teacher client + sampler
  provider for SDFT/RL) and set ``self._steps_per_epoch``.
* :meth:`build_batch` — produce the :class:`TrainingBatch` for one step. The
  loss spec rides on the returned batch (``loss_fn`` is not a separate
  override). SFT slices its static datums; RL/SDFT roll out on-policy here.
* :meth:`step_metrics` — *(optional)* per-step metrics from the
  forward-backward result. Defaults to ``{}``.

Researchers wanting a one-line tweak (focal loss, an extra metric, a custom
loss) subclass the concrete algorithm and override ``build_batch`` /
``step_metrics`` — no SDK change needed.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any, ClassVar

import tinker
from pydantic import BaseModel, ConfigDict, Field

from ..config import CallbackSpec
from ..protocols import RunContext, RunResult
from ..training.callbacks import build_callbacks, with_default_snapshots
from ..training.evaluators import build_in_loop_evaluators
from ..training.loop import TrainingBatch, TrainingLoop
from ..training.tinker_backend import TinkerBackend

# ---------------------------------------------------------------------------
# Shared config base
# ---------------------------------------------------------------------------


class BaseAlgorithmConfig(BaseModel):
    """Fields common to every training algorithm.

    Concrete algorithms subclass this and add their own knobs (SDFT: ``topk``;
    RL: ``num_samples`` / ``verifier_name`` / …). ``extra="forbid"`` so a YAML
    typo fails loudly. A subclass may re-declare a field to change its default
    (e.g. RL drops ``learning_rate`` to ``1e-5``).
    """

    model_config = ConfigDict(extra="forbid")

    # Training cadence
    learning_rate: float = 1.0e-4
    num_epochs: int = 1
    batch_size: int = Field(default=4, gt=0)
    max_steps: int | None = None
    """Hard step cap. When set, wins over ``num_epochs * steps_per_epoch``."""

    # Model / LoRA / renderer
    lora_rank: int = 8
    renderer_name: str | None = None
    enable_thinking: bool | None = None

    # Checkpoint cadence
    save_every: int = 0
    """If 0, computed from ``save_at_fractions`` (GCD-of-marks heuristic)."""
    save_at_fractions: list[float] = Field(default_factory=lambda: [1.0])

    # Training-loop callbacks ({kind, params}); resolved through the callback
    # registry and attached to the loop. e.g.
    #   callbacks: [{kind: early_stopping, params: {metric: pass_rate}}]
    callbacks: list[CallbackSpec] = Field(default_factory=list)

    # Adam knobs (passthrough to tinker.AdamParams)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8

    # Optional wandb hookup (off by default; the SDK doesn't pull wandb).
    wandb_project: str | None = None
    wandb_name: str | None = None


# ---------------------------------------------------------------------------
# Base algorithm (also a StepBuilder)
# ---------------------------------------------------------------------------


class BaseAlgorithm:
    """Template-method base for the SDK's training algorithms. See module
    docstring.

    Subclasses set the ``name`` / ``Config`` ClassVars (the registry contract)
    and override :meth:`setup` + :meth:`build_batch` (+ optionally
    :meth:`step_metrics`).
    """

    name: ClassVar[str]
    Config: ClassVar[type]

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = self.Config.model_validate(kwargs)
        # Set by setup(); read by _resolve_total_steps() and the loop.
        self._steps_per_epoch: int = 1

    # --- StepBuilder protocol surface --------------------------------------

    @property
    def steps_per_epoch(self) -> int:
        return self._steps_per_epoch

    async def setup(self, ctx: RunContext, backend: TinkerBackend) -> None:
        """One-time prep before the loop. Must set ``self._steps_per_epoch``.
        Override in the concrete algorithm."""
        raise NotImplementedError

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        """Produce the batch for ``step_idx`` (0-based). Override."""
        raise NotImplementedError

    def step_metrics(
        self, step_idx: int, batch: TrainingBatch, fb_result: Any,
    ) -> dict[str, float]:
        """Per-step metrics from the forward-backward result. Default no-op."""
        return {}

    # --- generic driver ----------------------------------------------------

    def train(self, ctx: RunContext) -> RunResult:
        if ctx.backend.name != "tinker":
            raise RuntimeError(
                f"{type(self).__name__} requires backend=tinker "
                f"(got '{ctx.backend.name}')."
            )
        return asyncio.run(self._train_async(ctx))

    async def _train_async(self, ctx: RunContext) -> RunResult:
        # Validate inputs BEFORE allocating the (real, costly) backend so a
        # missing dataset / model_name fails fast without spinning up tinker.
        self._check_inputs(ctx)

        handles = ctx.extras.get("backend_handles", {})
        model_name = handles.get("model_name") or ctx.extras.get("model_name")
        if not model_name:
            raise RuntimeError("model_name not set in backend handles")
        # Stash so setup() can reach it (e.g. SDFT builds a teacher client over
        # the same base model).
        self._model_name = model_name

        # 1. backend (async factory; allocates the LoRA training client)
        backend = await TinkerBackend.create(
            model_name=model_name,
            lora_rank=self.cfg.lora_rank,
            renderer_name=self.cfg.renderer_name or handles.get("renderer_name"),
            resume_state_path=handles.get("load_checkpoint_path"),
            init_weights_path=handles.get("init_from_checkpoint"),
        )

        # 2. per-algorithm prep (sets self._steps_per_epoch + stashes state)
        await self.setup(ctx, backend)

        # 3. total step count + save cadence
        total_steps = self._resolve_total_steps()
        save_every = self._resolve_save_every(total_steps)

        # 5. compose the loop (self IS the StepBuilder) and run
        evaluators = build_in_loop_evaluators(
            ctx.config.metadata if hasattr(ctx, "config") else None,
            tokenizer=backend.get_tokenizer(),
            store=getattr(ctx, "store", None) or ctx.extras.get("store"),
            model_name=model_name,
            workspace_dir=Path(ctx.output_dir) / ".harbor" / "val",
            run_id=ctx.extras.get("dashboard_run_id"),
        )
        # Algorithm's own loop-only callbacks (e.g. early_stopping) PLUS the
        # experiment's shared logger instances threaded down via extras, so one
        # logger sees both the loop-scope and experiment-scope hooks.
        callbacks = build_callbacks(self.cfg.callbacks) + list(
            ctx.extras.get("callbacks") or []
        )
        # Router-managed nodes snapshot by default — the yaml never mentions
        # delta_snapshot; the provisioner's env is the switch.
        callbacks = with_default_snapshots(callbacks)

        # Surface the final training data (post-transform / chat-template rows)
        # to loggers so they can persist exactly what went into the model.
        self._dispatch_train_data(ctx, callbacks)

        loop = TrainingLoop(
            backend=backend,
            step_builder=self,
            # Per-step / eval metrics flow ONLY through the callbacks
            # (-> local_logger): there is no log_store on the loop path.
            output_dir=Path(ctx.output_dir),
            adam_params=tinker.AdamParams(
                learning_rate=self.cfg.learning_rate,
                beta1=self.cfg.adam_beta1,
                beta2=self.cfg.adam_beta2,
                eps=self.cfg.adam_eps,
            ),
            save_every=save_every,
            evaluators=evaluators,
            callbacks=callbacks,
            log_context=ctx.extras.get("log_context"),
            log_rollouts=bool(ctx.extras.get("log_rollouts")),
        )
        artifacts = await loop.run(num_steps=total_steps)

        # 6. record run_dir + per-checkpoint sampler URIs so downstream
        # consumers (TinkerInference.from_run_result, Experiment._eval_arm)
        # keep working unchanged.
        # Checkpoint URIs ride out on RunResult.artifacts (consumed by
        # TinkerInference.from_run_result, Experiment._eval_arm).
        artifacts_dict = artifacts.as_dict()

        return RunResult(
            run_id=ctx.run_id,
            status="completed",
            metrics={},
            artifacts=artifacts_dict,
        )

    # --- hooks / helpers ---------------------------------------------------

    def _train_data_rows(self, ctx: RunContext) -> list[dict[str, Any]]:
        """Best-effort: the final examples fed to the model (post-transform /
        chat-template rows). Default returns ``ctx.extras['train_rows']`` coerced
        to dicts — which for SFT are the standardized chat-message rows. Override
        for algorithm-specific shapes."""
        rows = ctx.extras.get("train_rows") or []
        out: list[dict[str, Any]] = []
        for r in rows:
            if isinstance(r, dict):
                out.append(r)
            elif hasattr(r, "model_dump"):
                out.append(r.model_dump())
            else:
                out.append({"value": str(r)})
        return out

    def _dispatch_train_data(self, ctx: RunContext, callbacks: list[Any]) -> None:
        """Fire ``on_train_data`` on each callback with the final training rows.
        Never raises — logging must not break training."""
        log_ctx = ctx.extras.get("log_context")
        try:
            rows = self._train_data_rows(ctx)
        except Exception:  # pragma: no cover
            return
        for cb in callbacks:
            try:
                cb.on_train_data(log_ctx, rows)
            except Exception:  # pragma: no cover
                pass

    def _check_inputs(self, ctx: RunContext) -> None:
        """Validate ``ctx.extras`` before the backend is allocated. Override to
        fail fast on missing / empty inputs (e.g. no train_rows). Default
        no-op."""

    def _hyperparams_extra(self) -> dict[str, Any]:
        """Extra fields merged into the logged hyperparams. Override to add
        algorithm-specific counts (e.g. ``n_train_rows``)."""
        return {}

    def _resolve_total_steps(self) -> int:
        if self.cfg.max_steps is not None:
            return self.cfg.max_steps
        return self._steps_per_epoch * self.cfg.num_epochs

    def _resolve_save_every(self, total_steps: int) -> int:
        """Resolve the checkpoint cadence from ``save_every`` /
        ``save_at_fractions``. Identical heuristic across all algos: take the
        GCD of the requested fraction-marks, but fall back to ``total/10`` if
        that GCD would pathologically save every other step."""
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


__all__ = ["BaseAlgorithm", "BaseAlgorithmConfig"]

"""TinkerSDFT — Self-Distillation Fine-Tuning via tinker_cookbook.

Thin orchestrator over ``tinker_cookbook.distillation.sdft``. The student
generates completions on-policy; the teacher (a static or periodically
re-synced copy of the base model) scores them conditioned on a golden-answer
demonstration. The student is trained to match the teacher's top-K token
distribution via ``cross_entropy`` (or per-token importance sampling when
``topk = 0``).

Data contract:
    Each input row must have ``question`` and ``golden_answer`` string keys.
    Optional ``toolkit`` / ``tool_slug`` (or other metadata) are ignored by
    this wrapper.

    Differs from :class:`TinkerSFT`'s ``{messages: [...]}`` chat shape — see
    :mod:`tinker_cookbook.distillation.sdft` for the underlying contract.

Checkpoint harvest + run_dir reporting follow the same pattern as
``TinkerSFT`` so downstream consumers (forward_step_metrics,
Checkpoint.pick_final, inference factories) work unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..protocols import RunContext, RunResult
from ..registry import register_algorithm

# Raise ImportError at module load when tinker_cookbook isn't installed —
# matches TinkerSFT's "fail fast" pattern. Mock backends should use
# `mock_sdft` (not yet implemented) instead.
import chz  # noqa: E402  (used implicitly via sdft.Config)
import tinker  # noqa: E402
import torch as _torch  # noqa: E402
from tinker_cookbook import renderers  # noqa: E402
from tinker_cookbook.distillation import sdft  # noqa: E402
from tinker_cookbook.recipes.sdft.datasets import SDFTDataset  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-step loss logging
# ---------------------------------------------------------------------------
# The cookbook's `sdft.main` loop calls
# ``tinker_cookbook.rl.train.train_step`` and only merges
# ``optim_result.metrics`` (typically empty for SDFT's cross_entropy path)
# into the per-step metrics dict — so ``metrics.jsonl`` ends up without any
# loss field. We wrap the function so the per-position student logprobs it
# returns get aggregated into ``train/mean_loss`` (negative of mean logprob
# over the loss-mask positions) and ``train/mean_logprob`` for inspection.
#
# Patching ``sdft.train_step`` (the binding that the cookbook's ``sdft.main``
# looks up by name) is the lightest-touch way to do this without forking
# the whole 200-line training loop.

_orig_sdft_train_step = sdft.train_step


async def _sdft_train_step_with_loss(
    data_D,
    training_client,
    learning_rate,
    num_substeps,
    loss_fn,
    loss_fn_config=None,
    metrics=None,
):
    """Wrap cookbook ``train_step`` to record per-step training loss."""
    training_logprobs_D = await _orig_sdft_train_step(
        data_D=data_D,
        training_client=training_client,
        learning_rate=learning_rate,
        num_substeps=num_substeps,
        loss_fn=loss_fn,
        loss_fn_config=loss_fn_config,
        metrics=metrics,
    )
    if metrics is not None and training_logprobs_D:
        try:
            flat = _torch.cat([
                lp.detach().flatten().float() for lp in training_logprobs_D
            ])
            # Non-loss-mask positions are written as 0 by the forward pass;
            # ignore them in the aggregate so the metric reflects only the
            # tokens that actually contributed to the loss.
            valid = flat[flat != 0]
            n = int(valid.numel())
            if n > 0:
                mean_lp = float(valid.sum().item() / n)
                metrics["train/mean_logprob"] = mean_lp
                metrics["train/mean_loss"] = -mean_lp
                metrics["train/loss_n_tokens"] = float(n)
        except Exception:  # pragma: no cover  — purely additive; never block
            logger.exception("TinkerSDFT: failed to compute per-step loss metric")
    return training_logprobs_D


sdft.train_step = _sdft_train_step_with_loss


class _RepeatingSDFTProvider:
    """Wrap an ``SDFTDataset`` so it serves a target step count by wrapping
    around (modulo the underlying length).

    ``tinker_cookbook.distillation.sdft.main`` does
    ``num_batches = min(cfg.max_steps, len(sdft_dataset))`` — so a small
    dataset caps training even when ``max_steps`` is much larger. This
    proxy lets ``max_steps`` extend training past a single epoch by
    repeating ``get_batch`` indices.

    The proxy is a no-op when ``target_length <= len(base)`` — used in that
    case we just return the base.
    """

    def __init__(self, base: Any, target_length: int) -> None:
        self._base = base
        self._target_length = int(target_length)
        self._base_length = len(base)
        if self._base_length <= 0:
            raise ValueError("base SDFT provider has zero length")

    def __len__(self) -> int:
        return self._target_length

    def get_batch(self, index: int) -> Any:
        return self._base.get_batch(index % self._base_length)


class TinkerSDFTConfig(BaseModel):
    """Config for :class:`TinkerSDFT`.

    The defaults mirror the cookbook's recommendations for top-K distillation
    (``topk=20``) with a learning rate sized for LoRA on small models. Adjust
    ``learning_rate`` upward (5e-4 .. 1e-3) for larger LoRA ranks.
    """

    model_config = ConfigDict(extra="forbid")

    # Training cadence
    learning_rate: float = 1e-4
    num_epochs: int = 1
    batch_size: int = 4
    """``groups_per_batch`` in the cookbook's vocabulary."""
    group_size: int = 1
    """Rollouts per question per step. The cookbook defaults to 1."""
    max_steps: int | None = None

    # Model / LoRA
    lora_rank: int = 8
    renderer_name: str | None = None
    """Override ``model.renderer_name``; required by the cookbook."""

    # SDFT-specific
    topk: int = 20
    """Top-K teacher distribution to distill. ``0`` → IS fallback."""
    teacher_sync_every: int | None = None
    """Hard-sync student weights into the teacher every N steps. ``None`` → frozen teacher."""
    max_context_length: int = 2048
    demo_template: str = sdft.DEFAULT_DEMO_TEMPLATE
    system_prompt: str | None = None

    # Generation knobs used during on-policy rollout
    max_tokens: int = 256
    temperature: float = 1.0

    # Checkpointing / eval cadence (in steps)
    save_every: int = 0
    """If 0, computed from save_at_fractions."""
    save_at_fractions: list[float] = Field(default_factory=lambda: [1.0])
    eval_every: int = 0

    wandb_project: str | None = None
    wandb_name: str | None = None


_DEPRECATION_MSG = (
    "TinkerSDFT (algorithm.kind: tinker_sdft) delegates to tinker_cookbook "
    "and is deprecated in favor of `sdft`, which runs the loop "
    "natively in the SDK (no cookbook dep, no monkey-patching of "
    "sdft.train_step for per-step loss). Flip algorithm.kind to "
    "'sdft' when ready; the YAML surface is identical."
)


@register_algorithm("tinker_sdft")
class TinkerSDFT:
    name: ClassVar[str] = "tinker_sdft"
    Config: ClassVar[type] = TinkerSDFTConfig

    def __init__(self, **kwargs: Any) -> None:
        import warnings
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self.cfg = TinkerSDFTConfig.model_validate(kwargs)

    def _resolve_save_every(self, total_steps: int) -> int:
        if self.cfg.save_every:
            return self.cfg.save_every
        # Each requested fraction f → its target step ``round(f * total_steps)``.
        # The cookbook only saves when ``(step + 1) % save_every == 0``, so we
        # pick the largest save_every that divides every target — the GCD of
        # the mark set. That way save_at_fractions=[0.2, 0.4, 0.6, 0.8] with
        # total_steps=1000 → marks={200, 400, 600, 800} → save_every=200 →
        # checkpoints land exactly at steps 200, 400, 600, 800, 1000.
        marks = sorted({
            max(1, int(round(f * total_steps))) for f in self.cfg.save_at_fractions
        })
        if not marks:
            return max(1, total_steps)
        from math import gcd
        g = marks[0]
        for m in marks[1:]:
            g = gcd(g, m)
        return max(1, g)

    def train(self, ctx: RunContext) -> RunResult:
        if ctx.backend.name != "tinker":
            raise RuntimeError(
                f"TinkerSDFT requires backend=tinker (got '{ctx.backend.name}')."
            )

        rows = ctx.extras.get("train_rows")
        if not rows:
            raise RuntimeError("TinkerSDFT.train: ctx.extras['train_rows'] missing/empty")

        # Validate the SDFT data contract.
        missing_fields = [
            i for i, r in enumerate(rows[:5])
            if not r.get("question") or not r.get("golden_answer")
        ]
        if missing_fields:
            raise RuntimeError(
                f"TinkerSDFT: each row must have non-empty `question` and "
                f"`golden_answer`; rows at indices {missing_fields} (of first 5) are missing one or both."
            )

        handles = ctx.extras.get("backend_handles", {})
        model_name = handles.get("model_name") or ctx.extras.get("model_name")
        if not model_name:
            raise RuntimeError("model_name not set in backend handles")

        renderer_name = self.cfg.renderer_name or handles.get("renderer_name")
        if not renderer_name:
            raise RuntimeError(
                "renderer_name not set: pass via cfg or backend handles"
            )

        tokenizer = get_tokenizer(model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

        questions = [r["question"] for r in rows]
        golden_answers = [r["golden_answer"] for r in rows]

        provider: Any = SDFTDataset(
            questions=questions,
            golden_answers=golden_answers,
            batch_size=self.cfg.batch_size,
            group_size=self.cfg.group_size,
            renderer=renderer,
            dataset_name=f"evsys_sdk_sdft__{ctx.run_id}",
        )

        n = len(questions)
        steps_per_epoch = max(1, len(provider))
        total_steps = (
            self.cfg.max_steps
            if self.cfg.max_steps is not None
            else steps_per_epoch * self.cfg.num_epochs
        )
        # Wrap the provider when the target step count exceeds one epoch so
        # the cookbook's ``num_batches = min(max_steps, len(provider))`` cap
        # doesn't truncate training. The proxy serves indices modulo the
        # underlying length, giving us multi-epoch training transparently.
        if total_steps > steps_per_epoch:
            provider = _RepeatingSDFTProvider(provider, target_length=total_steps)
        save_every = self._resolve_save_every(total_steps)

        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        log_path = str(out)

        ctx.log_store.log_hyperparams({
            "algorithm": self.name,
            **self.cfg.model_dump(),
            "model_name": model_name,
            "renderer_name": renderer_name,
            "n_train_rows": n,
            "total_steps": total_steps,
            "save_every": save_every,
        })

        config = sdft.Config(
            model_name=model_name,
            renderer_name=renderer_name,
            lora_rank=self.cfg.lora_rank,
            learning_rate=self.cfg.learning_rate,
            topk=self.cfg.topk,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            teacher_sync_every=self.cfg.teacher_sync_every,
            max_context_length=self.cfg.max_context_length,
            demo_template=self.cfg.demo_template,
            system_prompt=self.cfg.system_prompt,
            eval_every=self.cfg.eval_every,
            save_every=save_every,
            max_steps=total_steps,  # honors num_epochs when max_steps unset
            log_path=log_path,
            wandb_project=self.cfg.wandb_project,
            wandb_name=self.cfg.wandb_name,
            load_checkpoint_path=handles.get("load_checkpoint_path"),
        )

        try:
            asyncio.run(sdft.main(config, sdft_dataset=provider))
        except Exception as e:
            logger.exception("TinkerSDFT.train failed")
            return RunResult(run_id=ctx.run_id, status="failed", error=str(e))

        # Harvest the same `checkpoints.jsonl` manifest TinkerSFT writes; the
        # downstream Checkpoint.pick_final + inference factory consume it the
        # same way for both algorithms.
        artifacts: dict[str, str] = {"run_dir": str(out)}
        ckpt_manifest = out / "checkpoints.jsonl"
        if ckpt_manifest.exists():
            for line in ckpt_manifest.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                step_label = str(entry.get("name", "?"))
                state_path = entry.get("state_path") or entry.get("sampler_path")
                if not state_path:
                    continue
                artifacts[f"checkpoint-{step_label}"] = state_path
        for k, v in artifacts.items():
            ctx.log_store.log_artifact(k, v, kind="checkpoint")

        return RunResult(
            run_id=ctx.run_id,
            status="completed",
            metrics={},
            artifacts=artifacts,
        )

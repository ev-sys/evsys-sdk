"""Combo — chain multiple algorithms sequentially in one run.

Use case: ``SFT warmup → GRPO refine``, ``cold-start distill → DAPO``, etc.
Each phase runs against the same RunContext but writes its artifacts to a
``phase{i}_{name}`` subdirectory; the previous phase's ``final_checkpoint``
artifact is threaded into the next phase via ``ctx.extras``.

YAML example::

    kind: combo
    phases:
      - kind: mock_sft
        config:
          num_epochs: 1
      - kind: mock_rl
        config:
          num_iterations: 50
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..protocols import RunContext, RunResult
from ..registry import get_algorithm, register_algorithm

logger = logging.getLogger(__name__)


class ComboPhaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    """Registry name of the sub-algorithm to run (e.g. 'mock_sft', 'rl')."""
    config: dict[str, Any] = Field(default_factory=dict)
    """Per-phase config dict passed to the sub-algorithm's Config constructor."""
    name: str | None = None
    """Optional human-readable phase name; defaults to ``phase{i}_{kind}``."""


class ComboConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phases: list[ComboPhaseConfig] = Field(min_length=1)
    fail_fast: bool = True
    """If True, abort on first phase that returns status != 'completed'."""


@register_algorithm("combo")
class ComboAlgorithm:
    name: ClassVar[str] = "combo"
    Config: ClassVar[type] = ComboConfig

    def __init__(self, **kwargs) -> None:
        self.cfg = ComboConfig.model_validate(kwargs)

    def train(self, ctx: RunContext) -> RunResult:
        out_root = Path(ctx.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        ctx.log_store.log_hyperparams({
            "algorithm": self.name,
            "n_phases":  len(self.cfg.phases),
            "phases":    [p.kind for p in self.cfg.phases],
        })

        last_artifacts: dict[str, str] = {}
        last_metrics: dict[str, float] = {}
        last_result: RunResult | None = None

        for i, phase in enumerate(self.cfg.phases, start=1):
            label = phase.name or f"phase{i}_{phase.kind}"
            phase_dir = out_root / label
            phase_dir.mkdir(exist_ok=True)

            logger.info("combo: starting %s (kind=%s)", label, phase.kind)

            try:
                algo_cls = get_algorithm(phase.kind)
                algo = algo_cls(**phase.config)
            except (KeyError, TypeError, ValueError) as e:
                msg = f"phase {label} could not be constructed: {e}"
                logger.error(msg)
                if self.cfg.fail_fast:
                    return RunResult(
                        run_id=ctx.run_id, status="failed",
                        metrics=last_metrics, artifacts=last_artifacts,
                        error=msg,
                    )
                continue

            sub_ctx = _phase_context(ctx, phase_dir, last_artifacts, label)
            try:
                result = algo.train(sub_ctx)
            except Exception as e:
                msg = f"phase {label} crashed: {e}"
                logger.exception(msg)
                if self.cfg.fail_fast:
                    return RunResult(
                        run_id=ctx.run_id, status="failed",
                        metrics=last_metrics, artifacts=last_artifacts,
                        error=msg,
                    )
                continue

            # Namespace metrics + artifacts by phase to avoid collisions.
            last_metrics.update({f"{label}/{k}": v for k, v in result.metrics.items()})
            last_artifacts.update({f"{label}/{k}": v for k, v in result.artifacts.items()})
            last_result = result

            if result.status != "completed":
                msg = f"phase {label} returned status={result.status} ({result.error})"
                logger.warning(msg)
                if self.cfg.fail_fast:
                    return RunResult(
                        run_id=ctx.run_id, status="failed",
                        metrics=last_metrics, artifacts=last_artifacts,
                        error=msg,
                    )

        return RunResult(
            run_id=ctx.run_id,
            status="completed" if last_result and last_result.status == "completed" else "failed",
            metrics=last_metrics,
            artifacts=last_artifacts,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _phase_context(
    parent: RunContext,
    phase_dir: Path,
    prior_artifacts: dict[str, str],
    label: str,
) -> RunContext:
    """Build a per-phase RunContext that:
        - writes into phase_dir
        - exposes the previous phase's final_checkpoint via ctx.extras
        - shares the parent's data_store, log_store, backend
    """
    new_extras = dict(parent.extras)
    # Thread the most recent final_checkpoint forward as `init_checkpoint`.
    for k, v in reversed(list(prior_artifacts.items())):
        if k.endswith("/final_checkpoint") or k == "final_checkpoint":
            new_extras["init_checkpoint"] = v
            break
    new_extras["combo_phase"] = label

    return replace(
        parent,
        run_id=f"{parent.run_id}/{label}",
        output_dir=str(phase_dir),
        extras=new_extras,
    ) if dataclasses.is_dataclass(parent) else parent  # type: ignore[arg-type]

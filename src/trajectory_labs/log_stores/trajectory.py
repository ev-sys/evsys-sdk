"""TrajectoryLogStore — persist metrics, evals and checkpoints to the backend.

Wraps :class:`trajectory_labs.store.TrajectoryStore` (the backend-routed data
gateway) so a training run's per-step metrics and post-train evals land on the
dashboard against an existing ``run`` row.

Every backend call is best-effort: if the store can't be built (no API key) or a
write fails (e.g. the backend isn't deployed yet), it logs a warning and the
run continues. Compose it under ``multiplex`` to keep local logs too::

    log_store:
      kind: multiplex
      params:
        children:
          - kind: jsonl
            params: { log_dir: ./logs }
          - kind: trajectory
            params: { run_id: "<backend-run-uuid>" }
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import register_log_store

logger = logging.getLogger(__name__)


class TrajectoryLogStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    """Backend run UUID this store writes against (create it via TrajectoryStore)."""
    base_url: str | None = None
    api_key: str | None = None
    model_ref: str | None = None
    """Optional model identifier recorded on persisted eval rows."""
    split: str = "train"


@register_log_store("trajectory")
class TrajectoryLogStore:
    name: ClassVar[str] = "trajectory"
    Config: ClassVar[type] = TrajectoryLogStoreConfig

    def __init__(
        self,
        *,
        run_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        model_ref: str | None = None,
        split: str = "train",
    ) -> None:
        self.run_id = run_id
        self.model_ref = model_ref
        self.split = split
        self._store = None
        try:
            from ..store import TrajectoryStore

            self._store = TrajectoryStore(base_url=base_url, api_key=api_key)
        except Exception as e:  # missing key / import / config
            logger.warning(
                "TrajectoryLogStore offline (run_id=%s): %s — writes will be skipped",
                run_id, e,
            )

    def _try(self, what: str, fn) -> None:
        if self._store is None:
            return
        try:
            fn()
        except Exception as e:
            logger.warning("TrajectoryLogStore %s failed: %s", what, e)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self._try(
            "log_scalar",
            lambda: self._store.log_metric(
                run_id=self.run_id, step=step, name=key, value=value, split=self.split
            ),
        )

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not metrics:
            return
        self._try(
            "log_metrics",
            lambda: self._store.log_metrics(
                run_id=self.run_id, step=step, metrics=dict(metrics), split=self.split
            ),
        )

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        # Run-level config is set when the run row is created; nothing to do here.
        return

    def log_artifact(self, name: str, path: str, *, kind: str = "file") -> None:
        if kind != "checkpoint":
            return
        self._try(
            "log_artifact",
            lambda: self._store.add_checkpoint(
                run_id=self.run_id, uri=path, label=name,
                is_final=(name == "final_checkpoint"),
            ),
        )

    def log_eval(
        self,
        *,
        name: str,
        metrics: dict[str, float],
        step: int | None = None,
        benchmark_id: str | None = None,
        model_ref: str | None = None,
    ) -> None:
        self._try(
            "log_eval",
            lambda: self._store.create_eval(
                run_id=self.run_id,
                benchmark_id=benchmark_id,
                model_ref=model_ref or self.model_ref,
                step=step,
                metrics=dict(metrics),
                breakdowns={"eval_name": name},
            ),
        )

    def close(self) -> None:
        return

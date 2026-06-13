"""Always-on local mirror of experiment data (wandb-offline style).

Every DashboardClient write is also persisted under ``EVSYS_LOG_DIR``
(default ``./evsys_sdk``). This guarantees no data is lost even
when the backend is unreachable, and is the *only* store used in offline mode.

Layout (flat by id, so each call only needs its own id)::

    {log_dir}/
      experiments/{experiment_id}/experiment.json
      generations/{generation_id}/generation.json
                                  metrics.jsonl
                                  evals.jsonl
                                  predictions.jsonl
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_LOG_DIR,
    LOCAL_EVALS_FILE,
    LOCAL_EXPERIMENT_FILE,
    LOCAL_GENERATION_FILE,
    LOCAL_METRICS_FILE,
    LOCAL_PREDICTIONS_FILE,
    EVSYS_LOG_DIR_ENV,
)
from .logger import get_logger

log = get_logger(__name__)


def resolve_log_dir(log_dir: str | None = None) -> Path:
    """Resolve the local mirror directory from arg or EVSYS_LOG_DIR."""
    raw = log_dir or os.environ.get(EVSYS_LOG_DIR_ENV) or DEFAULT_LOG_DIR
    return Path(raw).expanduser()


class LocalExperimentStore:
    """Thread-safe filesystem mirror for experiments and generations."""

    def __init__(self, log_dir: str | None = None) -> None:
        self.root = resolve_log_dir(log_dir)
        self._lock = threading.Lock()

    # -- paths -------------------------------------------------------------

    def _exp_dir(self, experiment_id: str) -> Path:
        return self.root / "experiments" / str(experiment_id)

    def _gen_dir(self, generation_id: str) -> Path:
        return self.root / "generations" / str(generation_id)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(path)

    @staticmethod
    def _merge_json(path: Path, patch: dict[str, Any]) -> None:
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = {}
        existing.update(patch)
        existing["_updated_at"] = time.time()
        LocalExperimentStore._write_json(path, existing)

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    # -- experiments -------------------------------------------------------

    def create_experiment(self, experiment_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            row = {"id": experiment_id, "_created_at": time.time(), **payload}
            self._write_json(self._exp_dir(experiment_id) / LOCAL_EXPERIMENT_FILE, row)
        log.debug("local: wrote experiment %s", experiment_id)

    def update_experiment(self, experiment_id: str, patch: dict[str, Any]) -> None:
        with self._lock:
            self._merge_json(self._exp_dir(experiment_id) / LOCAL_EXPERIMENT_FILE, patch)

    # -- generations -------------------------------------------------------

    def create_run(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            row = {"id": run_id, "_created_at": time.time(), **payload}
            self._write_json(self._gen_dir(run_id) / LOCAL_GENERATION_FILE, row)
        log.debug("local: wrote run %s", run_id)

    def update_run(self, run_id: str, patch: dict[str, Any]) -> None:
        with self._lock:
            self._merge_json(self._gen_dir(run_id) / LOCAL_GENERATION_FILE, patch)

    # -- logs --------------------------------------------------------------

    def log_step(self, generation_id: str, body: dict[str, Any]) -> None:
        with self._lock:
            self._append_jsonl(self._gen_dir(generation_id) / LOCAL_METRICS_FILE, body)

    def log_eval(self, generation_id: str, body: dict[str, Any]) -> None:
        with self._lock:
            self._append_jsonl(self._gen_dir(generation_id) / LOCAL_EVALS_FILE, body)

    def log_predictions(self, generation_id: str, predictions: list[dict[str, Any]]) -> None:
        with self._lock:
            path = self._gen_dir(generation_id) / LOCAL_PREDICTIONS_FILE
            for p in predictions:
                self._append_jsonl(path, p)


__all__ = ["LocalExperimentStore", "resolve_log_dir"]

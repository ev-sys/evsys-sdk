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
    LOCAL_CHECKPOINTS_FILE,
    LOCAL_EVALS_FILE,
    LOCAL_EXPERIMENT_FILE,
    LOCAL_GENERATION_FILE,
    LOCAL_GROUPS_FILE,
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

    def log_group(self, experiment_id: str, body: dict[str, Any]) -> None:
        with self._lock:
            self._append_jsonl(self._exp_dir(experiment_id) / LOCAL_GROUPS_FILE, body)

    def log_checkpoint(self, generation_id: str, body: dict[str, Any]) -> None:
        with self._lock:
            self._append_jsonl(self._gen_dir(generation_id) / LOCAL_CHECKPOINTS_FILE, body)

    # -- reads (parse the mirror back; shapes match EvsysStore) -------------

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue  # skip a malformed row, keep the rest
        return out

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        return self._read_json(self._exp_dir(experiment_id) / LOCAL_EXPERIMENT_FILE)

    def list_experiments(self) -> list[dict[str, Any]]:
        root = self.root / "experiments"
        if not root.is_dir():
            return []
        rows = [self._read_json(d / LOCAL_EXPERIMENT_FILE) for d in root.iterdir() if d.is_dir()]
        return sorted(
            (r for r in rows if r is not None),
            key=lambda r: r.get("_created_at") or 0,
            reverse=True,
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._read_json(self._gen_dir(run_id) / LOCAL_GENERATION_FILE)

    def list_runs(self, *, experiment_id: str | None = None,
                  group_id: str | None = None) -> list[dict[str, Any]]:
        root = self.root / "generations"
        if not root.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for d in root.iterdir():
            if not d.is_dir():
                continue
            r = self._read_json(d / LOCAL_GENERATION_FILE)
            if r is None:
                continue
            if experiment_id is not None and r.get("experiment_id") != experiment_id:
                continue
            if group_id is not None and r.get("group_id") != group_id:
                continue
            out.append(r)
        return sorted(out, key=lambda r: r.get("_created_at") or 0)

    def get_metrics(self, run_id: str, *, name: str | None = None,
                    split: str | None = None) -> list[dict[str, Any]]:
        """Long-format rows ``{step, name, value, split}`` (matches
        ``EvsysStore.get_metrics``), exploded from the nested mirror rows
        ``{step, split, metrics:{name:value}}``."""
        from .step_metrics import _extract_metrics

        out: list[dict[str, Any]] = []
        for row in self._read_jsonl(self._gen_dir(run_id) / LOCAL_METRICS_FILE):
            row_split = row.get("split", "train")
            if split is not None and row_split != split:
                continue
            metrics = _extract_metrics(row) or {}
            step = row.get("step")
            for k, v in metrics.items():
                if name is not None and k != name:
                    continue
                out.append({"step": step, "name": k, "value": v, "split": row_split})
        return out

    def list_evals(self, run_id: str) -> list[dict[str, Any]]:
        return self._read_jsonl(self._gen_dir(run_id) / LOCAL_EVALS_FILE)

    def list_predictions(self, run_id: str, *, kind: str | None = None) -> list[dict[str, Any]]:
        rows = self._read_jsonl(self._gen_dir(run_id) / LOCAL_PREDICTIONS_FILE)
        if kind is not None:
            rows = [r for r in rows if r.get("kind") == kind]
        return rows

    def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        return self._read_jsonl(self._gen_dir(run_id) / LOCAL_CHECKPOINTS_FILE)


class LocalStore:
    """A no-backend store: the same contract ``Experiment`` calls on
    :class:`~evsys_sdk.store.EvsysStore`, but every write lands in the local
    ``.evsys`` mirror (:class:`LocalExperimentStore`) instead of a remote
    backend — and the matching reads parse it back. Swap it in for
    ``EvsysStore`` to run fully offline (no creds) with all data on disk for the
    local UI. Same method names + return shapes, so callers are store-agnostic.
    """

    def __init__(self, log_dir: str | None = None) -> None:
        self.local = LocalExperimentStore(log_dir=log_dir)

    @property
    def root(self) -> Path:
        return self.local.root

    # -- writes (mirror the EvsysStore surface; allocate ids locally) -------

    def create_experiment(self, *, experiment_name: str, project_id: str | None = None,
                          hypothesis: str | None = None, project_goal_id: str | None = None,
                          tags: list[str] | None = None, **extra: Any) -> dict[str, Any]:
        exp_id = _new_id()
        body = {"experiment_name": experiment_name, "project_id": project_id,
                "hypothesis": hypothesis, "project_goal_id": project_goal_id,
                "tags": tags, "status": "running", **extra}
        body = {k: v for k, v in body.items() if v is not None}
        self.local.create_experiment(exp_id, body)
        return {"id": exp_id, **body}

    def update_experiment(self, experiment_id: str, **patch: Any) -> dict[str, Any]:
        self.local.update_experiment(experiment_id, patch)
        return {"id": experiment_id, **patch}

    def create_group(self, experiment_id: str, name: str, *,
                     description: str | None = None) -> dict[str, Any]:
        gid = _new_id()
        body = {"id": gid, "experiment_id": experiment_id, "name": name,
                "description": description}
        body = {k: v for k, v in body.items() if v is not None}
        self.local.log_group(experiment_id, body)
        return body

    def create_run(self, *, experiment_id: str, group_id: str | None = None,
                   dataset_id: str | None = None, seed: int | None = None,
                   recipe_kind: str | None = None, run_config: dict | None = None,
                   status: str = "pending", **extra: Any) -> dict[str, Any]:
        run_id = _new_id()
        body = {"experiment_id": experiment_id, "group_id": group_id,
                "dataset_id": dataset_id, "seed": seed, "recipe_kind": recipe_kind,
                "run_config": run_config, "status": status, **extra}
        body = {k: v for k, v in body.items() if v is not None}
        self.local.create_run(run_id, body)
        return {"id": run_id, **body}

    def update_run(self, run_id: str, **patch: Any) -> dict[str, Any]:
        self.local.update_run(run_id, patch)
        return {"id": run_id, **patch}

    def create_eval(self, *, run_id: str, benchmark_id: str | None = None,
                    checkpoint_id: str | None = None, model_ref: str | None = None,
                    step: int | None = None, metrics: dict | None = None,
                    breakdowns: dict | None = None, sdk_version: str | None = None,
                    **extra: Any) -> dict[str, Any]:
        eval_id = _new_id()
        body = {"id": eval_id, "metrics": dict(metrics or {}), "benchmark_id": benchmark_id,
                "checkpoint_id": checkpoint_id, "model_ref": model_ref, "step": step,
                "breakdowns": breakdowns, "sdk_version": sdk_version, **extra}
        body = {k: v for k, v in body.items() if v is not None}
        self.local.log_eval(run_id, body)
        return body

    def add_prediction(self, *, run_id: str, kind: str, **fields: Any) -> dict[str, Any]:
        row = {"kind": kind, **{k: v for k, v in fields.items() if v is not None}}
        self.local.log_predictions(run_id, [row])
        return {"ok": True}

    def log_predictions(self, run_id: str, predictions: list[dict]) -> dict[str, Any]:
        self.local.log_predictions(run_id, list(predictions))
        return {"ok": True, "inserted": len(predictions)}

    def log_metric(self, *, run_id: str, step: int, name: str, value: float,
                   split: str = "train") -> dict[str, Any]:
        return self.log_metrics(run_id=run_id, step=step, metrics={name: value}, split=split)

    def log_metrics(self, *, run_id: str, step: int, metrics: dict[str, float],
                    split: str = "train") -> dict[str, Any]:
        # Nested body — identical to the existing mirror format so local and
        # dashboard modes store metrics the same way.
        self.local.log_step(run_id, {"step": int(step), "split": split, "metrics": dict(metrics)})
        return {"ok": True}

    def add_checkpoint(self, *, run_id: str, uri: str, label: str | None = None,
                       step: int | None = None, base_model: str | None = None,
                       is_final: bool = False) -> dict[str, Any]:
        body = {"uri": uri, "label": label, "step": step, "base_model": base_model,
                "is_final": is_final}
        body = {k: v for k, v in body.items() if v is not None}
        self.local.log_checkpoint(run_id, body)
        return {"ok": True}

    # -- reads (delegate to the mirror) ------------------------------------

    def get_experiment(self, experiment_id: str) -> dict | None:
        return self.local.get_experiment(experiment_id)

    def list_experiments(self, project_id: str | None = None, **_: Any) -> list[dict]:
        return self.local.list_experiments()

    def get_run(self, run_id: str) -> dict | None:
        return self.local.get_run(run_id)

    def list_runs(self, *, experiment_id: str | None = None,
                  group_id: str | None = None) -> list[dict]:
        return self.local.list_runs(experiment_id=experiment_id, group_id=group_id)

    def get_metrics(self, run_id: str, *, name: str | None = None,
                    split: str | None = None) -> list[dict]:
        return self.local.get_metrics(run_id, name=name, split=split)

    def list_evals(self, run_id: str) -> list[dict]:
        return self.local.list_evals(run_id)

    def list_predictions(self, run_id: str, *, kind: str | None = None) -> list[dict]:
        return self.local.list_predictions(run_id, kind=kind)

    def list_checkpoints(self, run_id: str) -> list[dict]:
        return self.local.list_checkpoints(run_id)


def _new_id() -> str:
    import uuid
    return str(uuid.uuid4())


__all__ = ["LocalExperimentStore", "LocalStore", "resolve_log_dir"]

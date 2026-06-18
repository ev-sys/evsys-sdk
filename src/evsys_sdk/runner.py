"""run_experiment — orchestrates a single ExperimentConfig end-to-end.

Sequence per RunConfig:
  1. Build data store + log store from ExperimentConfig top-level specs.
  2. Read raw rows from data.source.
  3. Apply transforms in order.
  4. Build backend; backend.prepare(model=..., run_dir=...).
  5. Build algorithm from registry + run.algorithm.params; instantiate.
  6. algorithm.train(RunContext) -> RunResult.
  7. backend.teardown(handles).
  8. Run eval if enabled. (Eval is best-effort; a failure here doesn't kill the run.)
  9. Persist run_result.json.

Returns a list[RunResult] (one per run in the experiment).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .config import (
    DataConfig,
    ExperimentConfig,
    RunConfig,
)
from .protocols import RunContext, RunResult
from .registry import (
    get_algorithm,
    get_backend,
    get_data_store,
    get_log_store,
    get_transform,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — build instances from config specs.
# ---------------------------------------------------------------------------


def _build_from_spec(getter, spec) -> Any:
    cls = getter(spec.kind)
    return cls(**(spec.params or {}))


def _load_rows(data: DataConfig, data_store) -> list[dict[str, Any]]:
    # Preferred path: a dashboard dataset referenced by id (or name → latest
    # version's id). The SDK pulls it into the local .evsys/ workspace and
    # trains from that cache, so stored scripts don't depend on local files.
    if data.dataset_id or data.dataset_name:
        from .workspace import Workspace, read_jsonl_rows
        ws = Workspace()
        ds_id = data.dataset_id or ws.dataset_id_for_name(data.dataset_name)  # type: ignore[arg-type]
        mat = ws.pull_dataset(ds_id)
        return read_jsonl_rows(mat.path)
    if data.source_kind == "in_memory":
        return list(data.rows or [])
    if data.source_kind == "jsonl":
        if not data.path:
            raise ValueError("data.path required for source_kind=jsonl")
        return data_store.read_jsonl(data.path)
    if data.source_kind == "json":
        if not data.path:
            raise ValueError("data.path required for source_kind=json")
        v = data_store.read_json(data.path)
        if not isinstance(v, list):
            raise ValueError(f"json source must be a list, got {type(v).__name__}")
        return v
    if data.source_kind == "hf_dataset":
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError("source_kind=hf_dataset needs `datasets` installed") from e
        if not data.hf_dataset:
            raise ValueError("data.hf_dataset required")
        ds = load_dataset(data.hf_dataset, split=data.hf_split)
        return [dict(r) for r in ds]
    raise ValueError(f"Unknown source_kind: {data.source_kind}")


def _apply_transforms(rows: list[dict[str, Any]], data: DataConfig) -> list[dict[str, Any]]:
    for spec in data.transforms:
        cls = get_transform(spec.kind)
        t = cls(**(spec.params or {}))
        rows = list(t(rows))
    return rows


def _persist_result(run_dir: Path, result: RunResult, hparams: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": result.run_id,
        "status": result.status,
        "metrics": result.metrics,
        "artifacts": result.artifacts,
        "error": result.error,
        "hparams": hparams,
        "ts": time.time(),
    }
    (run_dir / "run_result.json").write_text(json.dumps(payload, indent=2, default=str))




# ---------------------------------------------------------------------------
# Per-run orchestration
# ---------------------------------------------------------------------------


def _execute_run(
    *,
    cfg: ExperimentConfig,
    run: RunConfig,
    base_output_dir: Path,
    extra_context: dict[str, Any] | None = None,
) -> RunResult:
    safe_name = run.name.replace("/", "_").replace(" ", "_")
    run_dir = base_output_dir / safe_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build stores. Log store gets log_dir wired in from run_dir if not provided.
    ds_cls = get_data_store(cfg.data_store.kind)
    data_store = ds_cls(**(cfg.data_store.params or {}))

    log_params = dict(cfg.log_store.params or {})
    if cfg.log_store.kind == "multiplex":
        # Auto-fill log_dir for any jsonl/tensorboard child that didn't set one,
        # rooting under <run_dir>/logs/<kind>.
        new_children = []
        for child in log_params.get("children") or []:
            ck = child.get("kind")
            cp = dict(child.get("params") or {})
            if ck in {"jsonl", "tensorboard"} and "log_dir" not in cp:
                cp["log_dir"] = str(run_dir / "logs" / ck)
            new_children.append({"kind": ck, "params": cp})
        log_params["children"] = new_children
    elif cfg.log_store.kind in {"jsonl", "tensorboard"} and "log_dir" not in log_params:
        log_params["log_dir"] = str(run_dir / "logs")
    log_cls = get_log_store(cfg.log_store.kind)
    log_store = log_cls(**log_params)

    # Backend.
    backend = _build_from_spec(get_backend, run.backend)

    # Data.
    raw_rows = _load_rows(run.data, data_store) # TODO : avoid loading entire dataset into memory
    train_rows = _apply_transforms(raw_rows, run.data)

    # Algorithm.
    alg_cls = get_algorithm(run.algorithm.kind)
    algorithm = alg_cls(**(run.algorithm.params or {}))

    try:
        handles = backend.prepare(
            model={
                "name": run.model.name,
                "load_checkpoint_path": run.model.load_checkpoint_path,
                "init_from_checkpoint": run.model.init_from_checkpoint,
                "renderer_name": run.model.renderer_name,
            },
            run_dir=str(run_dir),
        )
    except Exception as e:
        logger.exception("backend.prepare raised")
        result = RunResult(run_id=safe_name, status="failed", error=str(e))
        log_store.close()
        _persist_result(run_dir, result, hparams=run.model_dump())
        return result

    extras: dict[str, Any] = {
        "train_rows": train_rows,
        "n_train_rows": len(train_rows),
        "backend_handles": handles,
        "model_name": run.model.name,
        "agent": run.agent,
        "tags": run.tags,
    }
    # Dashboard plumbing (store + run_id) so in-loop benchmark eval can upload
    # its rollouts; injected by Experiment, absent for bare run_experiment calls.
    if extra_context:
        extras.update(extra_context)

    ctx = RunContext(
        run_id=safe_name,
        output_dir=str(run_dir),
        config=cfg,
        data_store=data_store,
        log_store=log_store,
        backend=backend,
        extras=extras,
    )

    log_store.log_hyperparams(
        {
            "experiment_name": cfg.name,
            "run_name": run.name,
            "model": run.model.model_dump(),
            "backend": run.backend.model_dump(),
            "tags": run.tags,
        }
    )

    try:
        result = algorithm.train(ctx)
    except Exception as e:
        logger.exception("algorithm.train raised")
        result = RunResult(run_id=safe_name, status="failed", error=str(e))
    finally:
        try:
            backend.teardown(handles)
        except Exception:
            logger.exception("backend.teardown raised")

    log_store.close()
    _persist_result(run_dir, result, hparams=run.model_dump())
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_experiment(cfg_or_path, *, extra_context: dict[str, Any] | None = None) -> list[RunResult]:
    """Run an experiment from a parsed config or YAML file.

    ``extra_context`` is merged into each run's ``RunContext.extras`` — used by
    ``Experiment`` to pass the dashboard ``store`` + ``dashboard_run_id`` so
    in-loop validation can upload its rollouts.
    """
    if isinstance(cfg_or_path, ExperimentConfig):
        cfg = cfg_or_path
    else:
        from .yaml_loader import load_yaml
        cfg = load_yaml(cfg_or_path)

    base_output_dir = Path(cfg.output_dir).expanduser()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[RunConfig]
    if cfg.run is not None:
        runs = [cfg.run]
    elif cfg.runs is not None:
        runs = list(cfg.runs)
    else:  # pragma: no cover — caught earlier in config validator
        raise RuntimeError("ExperimentConfig has no runs (matrix not expanded?)")

    results: list[RunResult] = []
    for run in runs:
        results.append(_execute_run(
            cfg=cfg, run=run, base_output_dir=base_output_dir,
            extra_context=extra_context,
        ))
    return results

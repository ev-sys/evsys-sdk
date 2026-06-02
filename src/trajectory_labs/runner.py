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
    get_inference,
    get_log_store,
    get_metric,
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
# Eval (best-effort) — runs after training.
# ---------------------------------------------------------------------------


def _run_eval(run: RunConfig, ctx: RunContext, train_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not run.eval.enabled or not run.eval.metrics:
        return {}

    eval_rows = train_rows
    if run.eval.eval_data is not None:
        eval_rows = _load_rows(run.eval.eval_data, ctx.data_store)
        eval_rows = _apply_transforms(eval_rows, run.eval.eval_data)
    if run.eval.n_samples is not None:
        eval_rows = eval_rows[: run.eval.n_samples]
    if not eval_rows:
        return {}

    inference_spec = run.eval.inference
    if inference_spec is None:
        return {}
    try:
        infer = _build_from_spec(get_inference, inference_spec)
    except Exception as e:
        logger.warning("eval inference build failed: %s", e)
        return {}

    # Retrieval-style clients (e.g. embedding_retrieval) expose retrieve();
    # if present, we capture ranked candidates for retrieval metrics.
    can_retrieve = hasattr(infer, "retrieve")

    predictions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for r in eval_rows:
        prompt = r.get("prompt") or r.get("messages", [{}])[-1].get("content", "")
        # Prefer the raw query for retrieval clients (no chat templating).
        query = r.get("anchor") or r.get("query") or prompt
        try:
            text = infer.generate(prompt=prompt, max_tokens=256, temperature=0.0)
        except Exception as e:
            logger.warning("eval generate failed: %s", e)
            text = ""
        # Extract <answer>X</answer> if present, else use the raw text.
        import re
        m = re.search(r"<answer>\s*([\w]+)\s*</answer>", text)
        ans = m.group(1) if m else text.strip()
        pred: dict[str, Any] = {"answer": ans, "raw": text}
        if can_retrieve:
            try:
                pred["candidates"] = infer.retrieve(query)
            except Exception as e:
                logger.warning("eval retrieve failed: %s", e)
                pred["candidates"] = []
        predictions.append(pred)
        targets.append({
            "answer": r.get("tool_slug", r.get("answer", "")),
            "tool_slug": r.get("tool_slug", r.get("answer", "")),
            "toolkit": r.get("toolkit", ""),
        })

    metrics: dict[str, float] = {}
    for ms in run.eval.metrics:
        try:
            cls = get_metric(ms.kind)
            inst = cls(**(ms.params or {}))
            metrics[f"eval/{ms.kind}"] = inst.compute(predictions=predictions, targets=targets)
        except Exception as e:
            logger.warning("eval metric %s failed: %s", ms.kind, e)
    return metrics


# ---------------------------------------------------------------------------
# Per-run orchestration
# ---------------------------------------------------------------------------


def _execute_run(
    *,
    cfg: ExperimentConfig,
    run: RunConfig,
    base_output_dir: Path,
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
    raw_rows = _load_rows(run.data, data_store)
    train_rows = _apply_transforms(raw_rows, run.data)

    # Algorithm.
    alg_cls = get_algorithm(run.algorithm.kind)
    algorithm = alg_cls(**(run.algorithm.params or {}))

    try:
        handles = backend.prepare(
            model={
                "name": run.model.name,
                "load_checkpoint_path": run.model.load_checkpoint_path,
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

    ctx = RunContext(
        run_id=safe_name,
        output_dir=str(run_dir),
        config=cfg,
        data_store=data_store,
        log_store=log_store,
        backend=backend,
        extras={
            "train_rows": train_rows,
            "n_train_rows": len(train_rows),
            "backend_handles": handles,
            "model_name": run.model.name,
            "tags": run.tags,
        },
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

    # Eval (best-effort).
    if result.status == "completed":
        try:
            extra = _run_eval(run, ctx, train_rows)
            if extra:
                # update result metrics + log
                result.metrics.update(extra)
                log_store.log_metrics(extra, step=int(result.metrics.get("total_steps", 0)) or 1)
        except Exception:
            logger.exception("eval phase raised")

    log_store.close()
    _persist_result(run_dir, result, hparams=run.model_dump())
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_experiment(cfg_or_path) -> list[RunResult]:
    """Run an experiment from a parsed config or YAML file."""
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
        results.append(_execute_run(cfg=cfg, run=run, base_output_dir=base_output_dir))
    return results

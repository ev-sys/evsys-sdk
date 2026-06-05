"""Forward per-step metrics from a local JSONL log to a TrajectoryStore.

Two row shapes are accepted:

  * **Nested** (the SDK's own ``log_store`` output)::

        {"ts": 1700000000.0, "step": 50, "metrics": {"loss": 0.42, "lr": 1e-4}}

  * **Flat** (``tinker_cookbook``'s native ``metrics.jsonl``)::

        {"step": 50, "epoch": 0, "train_mean_nll": 0.42, "learning_rate": 1e-4,
         "time/get_batch": 3e-06, "time/step": 18.1, ...}

The forwarder normalizes both into a ``dict[str, float]`` via
:func:`_extract_metrics` before pushing.

Until now researcher scripts have hand-rolled a forwarder loop after each
training run (see ``composio-bench/training/backfill_step_metrics.py``).
This module gives them one call:

    forward_step_metrics(store, run_id, run_dir)

It locates ``metrics.jsonl`` under ``run_dir`` (preferring ``run_dir/logs/``),
walks each row, and calls ``store.log_metrics(run_id=..., step=..., metrics={...})``.
Per-row store errors are swallowed (logged), so a flaky upload doesn't drop
the rest of the metrics.

Called automatically by ``Experiment._train_arm`` after each arm — researcher
scripts using the OOP path don't have to think about it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_METRICS_FILE = "metrics.jsonl"

# Keys that describe the row itself rather than a metric value. Used to
# filter the flat (tinker_cookbook) shape, where metric keys sit at the
# top level alongside these positional fields.
_FLAT_META_KEYS = frozenset({"step", "epoch", "ts", "timestamp", "progress", "metrics"})


def _extract_metrics(row: dict[str, Any]) -> dict[str, float] | None:
    """Return the metrics dict from a row, normalizing nested + flat shapes.

    Nested rows carry the dict under ``"metrics"``. Flat rows
    (``tinker_cookbook``) have each metric at the top level alongside ``step``;
    we collect every non-meta numeric value and coerce to float. Returns
    ``None`` when no usable metrics can be extracted.
    """
    nested = row.get("metrics")
    if isinstance(nested, dict) and nested:
        out = {k: float(v) for k, v in nested.items() if isinstance(v, (int, float))}
        return out or None
    flat = {
        k: float(v) for k, v in row.items()
        if k not in _FLAT_META_KEYS and isinstance(v, (int, float))
    }
    return flat or None


def forward_step_metrics(
    store: Any | None,
    run_id: str | None,
    run_dir: str | Path | None,
    *,
    metrics_file: str = DEFAULT_METRICS_FILE,
) -> int:
    """Push every row of a ``metrics.jsonl`` to the store. Returns row count.

    Returns ``0`` (silent no-op) if ``store`` is None, ``run_id`` is None,
    ``run_dir`` is missing, or the metrics file isn't found. Malformed JSON
    rows and rows missing ``step``/``metrics`` are skipped; per-row store
    exceptions are caught so one bad upload doesn't drop the rest.
    """
    if store is None or run_id is None or run_dir is None:
        return 0
    path = _locate_metrics_file(Path(run_dir), metrics_file)
    if path is None:
        return 0

    sent = 0
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("step_metrics: skip malformed row at %s:%d", path, lineno)
            continue
        step = row.get("step")
        if step is None:
            continue
        metrics = _extract_metrics(row)
        if metrics is None:
            continue
        # ``val/``-prefixed keys are in-loop validation metrics — forward them
        # under split="val" so the dashboard separates the validation curve from
        # training. Everything else stays on the default (train) split, with the
        # original call signature preserved for back-compat.
        val_metrics = {k: v for k, v in metrics.items() if k.startswith("val/")}
        train_metrics = {k: v for k, v in metrics.items() if not k.startswith("val/")}
        try:
            if train_metrics:
                store.log_metrics(run_id=run_id, step=int(step), metrics=dict(train_metrics))
                sent += 1
            if val_metrics:
                store.log_metrics(
                    run_id=run_id, step=int(step), metrics=dict(val_metrics), split="val"
                )
                sent += 1
        except Exception:
            logger.exception(
                "step_metrics: store.log_metrics failed for run %r step %s",
                run_id, step,
            )
    return sent


def _locate_metrics_file(run_dir: Path, metrics_file: str) -> Path | None:
    """Find ``metrics.jsonl`` under ``run_dir``. Prefer ``<run_dir>/logs/``."""
    if not run_dir.is_dir():
        return None
    direct = run_dir / "logs" / metrics_file
    if direct.is_file():
        return direct
    direct = run_dir / metrics_file
    if direct.is_file():
        return direct
    # Last resort: search recursively, shallowest wins.
    matches = sorted(run_dir.rglob(metrics_file), key=lambda p: len(p.parts))
    return matches[0] if matches else None


__all__ = ["DEFAULT_METRICS_FILE", "forward_step_metrics"]

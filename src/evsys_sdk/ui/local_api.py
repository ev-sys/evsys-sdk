"""Serve the dashboard's API shape from the local mirror on disk.

``LocalStore`` already writes the *same payloads* the hosted dashboard
receives — experiments, runs, long-format metrics, evals, predictions. So the
local UI needs no new data model: it needs these files reshaped into the
contract the existing frontend already consumes, and then every component
(metrics grid, eval comparison, run cards) renders unchanged. One schema, two
deployments, no drift.

The three endpoints the experiment views use:

  * ``experiments()``            → ``GET /api/experiments``
  * ``experiment_detail(id)``    → ``GET /api/experiments/<id>/detail``
  * ``eval_predictions(id, …)``  → ``GET /api/evals/<eval_id>/predictions``

Everything is read fresh per call — a training run is appending to these files
while you look at them, and a stale cache is worse than a slow read.

Runs carry their checkpoints too, now that ``LocalStore`` has a checkpoint
writer — a run that saved none still reports none rather than inventing any.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..constants import (
    LOCAL_CHECKPOINTS_FILE,
    LOCAL_EVALS_FILE,
    LOCAL_EXPERIMENT_FILE,
    LOCAL_GENERATION_FILE,
    LOCAL_METRICS_FILE,
    LOCAL_PREDICTIONS_FILE,
)
from ..local_store import resolve_log_dir
from ..logger import get_logger

log = get_logger(__name__)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict]:
    """Tolerant JSONL read: a run appending to this file right now may leave a
    half-written final line, which must not blank the whole panel."""
    rows: list[dict] = []
    try:
        text = path.read_text()
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _loose_dict(text: str) -> dict | None:
    """`run_config` reaches the mirror as a Python repr (LocalStore stringifies
    on the way out), which is not JSON. Recover it without eval."""
    import ast

    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, dict) else None


def _coerce(value: Any) -> Any:
    """``LocalStore`` stringifies values on the way to disk (``str(v)``), so
    ``seed`` comes back as ``"42"`` and ``run_config`` as a Python repr. Put
    numbers back to numbers; leave anything else alone."""
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


class LocalDashboard:
    """Reader over the ``EVSYS_LOG_DIR`` mirror."""

    def __init__(self, log_dir: str | Path | None = None) -> None:
        self.root = Path(resolve_log_dir(str(log_dir) if log_dir else None))

    # -- paths -------------------------------------------------------------

    @property
    def _experiments_dir(self) -> Path:
        return self.root / "experiments"

    @property
    def _generations_dir(self) -> Path:
        return self.root / "generations"

    # -- GET /api/experiments ---------------------------------------------

    def experiments(self, *, escalation: str | None = None,
                    agent: str | None = None) -> list[dict]:
        """Every experiment in the mirror, newest first.

        ``escalation`` / ``agent`` filter to what one agent run produced —
        the query the autoresearch view is built on ("show me what the agent
        tried for this escalation"). Reads the stamp
        :mod:`evsys_sdk.provenance` writes into ``config.trigger``.
        """
        out: list[dict] = []
        if not self._experiments_dir.is_dir():
            return out
        for d in self._experiments_dir.iterdir():
            rec = _read_json(d / LOCAL_EXPERIMENT_FILE)
            if rec is None:
                continue
            rec.setdefault("id", d.name)
            rec["n_runs"] = len(self._runs_for(d.name))
            # Lift the stamp to the top level so a reader never has to know it
            # rides inside `config`.
            rec["trigger"] = (rec.get("config") or {}).get("trigger")
            trig = rec["trigger"] or {}
            if escalation and trig.get("escalation") != escalation:
                continue
            if agent and trig.get("agent") != agent:
                continue
            out.append(rec)
        out.sort(key=lambda r: str(r.get("_created_at") or ""), reverse=True)
        return out

    def agent_runs(self) -> list[dict]:
        """One row per agent run that produced experiments — the index the
        autoresearch view lists. Empty until agents start stamping (an
        experiment a human launched by hand belongs to no agent run)."""
        by_key: dict[tuple, dict] = {}
        for exp in self.experiments():
            trig = exp.get("trigger")
            if not trig:
                continue
            key = (trig.get("escalation"), trig.get("agent"))
            row = by_key.setdefault(key, {
                "escalation": trig.get("escalation"),
                "agent": trig.get("agent"),
                "sandbox": trig.get("sandbox"),
                "experiments": [],
            })
            row["experiments"].append({
                "id": exp["id"],
                "experiment_name": exp.get("experiment_name"),
                "status": exp.get("status"),
                "best_score": exp.get("best_score"),
                "n_runs": exp.get("n_runs", 0),
            })
        return list(by_key.values())

    # -- GET /api/experiments/<id>/detail ----------------------------------

    def experiment_detail(self, experiment_id: str) -> dict:
        """``{experiment, groups, ungrouped_runs}`` — the frontend's
        ``ExperimentDetail``. Runs carry their metrics, evals and checkpoints
        inline, exactly as the hosted endpoint returns them."""
        exp = _read_json(self._experiments_dir / experiment_id / LOCAL_EXPERIMENT_FILE)
        runs = [self._run_detail(rid) for rid in self._runs_for(experiment_id)]

        groups_raw = _read_jsonl(self._experiments_dir / experiment_id / "groups.jsonl")
        groups: list[dict] = []
        grouped_ids: set[str] = set()
        for g in groups_raw:
            gid = str(g.get("id") or g.get("group_id") or g.get("name") or "")
            members = [r for r in runs if str(r.get("group_id") or "") == gid]
            grouped_ids.update(str(r["id"]) for r in members)
            groups.append({
                "id": gid,
                "name": g.get("name") or gid,
                "description": g.get("description"),
                "runs": members,
            })
        return {
            "experiment": exp,
            "groups": groups,
            "ungrouped_runs": [r for r in runs if str(r["id"]) not in grouped_ids],
        }

    # -- GET /api/evals/<eval_id>/predictions ------------------------------

    def eval_predictions(self, eval_id: str, *, limit: int = 50,
                         offset: int = 0) -> dict:
        """The rollouts recorded under one eval. Because capture is capped at
        the first N per kind, this is a small, complete set — not a page into
        millions."""
        for run_id in self._all_run_ids():
            rows = _read_jsonl(self._gen_dir(run_id) / LOCAL_PREDICTIONS_FILE)
            matching = [r for r in rows if str(r.get("eval_id") or "") == eval_id]
            if not matching:
                continue
            ev = next(
                (e for e in _read_jsonl(self._gen_dir(run_id) / LOCAL_EVALS_FILE)
                 if str(e.get("id") or "") == eval_id),
                {"id": eval_id, "metrics": {}},
            )
            page = matching[offset: offset + limit]
            return {
                "eval": ev,
                "predictions": [self._prediction(p, run_id, i)
                                for i, p in enumerate(page, start=offset)],
                "limit": limit,
                "offset": offset,
                "total": len(matching),
            }
        return {"eval": {"id": eval_id, "metrics": {}}, "predictions": [],
                "limit": limit, "offset": offset, "total": 0}

    def run_data(self, run_id: str) -> dict:
        """Both ends of the data pipeline for one run, plus the transform chain
        that connects them.

        A run's config declares `data.transforms`; what those transforms
        actually did to a row is invisible unless you can see the row before
        and after. Both stages are labelled with their detected format
        (``chat_messages`` for SFT, ``harbor_task`` for RL) so the UI can
        render each one the way it deserves rather than as raw JSON.
        """
        from ..data_types import detect_format

        rec = _read_json(self._gen_dir(run_id) / LOCAL_GENERATION_FILE) or {}
        cfg = rec.get("run_config")
        if isinstance(cfg, str):          # LocalStore stringifies; recover what we can
            cfg = _loose_dict(cfg)
        data_cfg = (cfg or {}).get("data") or {}

        out: dict[str, Any] = {
            "transforms": data_cfg.get("transforms") or [],
            "source": {k: data_cfg.get(k) for k in
                       ("source_kind", "path", "dataset_name", "dataset_id", "hf_dataset")
                       if data_cfg.get(k)},
            "stages": {},
        }
        for kind in ("raw", "train"):
            rows = _read_jsonl(self._gen_dir(run_id) / f"data_{kind}.jsonl")
            out["stages"][kind] = {
                "rows": rows,
                "n": len(rows),
                "format": detect_format(rows[0]) if rows else "unknown",
            }
        return out

    def run_predictions(self, run_id: str, *, kind: str | None = None) -> list[dict]:
        """Every captured rollout for a run, optionally one ``kind``
        (``train`` / ``validation`` / ``eval``)."""
        rows = _read_jsonl(self._gen_dir(run_id) / LOCAL_PREDICTIONS_FILE)
        if kind:
            rows = [r for r in rows if r.get("kind") == kind]
        return [self._prediction(r, run_id, i) for i, r in enumerate(rows)]

    # -- internals ---------------------------------------------------------

    def _gen_dir(self, run_id: str) -> Path:
        return self._generations_dir / run_id

    def _all_run_ids(self) -> list[str]:
        if not self._generations_dir.is_dir():
            return []
        return [d.name for d in self._generations_dir.iterdir() if d.is_dir()]

    def _runs_for(self, experiment_id: str) -> list[str]:
        out = []
        for rid in self._all_run_ids():
            rec = _read_json(self._gen_dir(rid) / LOCAL_GENERATION_FILE) or {}
            if str(rec.get("experiment_id") or "") == experiment_id:
                out.append(rid)
        return sorted(out)

    def _run_detail(self, run_id: str) -> dict:
        rec = _read_json(self._gen_dir(run_id) / LOCAL_GENERATION_FILE) or {}
        rec["id"] = rec.get("id") or run_id
        rec["seed"] = _coerce(rec.get("seed"))
        rec["metrics"] = self._metric_points(run_id)
        rec["evals"] = self._evals(run_id)
        rec["checkpoints"] = self._checkpoints(run_id)
        rec["rollout_counts"] = self._rollout_counts(run_id)
        return rec

    def _metric_points(self, run_id: str) -> list[dict]:
        """``{step, split, metrics:{name: value}}`` on disk → the frontend's
        flat ``{step, split, name, value}`` points."""
        points: list[dict] = []
        for row in _read_jsonl(self._gen_dir(run_id) / LOCAL_METRICS_FILE):
            step = row.get("step")
            split = row.get("split") or "train"
            for name, value in (row.get("metrics") or {}).items():
                try:
                    points.append({"step": int(step), "split": split,
                                   "name": str(name), "value": float(value)})
                except (TypeError, ValueError):
                    continue
        return points

    def _evals(self, run_id: str) -> list[dict]:
        out = []
        for i, row in enumerate(_read_jsonl(self._gen_dir(run_id) / LOCAL_EVALS_FILE)):
            out.append({
                "id": row.get("id") or f"{run_id}-eval-{i}",
                "benchmark_id": row.get("benchmark_id"),
                "checkpoint_id": row.get("checkpoint_id"),
                "step": row.get("step"),
                "metrics": {k: v for k, v in (row.get("metrics") or {}).items()},
                "breakdowns": row.get("breakdowns"),
            })
        return out

    def _checkpoints(self, run_id: str) -> list[dict]:
        out = []
        for i, row in enumerate(_read_jsonl(self._gen_dir(run_id) / LOCAL_CHECKPOINTS_FILE)):
            out.append({
                "id": row.get("id") or f"{run_id}-ckpt-{i}",
                "label": row.get("label"),
                "step": row.get("step"),
                "uri": row.get("uri") or "",
                "is_final": bool(row.get("is_final")),
            })
        return out

    def _rollout_counts(self, run_id: str) -> dict[str, int]:
        """How many rollouts of each kind were captured — so the UI can say
        '20 of 20 kept (capped)' instead of implying it has everything."""
        counts: dict[str, int] = {}
        for row in _read_jsonl(self._gen_dir(run_id) / LOCAL_PREDICTIONS_FILE):
            k = str(row.get("kind") or "eval")
            counts[k] = counts.get(k, 0) + 1
        return counts

    @staticmethod
    def _prediction(row: dict, run_id: str, idx: int) -> dict:
        meta = row.get("metadata") or {}
        return {
            "id": f"{run_id}-{idx}",
            "run_id": run_id,
            "eval_id": row.get("eval_id"),
            "kind": row.get("kind") or "eval",
            "step": row.get("step"),
            "task_id": row.get("task_id"),
            "sample_idx": row.get("sample_idx", 0),
            "instruction": row.get("instruction"),
            # Training rows carry decoded text; eval rows may only have token
            # ids, in which case there is nothing to show and we say so.
            "model_output": row.get("completion") or row.get("model_output"),
            "expected": row.get("expected"),
            "reward": row.get("reward"),
            "metadata": {
                **meta,
                "completion_token_ids": row.get("completion_token_ids") or [],
            },
        }


__all__ = ["LocalDashboard"]

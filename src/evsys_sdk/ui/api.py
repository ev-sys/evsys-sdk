"""JSON shaping for the local UI — pure functions over a store's read methods.

Socket-free so they're unit-testable without booting a server. Each takes any
object exposing the read surface (``LocalStore`` / ``EvsysStore``):
``list_experiments`` / ``list_runs`` / ``get_run`` / ``get_metrics`` /
``list_evals`` / ``list_predictions``.
"""

from __future__ import annotations

from typing import Any


def experiments(store: Any) -> list[dict[str, Any]]:
    """All experiments + a run count (runs grouped by their ``experiment_id``)."""
    runs = store.list_runs()
    n_by_exp: dict[str, int] = {}
    for r in runs:
        eid = r.get("experiment_id")
        if eid is not None:
            n_by_exp[eid] = n_by_exp.get(eid, 0) + 1
    out = []
    for e in store.list_experiments():
        eid = e.get("id")
        out.append({
            "id": eid,
            "name": e.get("experiment_name") or e.get("name") or eid,
            "status": e.get("status"),
            "best_score": e.get("best_score"),
            "n_runs": n_by_exp.get(eid, 0),
            "created_at": e.get("_created_at"),
        })
    return out


def experiment_runs(store: Any, experiment_id: str) -> list[dict[str, Any]]:
    """Runs belonging to one experiment (slim rows for the list view)."""
    out = []
    for r in store.list_runs(experiment_id=experiment_id):
        out.append({
            "id": r.get("id"),
            "recipe_kind": r.get("recipe_kind"),
            "status": r.get("status"),
            "seed": r.get("seed"),
            "group_id": r.get("group_id"),
        })
    return out


def run_detail(store: Any, run_id: str) -> dict[str, Any] | None:
    """The run record + an index of what artifacts it has (so the UI knows
    which sections to render and which metric series exist)."""
    run = store.get_run(run_id)
    if run is None:
        return None
    metric_rows = store.get_metrics(run_id)
    names = sorted({r["name"] for r in metric_rows})
    splits = sorted({r.get("split", "train") for r in metric_rows})
    return {
        "run": run,
        "metric_names": names,
        "splits": splits,
        "has_metrics": bool(metric_rows),
        "has_evals": bool(store.list_evals(run_id)),
        "has_predictions": bool(store.list_predictions(run_id)),
    }


def metrics(store: Any, run_id: str) -> dict[str, Any]:
    """Long-format rows pivoted for charting:
    ``{splits, series: {name: {split: [[step, value], ...]}}}`` (sorted by step)."""
    series: dict[str, dict[str, list[list[float]]]] = {}
    splits: set[str] = set()
    for r in store.get_metrics(run_id):
        name, split = r["name"], r.get("split", "train")
        splits.add(split)
        series.setdefault(name, {}).setdefault(split, []).append([r.get("step"), r.get("value")])
    for by_split in series.values():
        for pts in by_split.values():
            pts.sort(key=lambda p: (p[0] is None, p[0]))
    return {"splits": sorted(splits), "series": series}


def evals(store: Any, run_id: str) -> list[dict[str, Any]]:
    return store.list_evals(run_id)


def predictions(store: Any, run_id: str, *, limit: int = 200, offset: int = 0,
                kind: str | None = None) -> dict[str, Any]:
    """Paginated predictions (these files get large). Drops the bulky
    ``completion_token_ids`` from the list payload."""
    rows = store.list_predictions(run_id, kind=kind) if kind else store.list_predictions(run_id)
    total = len(rows)
    page = rows[offset: offset + limit]
    slim = [{k: v for k, v in r.items() if k != "completion_token_ids"} for r in page]
    return {"total": total, "limit": limit, "offset": offset, "predictions": slim}


__all__ = ["experiments", "experiment_runs", "run_detail", "metrics", "evals", "predictions"]

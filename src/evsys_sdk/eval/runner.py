"""High-level eval runner: loads dataset, runs model eval, computes summary.

Generic, domain-agnostic. Project-specific eval harnesses (e.g. an API search
eval) live in their own repos and reuse this infra (`score_rows`, `AliasMatcher`,
`load_eval_dataset`, …).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocols import InferenceClient
from .matcher import AliasMatcher
from .model_eval import ModelEvalConfig, ModelEvalResult, run_model_eval
from .report import (
    EvalSummary,
    format_summary_markdown,
    model_query_found,
    score_rows,
)


@dataclass
class EvalArtifacts:
    """Final outputs of an eval run."""

    summary: EvalSummary
    summary_strict_primary: EvalSummary | None
    """Optional strict variant (e.g. primary-only). None for plain model evals."""
    per_row_results: list[dict[str, Any]]
    """Raw per-row, per-query results so downstream tooling can re-score."""


def load_eval_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load an eval JSON file. Supports both:
      * list of rows `[{tool_slug, toolkit, queries}, ...]` (v2 shape), or
      * dict with `results: [...]` (older 3-query shape)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "results" in data:
        return [
            {
                "tool_slug": r.get("tool_slug"),
                "toolkit": r.get("toolkit"),
                "queries": [q.get("query") for q in r.get("queries", [])],
            }
            for r in data["results"]
        ]
    return data


def evaluate_model(
    *,
    dataset_path: str | Path,
    aliases_path: str | Path,
    client: InferenceClient,
    secondary_aliases_path: str | Path | None = None,
    config: ModelEvalConfig | None = None,
    output_dir: str | Path | None = None,
    progress: bool = True,
) -> EvalArtifacts:
    rows = load_eval_dataset(dataset_path)
    matcher = AliasMatcher.from_files(aliases_path, secondary_aliases_path)
    result: ModelEvalResult = run_model_eval(
        rows, client=client, config=config, progress=progress
    )

    summary = score_rows(
        result.rows,
        matcher=matcher,
        query_found_fn=model_query_found,
        retry_report=result.retry_report,
    )
    artifacts = EvalArtifacts(
        summary=summary,
        summary_strict_primary=None,
        per_row_results=result.rows,
    )

    if output_dir is not None:
        _write_artifacts(artifacts, output_dir, kind="model")
    return artifacts


def _write_artifacts(artifacts: EvalArtifacts, output_dir: str | Path, *, kind: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / f"{kind}_summary.json"
    summary_path.write_text(json.dumps(artifacts.summary.as_dict(), indent=2))

    md_path = out / f"{kind}_summary.md"
    md_path.write_text(format_summary_markdown(artifacts.summary, title=f"{kind} eval"))

    rows_path = out / f"{kind}_per_row.json"
    rows_path.write_text(json.dumps(artifacts.per_row_results, indent=2))

    if artifacts.summary_strict_primary:
        strict_path = out / f"{kind}_summary_strict_primary.json"
        strict_path.write_text(json.dumps(artifacts.summary_strict_primary.as_dict(), indent=2))

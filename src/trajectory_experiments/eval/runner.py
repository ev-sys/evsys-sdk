"""High-level eval runner: loads dataset, runs eval, computes summary.

Single entry point that ties matcher + retry + composio/model eval + report
together. Used by both the CLI and the existing ``run_experiment`` pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocols import InferenceClient
from .composio_search import (
    ComposioSearchConfig,
    ComposioSearchEvalResult,
    run_composio_search_eval,
)
from .matcher import AliasMatcher
from .model_eval import ModelEvalConfig, ModelEvalResult, run_model_eval
from .report import (
    EvalSummary,
    composio_query_found,
    composio_query_found_primary,
    format_summary_markdown,
    model_query_found,
    score_rows,
)


@dataclass
class EvalArtifacts:
    """Final outputs of an eval run."""

    summary: EvalSummary
    summary_strict_primary: EvalSummary | None
    """For Composio search: primary-only variant. None for model evals."""
    per_row_results: list[dict[str, Any]]
    """Raw per-row, per-query results (preserves the ``primary``/``related``
    or model ``completion`` data so downstream tooling can re-score."""


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


def evaluate_composio_search(
    *,
    dataset_path: str | Path,
    aliases_path: str | Path,
    secondary_aliases_path: str | Path | None = None,
    config: ComposioSearchConfig | None = None,
    output_dir: str | Path | None = None,
    progress: bool = True,
) -> EvalArtifacts:
    rows = load_eval_dataset(dataset_path)
    matcher = AliasMatcher.from_files(aliases_path, secondary_aliases_path)
    result: ComposioSearchEvalResult = run_composio_search_eval(
        rows, config=config, progress=progress
    )

    summary = score_rows(
        result.rows,
        matcher=matcher,
        query_found_fn=composio_query_found,
        retry_report=result.retry_report,
    )
    summary_strict = score_rows(
        result.rows,
        matcher=matcher,
        query_found_fn=composio_query_found_primary,
        retry_report=result.retry_report,
    )

    artifacts = EvalArtifacts(
        summary=summary,
        summary_strict_primary=summary_strict,
        per_row_results=result.rows,
    )

    if output_dir is not None:
        _write_artifacts(artifacts, output_dir, kind="composio_search")
    return artifacts


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

"""Composio meta-search eval — drives COMPOSIO_SEARCH_TOOLS over a dataset.

Wraps every API call in a 5-retry exponential backoff. Connection failures
are collected into a RetryReport and surfaced in the final eval result rather
than aborting the run.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from .retry import RetryReport, call_with_retry


@dataclass
class ComposioSearchConfig:
    api_key: str | None = None
    api_key_env: str = "COMPOSIO_API_KEY"
    user_id: str = "trajectory-eval-user"
    max_attempts: int = 5
    threads: int = 4
    toolkits_per_query: list[str] | None = None
    """If set, restrict the search to this set of toolkits (Composio's
    `toolkits` arg). None → search all enabled toolkits."""


def _strip_env_key(raw: str) -> str:
    # .env loaders sometimes leave \r on Windows-edited files; strip it.
    return raw.strip().strip("'\"").rstrip("\r\n")


def _build_client(cfg: ComposioSearchConfig) -> Any:
    from composio import Composio

    key = cfg.api_key or os.environ.get(cfg.api_key_env, "")
    key = _strip_env_key(key)
    if not key:
        raise RuntimeError(
            f"Composio API key missing: set {cfg.api_key_env} or pass api_key in config"
        )
    return Composio(api_key=key)


def _search_one(client: Any, query: str, cfg: ComposioSearchConfig) -> dict[str, Any]:
    """Execute COMPOSIO_SEARCH_TOOLS for a single query. Raises on failure."""
    args: dict[str, Any] = {"use_case": query}
    if cfg.toolkits_per_query:
        args["toolkits"] = cfg.toolkits_per_query
    resp = client.tools.execute(
        "COMPOSIO_SEARCH_TOOLS",
        arguments=args,
        user_id=cfg.user_id,
        dangerously_skip_version_check=True,
    )
    return resp


def _extract_slugs(resp: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Pull ``primary`` + ``related`` slug lists from a search response.

    Supports both the legacy flat shape (``data.primary`` / ``data.related``)
    and the newer plan-style shape (``data.results[*].primary_tool_slugs`` /
    ``data.results[*].related_tool_slugs``)."""
    data = resp.get("data", resp) if isinstance(resp, dict) else {}
    primary: list[str] = []
    related: list[str] = []

    # New plan-style shape (Composio meta-search >= 2026).
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        for result in data["results"]:
            if not isinstance(result, dict):
                continue
            for key, bucket in (
                ("primary_tool_slugs", primary),
                ("related_tool_slugs", related),
            ):
                val = result.get(key)
                if isinstance(val, list):
                    for t in val:
                        slug = t.get("slug") if isinstance(t, dict) else str(t)
                        if slug:
                            bucket.append(slug)

    # Legacy flat shape.
    if not primary:
        for key in ("primary", "primary_tools", "tools"):
            val = data.get(key) if isinstance(data, dict) else None
            if isinstance(val, list):
                primary = [t.get("slug") if isinstance(t, dict) else str(t) for t in val]
                break
    if not related:
        for key in ("related", "related_tools", "alternative_tools"):
            val = data.get(key) if isinstance(data, dict) else None
            if isinstance(val, list):
                related = [t.get("slug") if isinstance(t, dict) else str(t) for t in val]
                break
    return [s for s in primary if s], [s for s in related if s]


@dataclass
class ComposioSearchEvalResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    """Per-query rows: {tool_slug, toolkit, query, primary, related, error}."""
    retry_report: RetryReport = field(default_factory=RetryReport)


def run_composio_search_eval(
    eval_rows: list[dict[str, Any]],
    *,
    config: ComposioSearchConfig | None = None,
    progress: bool = True,
) -> ComposioSearchEvalResult:
    """Run COMPOSIO_SEARCH_TOOLS over ``eval_rows``.

    Each row in ``eval_rows`` is expected to have keys
    ``{tool_slug, toolkit, queries: [q1, q2, q3]}`` (the v2 eval shape).
    The output preserves per-query granularity so pass@k can be computed
    downstream.
    """
    cfg = config or ComposioSearchConfig()
    client = _build_client(cfg)
    report = RetryReport()

    # Flatten to (row_idx, query_idx, query) so we can multi-thread per-query.
    jobs: list[tuple[int, int, str]] = []
    for ri, row in enumerate(eval_rows):
        for qi, q in enumerate(row.get("queries", [])):
            jobs.append((ri, qi, q))

    per_query: dict[tuple[int, int], dict[str, Any]] = {}

    def _do_one(job: tuple[int, int, str]) -> tuple[tuple[int, int], dict[str, Any]]:
        ri, qi, q = job
        ctx = f"composio_search:row{ri}:q{qi}"
        resp = call_with_retry(
            _search_one,
            client,
            q,
            cfg,
            max_attempts=cfg.max_attempts,
            report=report,
            context=ctx,
        )
        if resp is None:
            return (ri, qi), {
                "query": q,
                "primary": [],
                "related": [],
                "error": "retry_exhausted",
            }
        primary, related = _extract_slugs(resp)
        return (ri, qi), {"query": q, "primary": primary, "related": related, "error": None}

    with ThreadPoolExecutor(max_workers=cfg.threads) as ex:
        futures = [ex.submit(_do_one, j) for j in jobs]
        done = 0
        for fut in as_completed(futures):
            key, value = fut.result()
            per_query[key] = value
            done += 1
            if progress and done % 50 == 0:
                print(f"  composio search: {done}/{len(jobs)} queries done")

    out_rows: list[dict[str, Any]] = []
    for ri, row in enumerate(eval_rows):
        qs: list[dict[str, Any]] = []
        for qi, q in enumerate(row.get("queries", [])):
            qs.append(per_query.get((ri, qi), {"query": q, "primary": [], "related": [], "error": "missing"}))
        out_rows.append(
            {
                "tool_slug": row.get("tool_slug"),
                "toolkit": row.get("toolkit"),
                "queries": qs,
            }
        )

    return ComposioSearchEvalResult(rows=out_rows, retry_report=report)

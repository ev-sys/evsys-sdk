"""Pass@1/pass@3/pass^3 aggregation + retry-failure section.

Works on any per-row, per-query result shape: the caller supplies a function
that maps one query-result to a bool ``found``, so the same scorer serves
model evals and any custom eval harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .matcher import AliasMatcher
from .retry import RetryReport


def model_query_found(qresult: dict[str, Any], expected_slug: str, matcher: AliasMatcher) -> bool:
    if qresult.get("error"):
        return False
    return matcher.matches(expected_slug, qresult.get("predicted", ""))


@dataclass
class EvalSummary:
    pass_at_1: float
    pass_at_3: float
    pass_pow_3: float
    n_rows: int
    n_queries: int
    per_toolkit: dict[str, dict[str, float]] = field(default_factory=dict)
    retry_report: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall": {
                "pass_at_1": round(self.pass_at_1, 4),
                "pass_at_3": round(self.pass_at_3, 4),
                "pass_pow_3": round(self.pass_pow_3, 4),
            },
            "n_rows": self.n_rows,
            "n_queries": self.n_queries,
            "per_toolkit": self.per_toolkit,
            "retry_report": self.retry_report,
            "notes": self.notes,
        }


def score_rows(
    rows: list[dict[str, Any]],
    *,
    matcher: AliasMatcher,
    query_found_fn: Callable[[dict[str, Any], str, AliasMatcher], bool],
    retry_report: RetryReport | None = None,
) -> EvalSummary:
    if not rows:
        return EvalSummary(0.0, 0.0, 0.0, 0, 0)

    per_toolkit_buckets: dict[str, dict[str, list[float]]] = {}
    sums = {"pass_at_1": 0.0, "pass_at_3": 0.0, "pass_pow_3": 0.0}
    n_queries = 0

    for row in rows:
        expected = row.get("tool_slug", "")
        tk = row.get("toolkit", "")
        qs = row.get("queries", [])
        flags = [query_found_fn(q, expected, matcher) for q in qs]
        n_queries += len(qs)

        if not flags:
            continue
        p1 = 1.0 if flags[0] else 0.0
        p3 = 1.0 if any(flags) else 0.0
        pp3 = 1.0 if all(flags) else 0.0

        sums["pass_at_1"] += p1
        sums["pass_at_3"] += p3
        sums["pass_pow_3"] += pp3

        bucket = per_toolkit_buckets.setdefault(
            tk, {"pass_at_1": [], "pass_at_3": [], "pass_pow_3": []}
        )
        bucket["pass_at_1"].append(p1)
        bucket["pass_at_3"].append(p3)
        bucket["pass_pow_3"].append(pp3)

    n = len(rows)
    per_toolkit: dict[str, dict[str, float]] = {}
    for tk, b in per_toolkit_buckets.items():
        cnt = len(b["pass_at_1"])
        per_toolkit[tk] = {
            "count": cnt,
            "pass_at_1": round(sum(b["pass_at_1"]) / cnt, 4),
            "pass_at_3": round(sum(b["pass_at_3"]) / cnt, 4),
            "pass_pow_3": round(sum(b["pass_pow_3"]) / cnt, 4),
        }

    summary = EvalSummary(
        pass_at_1=sums["pass_at_1"] / n,
        pass_at_3=sums["pass_at_3"] / n,
        pass_pow_3=sums["pass_pow_3"] / n,
        n_rows=n,
        n_queries=n_queries,
        per_toolkit=per_toolkit,
        retry_report=retry_report.as_dict() if retry_report else {"total_failures": 0, "failures": []},
    )
    return summary


def format_summary_markdown(summary: EvalSummary, *, title: str = "Eval Summary") -> str:
    o = summary.as_dict()
    lines = [
        f"# {title}",
        "",
        f"- Rows: **{summary.n_rows}**, queries: **{summary.n_queries}**",
        f"- pass@1: **{o['overall']['pass_at_1']:.1%}**",
        f"- pass@3: **{o['overall']['pass_at_3']:.1%}**",
        f"- pass^3: **{o['overall']['pass_pow_3']:.1%}**",
    ]
    rr = o.get("retry_report", {})
    if rr.get("total_failures"):
        lines += [
            "",
            f"## Retry-exhausted failures: {rr['total_failures']}",
            "",
            "| Exception | Count |",
            "|---|---|",
        ]
        for k, v in (rr.get("by_exception_type") or {}).items():
            lines.append(f"| `{k}` | {v} |")
    if summary.per_toolkit:
        lines += ["", "## Per toolkit", "", "| Toolkit | N | pass@1 | pass@3 | pass^3 |", "|---|---|---|---|---|"]
        for tk in sorted(summary.per_toolkit):
            m = summary.per_toolkit[tk]
            lines.append(
                f"| {tk} | {m['count']} | {m['pass_at_1']:.1%} | {m['pass_at_3']:.1%} | {m['pass_pow_3']:.1%} |"
            )
    if summary.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in summary.notes]
    return "\n".join(lines) + "\n"

"""RunLog — per-run, two-track local logging (human + agent).

One run writes two surfaces under its run dir:

* ``human/`` — clean, decoded-to-text, only the critical milestones and a few
  representative samples per phase. A researcher opens ``human/summary.md`` and
  sees how the run is going.
* ``agent/`` — dense, machine-shaped. For rollouts this is **not a copy**: harbor
  already persists every trial's ``result.json`` (token ids, reward, usage) under
  its ``jobs_dir``, so we simply route harbor's ``jobs_dir`` *into* ``agent/harbor/``
  and reference it. The human view is rendered from the already-harvested
  in-memory :class:`~evsys_sdk.training.trajectory.TrajectoryGroup`\\s, so nothing
  harbor stores is duplicated.

Layout::

    {run_dir}/
      human/
        summary.md
        01_data/            datasets.md, sample.jsonl
        02_rollouts/        rollouts.md            # curated; points at agent/harbor
        03_target_tokens/   supervised_examples.md
        04_training/        metrics.csv, checkpoints.md
        05_benchmark/       results.md, predictions.md
      agent/
        harbor/{phase}/     # harbor's own jobs_dir (rollout store) — referenced, not copied

The module is harbor-free and dependency-light: a ``tokenizer`` is duck-typed
(``.decode(ids)``) and optional, so every renderer degrades gracefully.
"""

from __future__ import annotations

import csv
import io
import json
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .logger import get_logger

if TYPE_CHECKING:  # avoid importing the (tinker-heavy) training package at runtime
    from .training.trajectory import TrajectoryGroup

log = get_logger(__name__)

# Per-step training scalars a human cares about (substring match, case-insensitive).
_METRIC_WHITELIST = (
    "loss", "nll", "reward/mean", "reward_mean", "lr", "learning_rate",
    "accuracy", "pass_rate", "kl",
)
_PREVIEW_ROWS = 5
_ROLLOUT_SAMPLES = 3
_SNIPPET_CHARS = 600


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))


def _decode(turn: Any, tokenizer: Any) -> str:
    """Best-effort decoded text for one turn's completion."""
    text = getattr(turn, "text", "") or ""
    if text:
        return text
    ids = list(getattr(turn, "completion_tokens", None) or [])
    if ids and tokenizer is not None:
        try:
            return str(tokenizer.decode(ids))
        except Exception:  # pragma: no cover - defensive
            pass
    return f"<{len(ids)} completion tokens>"


def _completion_text(traj: Any, tokenizer: Any) -> str:
    turns = getattr(traj, "turns", None) or []
    if not turns:
        return "<no turns>"
    return _decode(turns[-1], tokenizer)


def _clip(s: str, n: int = _SNIPPET_CHARS) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n].rstrip() + " …"


class RunLog:
    """Two-track logger for one run. All writes are best-effort and never raise
    into the training loop (a logging failure must not fail a run)."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        experiment_name: str | None = None,
        run_name: str | None = None,
        hypothesis: str | None = None,
    ) -> None:
        self.root = Path(run_dir).expanduser()
        self.human = self.root / "human"
        self.agent = self.root / "agent"
        self._meta = {
            "experiment_name": experiment_name,
            "run_name": run_name,
            "hypothesis": hypothesis,
        }
        self._evals: list[dict[str, Any]] = []
        for d in (self.human, self.agent):
            d.mkdir(parents=True, exist_ok=True)
        for sub in ("01_data", "02_rollouts", "03_target_tokens", "04_training", "05_benchmark"):
            (self.human / sub).mkdir(exist_ok=True)

    # -- harbor: route its jobs_dir here; reference, don't copy ---------------

    def harbor_dir(self, phase: str) -> Path:
        """Directory a caller passes as ``run_harbor_rollouts(workspace_dir=...)``.

        Harbor writes its task dirs + ``jobs/`` (the full rollout store) under
        here, so the dense rollout data lives in the agent track natively — no
        second copy. ``phase`` separates train / val / eval / sdft rollouts."""
        d = self.agent / "harbor" / _safe(phase)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- 01_data -------------------------------------------------------------

    def log_data(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        dataset_meta: dict[str, Any] | None = None,
        transforms: Sequence[Any] | None = None,
        n_preview: int = _PREVIEW_ROWS,
    ) -> None:
        try:
            self._log_data(rows, dataset_meta, transforms, n_preview)
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.log_data failed", exc_info=True)

    def _log_data(self, rows, dataset_meta, transforms, n_preview) -> None:
        d = self.human / "01_data"
        meta = dataset_meta or {}
        lines = ["# Data going in", ""]
        lines.append(f"- rows: **{len(rows)}**")
        for key in ("name", "version", "format", "source_kind"):
            if meta.get(key) is not None:
                lines.append(f"- {key}: `{meta[key]}`")
        tfs = list(transforms or [])
        if tfs:
            names = [getattr(t, "kind", None) or getattr(t, "name", None) or str(t) for t in tfs]
            lines.append(f"- transforms (in order): {', '.join(f'`{n}`' for n in names)}")
        lines.append("")
        lines.append(f"First {min(n_preview, len(rows))} rendered rows → `sample.jsonl`.")
        (d / "datasets.md").write_text("\n".join(lines) + "\n")
        with (d / "sample.jsonl").open("w") as f:
            for row in list(rows)[:n_preview]:
                f.write(json.dumps(row, default=str) + "\n")

    # -- 02_rollouts (curated view; full data lives in agent/harbor) ----------

    def note_rollouts(
        self,
        phase: str,
        groups: Sequence[TrajectoryGroup],
        *,
        tokenizer: Any = None,
        step: int | None = None,
        tasks: Sequence[Any] | None = None,
        k: int = _ROLLOUT_SAMPLES,
    ) -> None:
        """Append a curated section to ``human/02_rollouts/rollouts.md``: the
        reward distribution + the best / median / worst decoded completions.

        The full per-trial rollouts (token ids, every sample) are harbor's own
        ``result.json`` files under ``agent/harbor/{phase}`` — referenced here,
        never copied."""
        try:
            self._note_rollouts(phase, groups, tokenizer, step, tasks, k)
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.note_rollouts failed", exc_info=True)

    def _note_rollouts(self, phase, groups, tokenizer, step, tasks, k) -> None:
        # Flatten to (group_idx, trajectory) and collect rewards.
        flat: list[tuple[int, Any]] = [
            (gi, t) for gi, g in enumerate(groups) for t in getattr(g, "trajectories", [])
        ]
        rewards = [float(getattr(t, "reward", 0.0)) for _, t in flat]
        header = f"## {phase}" + (f" — step {step}" if step is not None else "")
        out = [header, ""]
        if rewards:
            out.append(
                f"- rollouts: **{len(rewards)}** across {len(groups)} task(s) · "
                f"reward mean **{statistics.fmean(rewards):.3f}** "
                f"(min {min(rewards):.3f}, max {max(rewards):.3f})"
            )
        else:
            out.append(f"- rollouts: 0 across {len(groups)} task(s)")
        out.append(f"- full rollouts (token ids, all samples): `agent/harbor/{_safe(phase)}/`")
        out.append("")

        # Pick best / median / worst by reward (deduped for tiny sets).
        if flat:
            order = sorted(range(len(flat)), key=lambda i: rewards[i])
            picks: list[tuple[str, int]] = []
            for label, idx in (("worst", order[0]), ("median", order[len(order) // 2]),
                               ("best", order[-1])):
                if idx not in [p for _, p in picks]:
                    picks.append((label, idx))
            for label, idx in picks[:k]:
                gi, traj = flat[idx]
                out.append(f"**{label}** · reward {rewards[idx]:.3f}")
                if tasks is not None and gi < len(tasks):
                    instr = getattr(tasks[gi], "instruction", None)
                    if instr:
                        out.append(f"> task: {_clip(str(instr), 200)}")
                out.append("```")
                out.append(_clip(_completion_text(traj, tokenizer)))
                out.append("```")
                out.append("")

        path = self.human / "02_rollouts" / "rollouts.md"
        prefix = "" if path.exists() else "# Rollouts\n\n"
        with path.open("a") as f:
            f.write(prefix + "\n".join(out) + "\n")

    # -- 03_target_tokens ----------------------------------------------------

    def log_target_tokens(self, examples: Sequence[dict[str, Any]]) -> None:
        """Record a few supervised-span examples (the tokens loss is computed on).

        Each example: ``{"text": str, "supervised": str, "n_supervised": int,
        "n_total": int, "topk"?: list}``. Caller decides how to render the mask;
        we keep it to a handful for the human view."""
        try:
            self._log_target_tokens(examples)
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.log_target_tokens failed", exc_info=True)

    def _log_target_tokens(self, examples) -> None:
        out = ["# Loss / supervised (target) tokens", "",
               "The spans below are where loss is computed (what the model is "
               "trained to produce).", ""]
        for i, ex in enumerate(list(examples)[:_PREVIEW_ROWS]):
            n_sup, n_tot = ex.get("n_supervised"), ex.get("n_total")
            counts = f" — supervised {n_sup}/{n_tot} tokens" if n_sup is not None else ""
            out.append(f"### example {i}{counts}")
            if ex.get("text"):
                out.append("context:")
                out.append("```")
                out.append(_clip(str(ex["text"])))
                out.append("```")
            if ex.get("supervised"):
                out.append("supervised span:")
                out.append("```")
                out.append(_clip(str(ex["supervised"])))
                out.append("```")
            if ex.get("topk"):
                out.append(f"teacher top-K (first positions): `{ex['topk']}`")
            out.append("")
        (self.human / "03_target_tokens" / "supervised_examples.md").write_text(
            "\n".join(out) + "\n"
        )

    # -- 04_training ---------------------------------------------------------

    def render_training(self, *, checkpoints: Sequence[dict[str, Any]] | None = None) -> None:
        """Render ``metrics.csv`` (whitelisted scalars) from the run's
        ``metrics.jsonl`` and a ``checkpoints.md`` list."""
        try:
            self._render_training(checkpoints)
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.render_training failed", exc_info=True)

    def _find_metrics_jsonl(self) -> Path | None:
        for cand in (
            self.root / "logs" / "metrics.jsonl",
            self.root / "logs" / "jsonl" / "metrics.jsonl",
        ):
            if cand.exists():
                return cand
        hits = list(self.root.glob("logs/**/metrics.jsonl"))
        return hits[0] if hits else None

    def _render_training(self, checkpoints) -> None:
        path = self._find_metrics_jsonl()
        if path is not None:
            rows = []
            keys: list[str] = []
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metrics = rec.get("metrics", rec)
                kept = {
                    k: v for k, v in metrics.items()
                    if isinstance(v, (int, float))
                    and any(w in k.lower() for w in _METRIC_WHITELIST)
                }
                if not kept:
                    continue
                row = {"step": rec.get("step")}
                row.update(kept)
                rows.append(row)
                for k in row:
                    if k not in keys:
                        keys.append(k)
            if rows:
                buf = io.StringIO()
                w = csv.DictWriter(buf, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
                (self.human / "04_training" / "metrics.csv").write_text(buf.getvalue())
        if checkpoints:
            lines = ["# Checkpoints", ""]
            for c in checkpoints:
                tag = " (final)" if c.get("is_final") else ""
                lines.append(f"- step {c.get('step', '?')} · `{c.get('label') or c.get('uri')}`{tag}")
            (self.human / "04_training" / "checkpoints.md").write_text("\n".join(lines) + "\n")

    # -- 05_benchmark --------------------------------------------------------

    def log_eval(
        self,
        name: str,
        metrics: dict[str, float],
        *,
        predictions: Sequence[dict[str, Any]] | None = None,
        breakdowns: dict[str, Any] | None = None,
    ) -> None:
        """Render a benchmark's metrics table + a failure-biased sample of
        predictions, and remember the headline for ``summary.md``."""
        try:
            self._log_eval(name, metrics, predictions, breakdowns)
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.log_eval failed", exc_info=True)

    def _log_eval(self, name, metrics, predictions, breakdowns) -> None:
        self._evals.append({"name": name, "metrics": dict(metrics)})
        d = self.human / "05_benchmark"
        # results.md (append per benchmark).
        res = [f"## {name}", "", "| metric | value |", "| --- | --- |"]
        for k, v in metrics.items():
            res.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
        res.append("")
        for field, buckets in (breakdowns or {}).items():
            if not isinstance(buckets, dict):
                continue
            res.append(f"by `{field}`:")
            for value, stats in buckets.items():
                mr = stats.get("mean_reward") if isinstance(stats, dict) else stats
                res.append(f"- {value}: {mr}")
            res.append("")
        rp = d / "results.md"
        with rp.open("a") as f:
            f.write(("" if rp.exists() else "# Benchmark results\n\n") + "\n".join(res) + "\n")
        # predictions.md — failures first, then a couple passes.
        if predictions:
            preds = sorted(predictions, key=lambda p: float(p.get("reward") or 0.0))
            sample = preds[:3] + [p for p in preds if float(p.get("reward") or 0.0) > 0][:2]
            pl = [f"## {name} — sample predictions", ""]
            seen = set()
            for p in sample:
                key = (p.get("task_id"), p.get("sample_idx"))
                if key in seen:
                    continue
                seen.add(key)
                pl.append(f"**task `{p.get('task_id')}`** · reward {float(p.get('reward') or 0.0):.3f}")
                if p.get("instruction"):
                    pl.append(f"> {_clip(str(p['instruction']), 200)}")
                if p.get("expected") is not None:
                    pl.append(f"- expected: `{_clip(str(p['expected']), 200)}`")
                pl.append("")
            pp = d / "predictions.md"
            with pp.open("a") as f:
                f.write(("" if pp.exists() else "# Sample predictions\n\n") + "\n".join(pl) + "\n")

    # -- summary -------------------------------------------------------------

    def write_summary(
        self,
        *,
        status: str | None = None,
        best_metric: str | None = None,
        best_value: float | None = None,
        conclusion: str | None = None,
    ) -> None:
        try:
            self._write_summary(status, best_metric, best_value, conclusion)
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.write_summary failed", exc_info=True)

    def _write_summary(self, status, best_metric, best_value, conclusion) -> None:
        m = self._meta
        out = [f"# {m.get('experiment_name') or 'experiment'} — run `{m.get('run_name') or ''}`", ""]
        if m.get("hypothesis"):
            out.append(f"**Hypothesis:** {m['hypothesis']}")
            out.append("")
        if status:
            out.append(f"- status: **{status}**")
        if best_metric and best_value is not None:
            out.append(f"- {best_metric}: **{best_value:.4f}**")
        out.append("")
        if self._evals:
            out.append("## Benchmarks")
            out.append("")
            for ev in self._evals:
                head = ", ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in list(ev["metrics"].items())[:4]
                )
                out.append(f"- **{ev['name']}** — {head}")
            out.append("")
        if conclusion:
            out.append("## Conclusion")
            out.append("")
            out.append(conclusion)
            out.append("")
        out.append("---")
        out.append("Full data: `agent/harbor/` (rollouts), `04_training/metrics.csv`, "
                   "`05_benchmark/`.")
        (self.human / "summary.md").write_text("\n".join(out) + "\n")


__all__ = ["RunLog"]

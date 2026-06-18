"""RunLog — one clean, human-readable log per run, organized into folders.

There is a single log (no separate "agent" dump). The SDK writes a fixed set of
ordered folders under each run dir, and **user code** (custom ``transforms``,
custom ``build_batch`` / algorithms, callbacks) can grab the same logger and
write into its own named folder.

SDK default folders (only the essentials, decoded to text — never token ids)::

    {run_dir}/
      01_data/              data after transforms + the exact chat template going in
      02_training_rollouts/ training rollout predictions + per-token logprobs +
                            reward & advantage per sample per trajectory group
      03_validation_rollouts/ validation rollout predictions
      04_training_metrics/  training metrics (metrics.csv)
      05_validation_metrics/ validation metrics, one block per eval
      summary.md
      <your-folder>/        anything user code logs via run_log.note()/record()/dir()

Accessing the logger from user code:

* in a custom algorithm / ``build_batch``: ``ctx.extras["run_log"]``
* in a ``Callback``: ``state.run_log``
* anywhere during a run (e.g. inside a ``transforms`` ``__call__``)::

      from evsys_sdk import get_run_log
      log = get_run_log()
      if log:
          log.note("my_transform", f"dropped {n} rows missing a tool_slug")

All writes are best-effort and never raise into training.
"""

from __future__ import annotations

import contextvars
import csv
import io
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .logger import get_logger

if TYPE_CHECKING:  # avoid importing the (tinker-heavy) training package at runtime
    from .training.trajectory import TrajectoryGroup

log = get_logger(__name__)

# SDK default folders, numbered for reading order.
DATA = "01_data"
TRAIN_ROLLOUTS = "02_training_rollouts"
VAL_ROLLOUTS = "03_validation_rollouts"
TRAIN_METRICS = "04_training_metrics"
VAL_METRICS = "05_validation_metrics"

# Training scalars worth showing a human (substring match, case-insensitive).
_METRIC_WHITELIST = (
    "loss", "nll", "reward/mean", "reward_mean", "lr", "learning_rate",
    "accuracy", "pass_rate", "kl", "advantage",
)
_PREVIEW_ROWS = 5
_SNIPPET_CHARS = 800
_MAX_TOKENS_SHOWN = 80

# Active RunLog for the current run, so user code (transforms, etc.) can reach it.
_CURRENT: contextvars.ContextVar = contextvars.ContextVar("evsys_run_log", default=None)


def get_run_log() -> RunLog | None:
    """Return the RunLog for the run currently executing, or ``None`` outside a run.

    Lets user-written ``transforms`` / helpers log into the same per-run log::

        from evsys_sdk import get_run_log
        log = get_run_log()
        if log:
            log.note("my_transform", "...")
    """
    return _CURRENT.get()


def _clip(s: str, n: int = _SNIPPET_CHARS) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + " …"


def _decode(turn: Any, tokenizer: Any) -> str:
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


def _decode_token(tokenizer: Any, tid: int) -> str:
    if tokenizer is not None:
        try:
            return repr(str(tokenizer.decode([tid])))
        except Exception:  # pragma: no cover - defensive
            pass
    return str(tid)


class RunLog:
    """One human-readable log for a single run. Methods are best-effort: a
    logging failure is swallowed (it must never fail a training run)."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        experiment_name: str | None = None,
        run_name: str | None = None,
        hypothesis: str | None = None,
    ) -> None:
        self.root = Path(run_dir).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta = {
            "experiment_name": experiment_name,
            "run_name": run_name,
            "hypothesis": hypothesis,
        }
        self._evals: list[dict[str, Any]] = []

    # -- harbor: its own jobs dir (full rollout store) — referenced, not copied --

    def harbor_dir(self, phase: str) -> Path:
        """Directory a caller passes as ``run_harbor_rollouts(workspace_dir=...)``.
        Harbor writes its full per-trial ``result.json`` store here; the clean
        decoded view goes in ``02_training_rollouts`` / ``03_validation_rollouts``,
        so we never copy harbor's data — this is the reference for full detail."""
        d = self.root / "harbor" / _safe(phase)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- generic API for user code (custom transforms / build_batch / callbacks) --

    def dir(self, name: str) -> Path:
        """Create and return a named folder under the run log. Use this to write
        whatever custom files your transform / algorithm wants."""
        d = self.root / _safe(name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def note(self, folder: str, text: str, *, title: str | None = None) -> None:
        """Append a markdown note to ``{folder}/notes.md``."""
        try:
            path = self.dir(folder) / "notes.md"
            block = (f"### {title}\n\n" if title else "") + text.rstrip() + "\n\n"
            with path.open("a") as f:
                f.write(block)
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.note failed", exc_info=True)

    def record(self, folder: str, row: dict[str, Any], *, filename: str = "records.jsonl") -> None:
        """Append one JSON row to ``{folder}/{filename}``."""
        try:
            with (self.dir(folder) / filename).open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.record failed", exc_info=True)

    # -- 01_data -------------------------------------------------------------

    def log_data(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        dataset_meta: dict[str, Any] | None = None,
        transforms: Sequence[Any] | None = None,
        n_preview: int = _PREVIEW_ROWS,
    ) -> None:
        """Record the data *after transforms* — the meta + a small readable
        preview of the rows the algorithm actually receives."""
        try:
            d = self.dir(DATA)
            meta = dataset_meta or {}
            lines = ["# Data going in (after transforms)", "", f"- rows: **{len(rows)}**"]
            for key in ("name", "version", "format", "source_kind"):
                if meta.get(key) is not None:
                    lines.append(f"- {key}: `{meta[key]}`")
            names = [getattr(t, "kind", None) or getattr(t, "name", None) or str(t)
                     for t in (transforms or [])]
            if names:
                lines.append(f"- transforms (in order): {', '.join(f'`{n}`' for n in names)}")
            lines.append("")
            lines.append(f"First {min(n_preview, len(rows))} rows → `after_transform.jsonl`.")
            (d / "data.md").write_text("\n".join(lines) + "\n")
            with (d / "after_transform.jsonl").open("w") as f:
                for row in list(rows)[:n_preview]:
                    f.write(json.dumps(row, default=str) + "\n")
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.log_data failed", exc_info=True)

    def log_chat_templates(
        self, rendered: Sequence[str], *, n: int = 3, label: str = "train"
    ) -> None:
        """Record the **exact chat template text** (not token ids) the model
        sees for the first ``n`` examples."""
        try:
            d = self.dir(DATA)
            out = [f"# Chat template going in ({label})", "",
                   "Exact rendered prompt(s) the model sees (decoded text, not token ids).", ""]
            for i, text in enumerate(list(rendered)[:n]):
                out.append(f"### example {i}")
                out.append("```")
                out.append(_clip(str(text)))
                out.append("```")
                out.append("")
            (d / "chat_template.md").write_text("\n".join(out) + "\n")
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.log_chat_templates failed", exc_info=True)

    # -- 02_training_rollouts ------------------------------------------------

    def log_training_rollouts(
        self,
        step: int,
        groups: Sequence[TrajectoryGroup],
        *,
        advantages: Sequence[Sequence[float]] | None = None,
        tokenizer: Any = None,
        tasks: Sequence[Any] | None = None,
        k_groups: int = 2,
        max_tokens: int = _MAX_TOKENS_SHOWN,
    ) -> None:
        """Record training rollouts for ``step``: the predicted text, per-token
        logprobs, and the reward & advantage per sample per trajectory group."""
        try:
            self._log_rollouts(TRAIN_ROLLOUTS, f"step_{step}", groups, advantages,
                               tokenizer, tasks, k_groups, max_tokens)
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.log_training_rollouts failed", exc_info=True)

    # -- 03_validation_rollouts ----------------------------------------------

    def log_validation_rollouts(
        self,
        label: str,
        groups: Sequence[TrajectoryGroup],
        *,
        split: str = "val",
        tokenizer: Any = None,
        tasks: Sequence[Any] | None = None,
        k_groups: int = 3,
        max_tokens: int = _MAX_TOKENS_SHOWN,
    ) -> None:
        """Record eval rollout predictions for one eval, tagged by ``split``
        (the benchmark's tag — ``val`` or ``test``). Files land under
        ``03_validation_rollouts/{split}/{label}.md`` so val and test stay
        separated even when both run in-loop."""
        try:
            self._log_rollouts(f"{VAL_ROLLOUTS}/{_safe(split)}", _safe(label), groups,
                               None, tokenizer, tasks, k_groups, max_tokens)
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.log_validation_rollouts failed", exc_info=True)

    def _log_rollouts(self, folder, fname, groups, advantages, tokenizer, tasks,
                      k_groups, max_tokens) -> None:
        d = self.dir(folder)
        all_rewards = [float(getattr(t, "reward", 0.0))
                       for g in groups for t in getattr(g, "trajectories", [])]
        out = [f"# {fname}", ""]
        if all_rewards:
            out.append(
                f"- rollouts: **{len(all_rewards)}** over {len(groups)} task(s) · "
                f"reward mean **{statistics.fmean(all_rewards):.3f}** "
                f"(min {min(all_rewards):.3f}, max {max(all_rewards):.3f})"
            )
        out.append("")
        for gi, g in enumerate(list(groups)[:k_groups]):
            trajs = list(getattr(g, "trajectories", []))
            if not trajs:
                continue
            out.append(f"## trajectory group {gi}")
            if tasks is not None and gi < len(tasks):
                instr = getattr(tasks[gi], "instruction", None)
                if instr:
                    out.append(f"> task: {_clip(str(instr), 300)}")
            out.append("")
            adv_g = list(advantages[gi]) if advantages and gi < len(advantages) else None
            for si, traj in enumerate(trajs):
                reward = float(getattr(traj, "reward", 0.0))
                adv = adv_g[si] if adv_g and si < len(adv_g) else None
                head = f"**sample {si}** · reward {reward:.3f}"
                if adv is not None:
                    head += f" · advantage {adv:.3f}"
                out.append(head)
                turns = list(getattr(traj, "turns", []))
                last = turns[-1] if turns else None
                out.append("prediction:")
                out.append("```")
                out.append(_clip(_decode(last, tokenizer)) if last else "<no turns>")
                out.append("```")
                # Per-token logprobs (loss signal), bounded.
                if last is not None:
                    toks = list(getattr(last, "completion_tokens", None) or [])
                    lps = list(getattr(last, "logprobs", None) or [])
                    if toks and lps:
                        out.append(f"per-token logprobs (first {min(max_tokens, len(toks))}):")
                        out.append("| i | token | logprob |")
                        out.append("| - | --- | --- |")
                        for i in range(min(max_tokens, len(toks), len(lps))):
                            out.append(f"| {i} | {_decode_token(tokenizer, toks[i])} | {lps[i]:.4f} |")
                out.append("")
        path = d / f"{fname}.md"
        path.write_text("\n".join(out) + "\n")

    # -- 04 / 05 metrics -----------------------------------------------------

    def render_metrics(self) -> None:
        """Split the run's ``metrics.jsonl`` into training (``04``) vs eval
        (``05``) CSVs. Eval metrics keep their tag (``val`` / ``test``) — taken
        from the row's ``split`` or the metric key's leading segment — in a
        ``split`` column, so val and test stay distinct even when both run
        in-loop n times."""
        try:
            path = self._find_metrics_jsonl()
            if path is None:
                return
            train_rows, eval_rows = [], []
            train_keys, eval_keys = [], []
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metrics = rec.get("metrics", rec)
                step = rec.get("step")
                row_split = str(rec.get("split", "") or "").lower()
                tr: dict[str, Any] = {}
                ev: dict[str, dict[str, Any]] = {}   # split -> {key: val}
                for k, v in metrics.items():
                    if not isinstance(v, (int, float)):
                        continue
                    seg = k.split("/", 1)[0].lower()
                    split = (seg if seg in ("val", "test")
                             else row_split if row_split in ("val", "test", "valid", "validation")
                             else None)
                    if split:
                        split = "val" if split.startswith("val") else split
                        ev.setdefault(split, {})[k] = v
                    elif any(w in k.lower() for w in _METRIC_WHITELIST):
                        tr[k] = v
                if tr:
                    train_rows.append({"step": step, **tr})
                    train_keys += [k for k in tr if k not in train_keys]
                for split, m in ev.items():
                    eval_rows.append({"step": step, "split": split, **m})
                    eval_keys += [k for k in m if k not in eval_keys]
            if train_rows:
                self._write_csv(self.dir(TRAIN_METRICS) / "metrics.csv",
                               ["step", *train_keys], train_rows)
            if eval_rows:
                self._write_csv(self.dir(VAL_METRICS) / "metrics.csv",
                               ["step", "split", *eval_keys], eval_rows)
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.render_metrics failed", exc_info=True)

    def log_validation_metrics(self, name: str, metrics: dict[str, float], *,
                               step: int | None = None, split: str = "val") -> None:
        """Append one benchmark's eval metrics block, tagged by ``split`` (the
        benchmark's tag — ``val`` or ``test``). One block per eval, so repeated
        in-loop evals each leave a record."""
        try:
            self._evals.append({"name": name, "split": split, "metrics": dict(metrics)})
            head = f"## [{split}] {name}" + (f" @ step {step}" if step is not None else "")
            out = [head, "", "| metric | value |", "| --- | --- |"]
            for k, v in metrics.items():
                out.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
            out.append("")
            path = self.dir(VAL_METRICS) / "metrics.md"
            with path.open("a") as f:
                f.write(("" if path.exists() else "# Eval metrics (val / test)\n\n") + "\n".join(out) + "\n")
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.log_validation_metrics failed", exc_info=True)

    # -- summary -------------------------------------------------------------

    def write_summary(self, *, status: str | None = None, best_metric: str | None = None,
                      best_value: float | None = None, conclusion: str | None = None) -> None:
        try:
            m = self._meta
            out = [f"# {m.get('experiment_name') or 'experiment'} — run `{m.get('run_name') or ''}`", ""]
            if m.get("hypothesis"):
                out += [f"**Hypothesis:** {m['hypothesis']}", ""]
            if status:
                out.append(f"- status: **{status}**")
            if best_metric and best_value is not None:
                out.append(f"- {best_metric}: **{best_value:.4f}**")
            out.append("")
            if self._evals:
                out += ["## Eval (val / test)", ""]
                for ev in self._evals:
                    head = ", ".join(
                        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in list(ev["metrics"].items())[:4]
                    )
                    out.append(f"- `[{ev.get('split', 'val')}]` **{ev['name']}** — {head}")
                out.append("")
            if conclusion:
                out += ["## Conclusion", "", conclusion, ""]
            (self.root / "summary.md").write_text("\n".join(out) + "\n")
        except Exception:  # pragma: no cover - defensive
            log.debug("run_log.write_summary failed", exc_info=True)

    # -- internals -----------------------------------------------------------

    def _find_metrics_jsonl(self) -> Path | None:
        for cand in (self.root / "logs" / "metrics.jsonl",
                     self.root / "logs" / "jsonl" / "metrics.jsonl"):
            if cand.exists():
                return cand
        hits = list(self.root.glob("logs/**/metrics.jsonl"))
        return hits[0] if hits else None

    @staticmethod
    def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        path.write_text(buf.getvalue())


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_/") else "_" for c in str(name))


def split_from_tags(tags: Sequence[str] | None) -> str:
    """Map a benchmark's ``tags`` to its eval split. ``test`` wins if present
    (the held-out set), else ``val``. Used to tag eval metrics + rollouts so
    val and test stay distinct when both run in-loop."""
    tl = {str(t).lower() for t in (tags or [])}
    return "test" if "test" in tl else "val"


__all__ = ["RunLog", "get_run_log", "split_from_tags"]

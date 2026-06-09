"""Scaffold a new experiment dir under ``experiments/<yyyymmdd>_<slug>/``.

Emits two files:
  * ``config.yaml`` — a minimal, runnable ``ExperimentConfig`` skeleton with
    placeholders for the model, data, algorithm, and a metadata block where
    hypothesis / success_metric / benchmark live.
  * ``run.py`` — the 3-line declarative entrypoint researchers run:

        from evsys_sdk import Experiment
        import scripts   # registers project verifiers / metrics / transforms

        Experiment.from_yaml("config.yaml").run()

Use via the CLI: ``evsys new-experiment <slug>``. Programmatically:
``new_experiment(project_root, slug)``.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path


def new_experiment(
    project_root: str | Path,
    slug: str,
    *,
    today: date | None = None,
) -> Path:
    """Create ``<project_root>/experiments/<yyyymmdd>_<slug>/`` and write
    ``config.yaml`` + ``run.py`` inside. Returns the experiment dir.

    Refuses if the dir already exists — researchers can pick a different
    slug or delete the prior one.
    """
    project_root = Path(project_root).expanduser().resolve()
    safe_slug = _normalize_slug(slug)
    today_str = (today or date.today()).strftime("%Y%m%d")
    dir_name = f"{today_str}_{safe_slug}"

    experiments_root = project_root / "experiments"
    experiments_root.mkdir(parents=True, exist_ok=True)

    exp_dir = experiments_root / dir_name
    if exp_dir.exists():
        raise FileExistsError(
            f"{exp_dir} already exists — pick a different slug or remove the prior one"
        )
    exp_dir.mkdir()

    (exp_dir / "config.yaml").write_text(_config_yaml(safe_slug))
    (exp_dir / "run.py").write_text(_run_py())

    return exp_dir


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_SLUG_INVALID_RE = re.compile(r"[^a-z0-9_\-]+")


def _normalize_slug(slug: str) -> str:
    """Sanitize into ``[a-z0-9_-]+``."""
    cleaned = slug.strip().lower().replace(" ", "_")
    cleaned = _SLUG_INVALID_RE.sub("", cleaned)
    if not cleaned:
        raise ValueError(f"slug {slug!r} reduces to empty after normalization")
    return cleaned


def _config_yaml(slug: str) -> str:
    return f"""\
# Experiment config — read by `Experiment.from_yaml('config.yaml').run()`.
# Fields under `metadata` drive Experiment-level behavior (hypothesis,
# tags, success_metric, benchmark). The rest is a regular ExperimentConfig.

name: {slug}
output_dir: ./.evsys/outputs/{slug}

metadata:
  hypothesis: "TODO: one-line claim this experiment is testing"
  tags: []
  # success_metric: pass_rate          # ranks arms; sets experiment.best_score
  # benchmark:
  #   path: data/benchmark/<name>      # local harbor dir to score against
  #   id: <dashboard-benchmark-id>     # paste from `evsys benchmark upload`
  #   breakdown_keys: [toolkit]
  #   max_tokens: 512

# One run, or a `matrix:` sweep — see docs/DESIGN.md for the full schema.
run:
  name: {slug}
  seed: 42
  data:
    source_kind: jsonl
    path: data/datasets/<name>/v1/train.jsonl
    transforms: []
  model:
    name: Qwen/Qwen3-4B
  algorithm:
    kind: local_sft
    params:
      learning_rate: 1.0e-4
      num_epochs: 1
      batch_size: 8
  backend:
    kind: mock
"""


def _run_py() -> str:
    return '''\
"""Entrypoint for this experiment.

This file is intentionally tiny: all knobs live in config.yaml, all
project-specific verifiers / metrics / transforms live in scripts/.
"""
from __future__ import annotations

from pathlib import Path

from evsys_sdk import Experiment
import scripts  # noqa: F401 — registers project verifiers / metrics / transforms


if __name__ == "__main__":
    Experiment.from_yaml(Path(__file__).with_name("config.yaml")).run()
'''


__all__ = ["new_experiment"]

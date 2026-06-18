"""Standalone benchmark run — score a benchmark on a model, no training.

Reuses the *same* harbor rollout path as in-training eval
(:func:`evsys_sdk.training.harbor_eval.score_via_harbor`) plus the same metrics
and eval-rollout upload. The only differences from a post-training eval: there's
no checkpoint, and the rollout LLM is a closed / API model via litellm
(``model_client="litellm"``). Rollouts persist under the local ``.evsys/``
workspace and, when a ``store`` + ``run_id`` are given, push to the dashboard.

    from evsys_sdk import run_benchmark

    # by local path, dashboard id, or name — same resolver as the config
    metrics = run_benchmark(path="data/benchmark/tool-search",
                            model="anthropic/claude-opus-4-1")
    metrics = run_benchmark(id="bench_abc123", model="openai/gpt-4o", store=store)

API keys come from the standard provider env vars (``ANTHROPIC_API_KEY``,
``OPENAI_API_KEY``, …).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .benchmark import Benchmark

_WORKSPACE_ROOT = os.environ.get("EVSYS_WORKSPACE") or "./.evsys"


def run_benchmark(
    benchmark: Benchmark | None = None,
    *,
    model: str,
    path: str | None = None,
    id: str | None = None,
    name: str | None = None,
    num_samples: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    limit: int | None = None,
    n_concurrent: int = 8,
    workspace_dir: str | Path | None = None,
    store: Any = None,
    run_id: str | None = None,
    eval_id: str | None = None,
) -> dict[str, float]:
    """Score a benchmark on a closed / API ``model`` through harbor — no training.

    The benchmark is given directly (``benchmark=``) or resolved by
    ``path`` / ``id`` / ``name`` via the shared :meth:`Benchmark.load` resolver
    (same references the config accepts). ``model`` is a litellm string, e.g.
    ``"anthropic/claude-opus-4-1"`` / ``"openai/gpt-4o"``; repeats use the
    per-task ``num_samples`` in one async harbor job. Returns the eval metric
    dict (mean_reward, pass_rate, n_tasks, and time/tokens/cost per task), and
    uploads the per-task eval rollouts when both ``store`` and ``run_id`` are set.
    """
    import asyncio
    import tempfile

    from .training.harbor_eval import eval_predictions, upload_eval_rollouts

    if benchmark is None:
        benchmark = Benchmark.load({"path": path, "id": id, "name": name}, store=store)
        if benchmark is None:
            raise ValueError("run_benchmark: pass a Benchmark or one of path / id / name")

    tasks = benchmark.tasks if limit is None else benchmark.tasks[: max(0, int(limit))]

    # Persist rollouts under the local .evsys/ workspace (not an ephemeral
    # tempdir) when we can; fall back to a tempdir if .evsys isn't writable.
    if workspace_dir is not None:
        ws = Path(workspace_dir)
    else:
        safe = f"{benchmark.name}_{model}".replace("/", "_").replace(" ", "_")
        ws = Path(_WORKSPACE_ROOT) / "outputs" / "benchmark_runs" / safe
    try:
        ws.mkdir(parents=True, exist_ok=True)
    except OSError:
        ws = Path(tempfile.mkdtemp(prefix="evsys_bench_"))

    score = asyncio.run(benchmark.score_via_harbor(
        model_name=model,
        model_path=None,            # no checkpoint — the API model *is* the policy
        model_client="litellm",
        workspace_dir=ws,
        num_samples=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        limit=limit,
        n_concurrent=n_concurrent,
    ))

    if store is not None and run_id:
        preds = eval_predictions(tasks, score.rollouts, eval_id=eval_id, step=None)
        upload_eval_rollouts(store, run_id, preds)
    return score.metrics


__all__ = ["run_benchmark"]

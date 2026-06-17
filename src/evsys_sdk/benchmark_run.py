"""Standalone benchmark run — score a benchmark on a model, no training.

Reuses the *same* harbor rollout path as in-training eval
(:func:`evsys_sdk.training.harbor_eval.score_via_harbor`), so the only thing
that differs from a post-training eval is: there's no checkpoint and the rollout
LLM is a closed / API model selected via litellm (``model_client="litellm"``).

    from evsys_sdk import run_benchmark
    from evsys_sdk.benchmark import Benchmark

    bench = Benchmark.from_dir("data/benchmark/tool-search")
    metrics = run_benchmark(bench, model="anthropic/claude-opus-4-1")
    # -> {"mean_reward", "pass_rate", "n_tasks", "time_per_task",
    #     "tokens_per_task", "cost_per_task"}

API keys come from the standard provider env vars (``ANTHROPIC_API_KEY``,
``OPENAI_API_KEY``, …). When ``store`` + ``run_id`` are given the per-task eval
rollouts are uploaded (``kind='eval'``) just like in-training eval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .benchmark import Benchmark


def run_benchmark(
    benchmark: Benchmark,
    *,
    model: str,
    num_samples: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    limit: int | None = None,
    workspace_dir: str | Path | None = None,
    store: Any = None,
    run_id: str | None = None,
    eval_id: str | None = None,
) -> dict[str, float]:
    """Score ``benchmark`` on a closed / API ``model`` through harbor — no training.

    ``model`` is a litellm string, e.g. ``"anthropic/claude-opus-4-1"`` /
    ``"openai/gpt-4o"``. Returns the eval metric dict (mean_reward, pass_rate,
    n_tasks, and time/tokens/cost per task). Uploads the per-task eval rollouts
    when both ``store`` and ``run_id`` are provided.
    """
    import asyncio
    import tempfile

    from .training.harbor_eval import (
        eval_metrics,
        eval_predictions,
        score_via_harbor,
        upload_eval_rollouts,
    )

    tasks = benchmark.tasks if limit is None else benchmark.tasks[: max(0, int(limit))]
    ws = Path(workspace_dir) if workspace_dir else Path(tempfile.mkdtemp(prefix="evsys_bench_"))
    ws.mkdir(parents=True, exist_ok=True)

    groups = asyncio.run(score_via_harbor(
        tasks,
        model_name=model,
        model_path=None,            # no checkpoint — the API model *is* the policy
        workspace_dir=ws,
        model_client="litellm",
        num_samples=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    ))

    metrics = eval_metrics(groups)
    if store is not None and run_id:
        preds = eval_predictions(tasks, groups, eval_id=eval_id, step=None)
        upload_eval_rollouts(store, run_id, preds)
    return metrics


__all__ = ["run_benchmark"]

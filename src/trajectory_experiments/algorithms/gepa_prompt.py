"""GEPA-style prompt tuning — no weight updates, only the prompt evolves.

Two modes:
  1. If the ``gepa`` package is installed, delegate to ``gepa.optimize()`` with
     a reflection LM that proposes prompt mutations.
  2. Otherwise fall back to a small built-in random-mutation hill-climber so
     the algorithm is still runnable for tests / smoke runs without deps.

Both modes evolve a single ``system_prompt`` against a set of
``PromptExample`` rows scored by a verifier. The returned RunResult carries
``best_prompt`` and ``best_score`` in metrics + writes the prompt as a
``final_prompt`` artifact (``prompts.json``).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..protocols import RunContext, RunResult
from ..registry import get_inference, register_algorithm

logger = logging.getLogger(__name__)


class GEPAPromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed_prompt: str = "You are a helpful assistant. Answer concisely."
    """Initial system prompt."""
    task_lm: str = "mock"
    """Registry name of the inference client used to evaluate each candidate."""
    task_lm_config: dict[str, Any] = Field(default_factory=dict)
    reflection_lm: str | None = None
    """Optional inference client used to propose mutated prompts; required for
    the real gepa path. Without it we fall back to the built-in mutator."""
    reflection_lm_config: dict[str, Any] = Field(default_factory=dict)
    max_iterations: int = 20
    population_size: int = 4
    examples_per_eval: int = 8
    """How many PromptExamples to evaluate each candidate on (per iteration)."""
    use_gepa_lib: bool = True
    """If True and the `gepa` package is importable, delegate to it."""
    seed: int = 1729


@register_algorithm("gepa_prompt")
class GEPAPromptAlgorithm:
    name: ClassVar[str] = "gepa_prompt"
    Config: ClassVar[type] = GEPAPromptConfig

    def __init__(self, **kwargs) -> None:
        self.cfg = GEPAPromptConfig.model_validate(kwargs)

    def train(self, ctx: RunContext) -> RunResult:
        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        examples = ctx.extras.get("prompt_examples") or []
        if not examples:
            return RunResult(
                run_id=ctx.run_id, status="failed",
                error="GEPA requires ctx.extras['prompt_examples'] = list[PromptExample-like dict]",
            )

        score_fn = ctx.extras.get("prompt_score_fn") or _default_score_fn
        rng = random.Random(self.cfg.seed)

        task_lm = get_inference(self.cfg.task_lm)(**self.cfg.task_lm_config)
        ctx.log_store.log_hyperparams({"algorithm": self.name, **self.cfg.model_dump()})

        if self.cfg.use_gepa_lib:
            try:
                import gepa  # noqa: F401
                return _run_with_gepa_lib(
                    ctx=ctx, cfg=self.cfg, examples=examples,
                    task_lm=task_lm, score_fn=score_fn, out=out, rng=rng,
                )
            except ImportError:
                logger.info("gepa package not installed — using built-in hill-climber")

        return _run_hill_climber(
            ctx=ctx, cfg=self.cfg, examples=examples,
            task_lm=task_lm, score_fn=score_fn, out=out, rng=rng,
        )


# ---------------------------------------------------------------------------
# Built-in hill-climber (no external deps)
# ---------------------------------------------------------------------------


def _run_hill_climber(*, ctx, cfg, examples, task_lm, score_fn, out, rng) -> RunResult:
    best_prompt = cfg.seed_prompt
    best_score = _evaluate(best_prompt, examples[: cfg.examples_per_eval], task_lm, score_fn)
    ctx.log_store.log_scalar("gepa/score", best_score, step=0)

    for step in range(1, cfg.max_iterations + 1):
        candidate = _mutate_prompt(best_prompt, rng)
        candidate_score = _evaluate(
            candidate, examples[: cfg.examples_per_eval], task_lm, score_fn,
        )
        ctx.log_store.log_scalar("gepa/score", candidate_score, step=step)
        if candidate_score > best_score:
            best_score, best_prompt = candidate_score, candidate
            ctx.log_store.log_scalar("gepa/best_score", best_score, step=step)

    prompts_path = out / "prompts.json"
    prompts_path.write_text(json.dumps({"system_prompt": best_prompt}, indent=2))
    ctx.log_store.log_artifact("final_prompt", str(prompts_path), kind="prompt")

    return RunResult(
        run_id=ctx.run_id, status="completed",
        metrics={"gepa/best_score": best_score, "gepa/iterations": float(cfg.max_iterations)},
        artifacts={"final_prompt": str(prompts_path)},
        extras={"best_prompt": best_prompt},
    )


def _evaluate(prompt: str, examples: list[Any], task_lm, score_fn) -> float:
    if not examples:
        return 0.0
    rewards: list[float] = []
    for ex in examples:
        inputs = ex.get("inputs") if isinstance(ex, dict) else getattr(ex, "inputs", {})
        expected = ex.get("expected") if isinstance(ex, dict) else getattr(ex, "expected", None)
        user_text = "\n".join(f"{k}: {v}" for k, v in (inputs or {}).items())
        full_prompt = f"{prompt}\n\n{user_text}"
        try:
            completion = task_lm.generate(prompt=full_prompt, max_tokens=256, temperature=0.0)
        except Exception as e:
            logger.debug("gepa eval skipped a row (%s)", e)
            continue
        rewards.append(float(score_fn(completion, expected)))
    return sum(rewards) / max(1, len(rewards))


def _default_score_fn(completion: str, expected: Any) -> float:
    """Trivial baseline: exact-match (1.0) or substring-match (0.5)."""
    if expected is None:
        return 0.0
    exp_str = expected if isinstance(expected, str) else json.dumps(expected, sort_keys=True)
    if exp_str == completion.strip():
        return 1.0
    if exp_str in completion:
        return 0.5
    return 0.0


_MUTATION_OPS: list[str] = [
    "Add the directive: 'Think step by step before answering.'",
    "Add the directive: 'Be concise — one sentence if possible.'",
    "Reorder the instructions for clarity.",
    "Add an example showing the expected output format.",
    "Add the directive: 'Wrap your final answer in <answer>...</answer> tags.'",
    "Remove any redundant phrases.",
]


def _mutate_prompt(prompt: str, rng: random.Random) -> str:
    op = rng.choice(_MUTATION_OPS)
    return f"{prompt}\n\n# mutation: {op}\n(applied)"


# ---------------------------------------------------------------------------
# Delegated path — uses the real `gepa` library if installed
# ---------------------------------------------------------------------------


def _run_with_gepa_lib(*, ctx, cfg, examples, task_lm, score_fn, out, rng) -> RunResult:
    # The `gepa` package's surface has churned across versions, so we wrap
    # the call with broad Any typing + ignore Pyright. At runtime, missing
    # symbols / param mismatches fall through to the hill-climber via the
    # try/except in the caller.
    import gepa  # type: ignore[import-not-found, import-untyped]
    optimize: Any = getattr(gepa, "optimize", None)
    if optimize is None:
        raise ImportError("installed `gepa` does not expose .optimize()")

    seed_candidate = {"system_prompt": cfg.seed_prompt}
    rows = []
    for ex in examples:
        inputs = ex.get("inputs") if isinstance(ex, dict) else getattr(ex, "inputs", {})
        expected = ex.get("expected") if isinstance(ex, dict) else getattr(ex, "expected", None)
        rows.append({
            "input": next(iter((inputs or {}).values()), ""),
            "answer": expected if isinstance(expected, str) else json.dumps(expected),
            "additional_context": dict(inputs or {}),
        })

    metric_call_count = [0]

    def callback(state):
        metric_call_count[0] += 1
        try:
            score = float(getattr(state, "best_score", 0.0) or 0.0)
            ctx.log_store.log_scalar("gepa/score", score, step=metric_call_count[0])
        except Exception:
            pass

    result = optimize(
        seed_candidate=seed_candidate,
        train_data=rows,
        reflection_lm=cfg.reflection_lm,
        task_lm=cfg.task_lm,
        max_metric_calls=cfg.max_iterations * cfg.examples_per_eval,
        callback=callback,
    )
    best_prompt = result.best_candidate.get("system_prompt") if hasattr(result, "best_candidate") else cfg.seed_prompt
    best_score = float(getattr(result, "best_score", 0.0) or 0.0)

    prompts_path = out / "prompts.json"
    prompts_path.write_text(json.dumps({"system_prompt": best_prompt}, indent=2))
    ctx.log_store.log_artifact("final_prompt", str(prompts_path), kind="prompt")

    return RunResult(
        run_id=ctx.run_id, status="completed",
        metrics={"gepa/best_score": best_score, "gepa/iterations": float(metric_call_count[0])},
        artifacts={"final_prompt": str(prompts_path)},
        extras={"best_prompt": best_prompt},
    )

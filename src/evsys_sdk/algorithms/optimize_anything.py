"""``optimize_anything`` — engine-pluggable prompt/artifact optimization.

Connects gepa's new engine-pluggable ``optimize_anything`` API (v0.1.4 blog
release) to the SDK: the same ``{kind, params}`` algorithm surface as
``gepa_prompt``, but the optimizer loop is selectable — gepa's reflective
mutation, an autonomous Claude Code session (``autoresearch``), an
agent-based proposer under framework orchestration (``meta_harness``), or
``best_of_n`` — and composable: set ``explore_engines`` to run several
engines on a small budget first and continue from the best candidate with
``engine`` (the "omni" pattern from the release post).

Contract (same as ``gepa_prompt``): rows come from
``ctx.extras["prompt_examples"]`` (``{inputs, expected}``); the project may
supply ``ctx.extras["prompt_score_fn"]``. The score fn may return a bare
float or ``(float, feedback)`` — feedback strings become Actionable Side
Information for the engine, which is what the proposers mutate against, so
rich feedback matters more here than under ``gepa_prompt``.

Requires the engine API (``gepa.optimize_anything.OptimizeAnythingConfig``);
PyPI 0.1.4 predates it — install gepa from GitHub main until it ships.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..protocols import RunContext, RunResult
from ..registry import get_inference, register_algorithm
from ..training.callbacks import dispatch, make_loop_state

logger = logging.getLogger(__name__)

_ENGINES = ("gepa", "autoresearch", "meta_harness", "best_of_n")


class OptimizeAnythingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_prompt: str | None = None
    """Initial artifact text. Exactly one of seed_prompt / seed_path."""
    seed_path: str | None = None
    """Read the initial artifact from this file (e.g. a live prompt.txt)."""
    objective: str = "Maximize the evaluator score."
    """What "good" looks like — handed to every engine."""
    background: str | None = None
    """Extra task context for the engines (domain, constraints)."""

    engine: str = "gepa"
    """Main engine: gepa | autoresearch | meta_harness | best_of_n."""
    explore_engines: list[str] = Field(default_factory=list)
    """Omni phase 1: run each of these on ``explore_max_evals``, keep the best
    candidate, then continue with ``engine`` under the main budget. Empty →
    single-engine run."""
    explore_max_evals: int = 10
    """Per-engine eval budget for the explore phase."""

    max_evals: int | None = 60
    """Main-phase cap on evaluator calls (None = uncapped; then
    max_token_cost must be set)."""
    max_token_cost: float | None = None
    """USD cap on the engine's own proposer LLM spend."""
    stop_at_score: float | None = None
    max_concurrency: int = 4
    sandbox: bool = True
    """OS-jail subprocess engines' claude sessions (autoresearch/meta_harness)."""
    engine_config: dict[str, Any] = Field(default_factory=dict)
    """Engine-specific passthrough — e.g. for gepa:
    {"reflection": {"reflection_lm": "anthropic/claude-sonnet-4-6"}}. Applied
    ONLY to phases running ``engine`` (each engine rejects keys it doesn't
    know, so other explore engines get an empty config)."""

    task_lm: str = "mock"
    """Inference-registry client used to run each candidate on the examples."""
    task_lm_config: dict[str, Any] = Field(default_factory=dict)
    max_tokens: int = 512
    temperature: float = 0.0
    examples_per_eval: int | None = None
    """Cap rows per evaluation (None = all rows)."""
    score_fn_import: str | None = None
    """Load the score fn from ``file.py:fn`` or ``dotted.module:fn`` so a pure
    config.yaml run can bring its own judge (mirrors trigger.import_path).
    Overridden by ctx.extras['prompt_score_fn'] when present."""


def _load_score_fn(spec: str):
    """Resolve ``file.py:fn`` / ``dotted.module:fn`` to a callable."""
    import importlib
    import importlib.util

    target, _, fn_name = spec.partition(":")
    if not fn_name:
        raise ValueError(f"score_fn_import must look like 'module_or_file:fn' (got {spec!r})")
    p = Path(target)
    if p.suffix == ".py":
        module_spec = importlib.util.spec_from_file_location(f"_evsys_oa_{p.stem}", p)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, fn_name)


def _load_oa() -> Any:  # seam: tests monkeypatch this
    import gepa.optimize_anything as oa

    if not hasattr(oa, "OptimizeAnythingConfig"):
        raise ImportError(
            "installed `gepa` lacks the engine-pluggable optimize_anything API "
            "(need the v0.1.4 blog release; PyPI 0.1.4 predates it — "
            "pip install 'gepa @ git+https://github.com/gepa-ai/gepa')"
        )
    return oa


def _score(score_fn, completion: str, expected: Any) -> tuple[float, dict]:
    """Normalize score_fn output to (score, info): bare float or (float, feedback)."""
    out = score_fn(completion, expected)
    if isinstance(out, tuple):
        score, feedback = out
        info = feedback if isinstance(feedback, dict) else {"Feedback": str(feedback)}
        return float(score), info
    return float(out), {}


@register_algorithm("optimize_anything")
class OptimizeAnythingAlgorithm:
    name: ClassVar[str] = "optimize_anything"
    Config: ClassVar[type] = OptimizeAnythingConfig

    def __init__(self, **kwargs) -> None:
        self.cfg = OptimizeAnythingConfig.model_validate(kwargs)
        unknown = [e for e in [self.cfg.engine, *self.cfg.explore_engines] if e not in _ENGINES]
        if unknown:
            raise ValueError(f"unknown optimize_anything engine(s) {unknown}; pick from {_ENGINES}")
        if not self.cfg.max_evals and not self.cfg.max_token_cost:
            raise ValueError("set max_evals and/or max_token_cost so the run is bounded")
        if bool(self.cfg.seed_prompt) == bool(self.cfg.seed_path):
            raise ValueError("set exactly one of seed_prompt / seed_path")

    # -- pieces -------------------------------------------------------------

    def _seed(self) -> str:
        if self.cfg.seed_prompt is not None:
            return self.cfg.seed_prompt
        assert self.cfg.seed_path is not None  # enforced in __init__
        return Path(self.cfg.seed_path).expanduser().read_text().strip()

    def _rows(self, ctx: RunContext) -> list[dict]:
        rows = []
        # prompt_examples (programmatic) wins; config-driven runs land in train_rows.
        for ex in ctx.extras.get("prompt_examples") or ctx.extras.get("train_rows") or []:
            inputs = ex.get("inputs") if isinstance(ex, dict) else getattr(ex, "inputs", {})
            expected = ex.get("expected") if isinstance(ex, dict) else getattr(ex, "expected", None)
            rows.append({"inputs": dict(inputs or {}), "expected": expected})
        k = self.cfg.examples_per_eval
        if k and k < len(rows):
            # Evenly-spaced sample, not rows[:k] — a head slice of a
            # chronologically sorted dataset optimizes against one era/template
            # and overfits (seen live: prompt tuned on 8 cold-intros regressed
            # on held-out follow-up emails).
            step = len(rows) / k
            rows = [rows[int(i * step)] for i in range(k)]
        return rows

    def _make_evaluator(self, ctx: RunContext, cbs, state):
        from .gepa_prompt import _default_score_fn  # shared default contract

        score_fn = ctx.extras.get("prompt_score_fn")
        if score_fn is None and self.cfg.score_fn_import:
            score_fn = _load_score_fn(self.cfg.score_fn_import)
        score_fn = score_fn or _default_score_fn
        task_lm = get_inference(self.cfg.task_lm)(**self.cfg.task_lm_config)
        n_calls = [0]

        def evaluate(candidate: str, example: dict) -> tuple[float, dict]:
            user_text = "\n".join(f"{k}: {v}" for k, v in example["inputs"].items())
            try:
                completion = task_lm.generate(
                    prompt=f"{candidate}\n\n{user_text}",
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                )
            except Exception as e:  # a failed rollout is a scored failure, not a crash
                return 0.0, {"Feedback": f"generation error: {e}"}
            score, info = _score(score_fn, completion, example["expected"])
            info.setdefault("Generated", completion[:2000])
            n_calls[0] += 1
            state.step = n_calls[0]
            dispatch(cbs, "on_step_end", state, n_calls[0], None, {"oa/eval_score": score})
            return score, info

        return evaluate, n_calls

    def _oa_config(self, oa, *, engine: str, max_evals: int | None, out: Path,
                   phase: str = "main"):
        # Phase-distinct dirs: explore-gepa and the gepa continuation must not
        # share an eval log (the server numbers evals per output_dir), and the
        # UI attributes points to phases by this layout.
        stem = f"oa-{engine}" if phase == "main" else f"oa-{phase}-{engine}"
        return oa.OptimizeAnythingConfig(
            engine=engine,
            max_evals=max_evals,
            max_token_cost=self.cfg.max_token_cost,
            max_concurrency=self.cfg.max_concurrency,
            stop_at_score=self.cfg.stop_at_score,
            sandbox=self.cfg.sandbox,
            output_dir=str(out / stem),
            run_dir=str(out / f"{stem}-work"),
            # engines hard-reject unknown keys, so the passthrough only goes to
            # the engine it was written for
            engine_config=dict(self.cfg.engine_config) if engine == self.cfg.engine else {},
        )

    # -- entry point ----------------------------------------------------------

    def train(self, ctx: RunContext) -> RunResult:
        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = self._rows(ctx)
        if not rows:
            return RunResult(
                run_id=ctx.run_id, status="failed",
                error="optimize_anything requires ctx.extras['prompt_examples']",
            )
        try:
            oa = _load_oa()
        except ImportError as e:
            return RunResult(run_id=ctx.run_id, status="failed", error=str(e))

        total_budget = self.cfg.max_evals
        cbs, state = make_loop_state(ctx, num_steps=total_budget or 0)
        evaluate, n_calls = self._make_evaluator(ctx, cbs, state)
        seed = self._seed()
        task = dict(
            evaluator=evaluate, dataset=rows,
            objective=self.cfg.objective, background=self.cfg.background,
        )

        phases: list[dict] = []
        if self.cfg.explore_engines:
            explore = oa.optimize_best_of(
                seed, **task,
                configs=[
                    self._oa_config(oa, engine=e, max_evals=self.cfg.explore_max_evals,
                                    out=out, phase="explore")
                    for e in self.cfg.explore_engines
                ],
            )
            phases.append({"phase": "explore", "engines": self.cfg.explore_engines,
                           "best_score": float(explore.best_score or 0.0)})
            seed = explore.best_candidate or seed

        result = oa.optimize_anything(
            seed, **task,
            config=self._oa_config(oa, engine=self.cfg.engine, max_evals=total_budget, out=out),
        )
        best = result.best_candidate if isinstance(result.best_candidate, str) else (
            (result.best_candidate or {}).get("system_prompt") or seed
        )
        best_score = float(result.best_score or 0.0)
        phases.append({"phase": "main", "engine": self.cfg.engine, "best_score": best_score})

        prompts_path = out / "prompts.json"
        prompts_path.write_text(json.dumps({"system_prompt": best}, indent=2))
        (out / "oa_summary.json").write_text(json.dumps(
            {"phases": phases, "eval_calls": n_calls[0], "engine": self.cfg.engine}, indent=2))

        return RunResult(
            run_id=ctx.run_id, status="completed",
            metrics={"oa/best_score": best_score, "oa/eval_calls": float(n_calls[0])},
            artifacts={"final_prompt": str(prompts_path)},
            extras={"best_prompt": best, "engine": self.cfg.engine, "phases": phases},
        )

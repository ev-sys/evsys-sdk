"""Experiment — the top-level OOP orchestrator.

What it replaces in researcher scripts: the manual
``create_experiment`` → ``create_group`` → per-arm ``create_run`` →
``run_experiment(cfg)`` → ``create_eval`` → ``set_conclusion`` choreography.
Today every sweep script hand-rolls that loop. An ``Experiment`` collapses it
to one declarative ``.run()`` call.

Usage:

    # experiments/<date>_<slug>/run.py
    from trajectory_labs import Experiment
    import scripts   # registers custom verifiers/transforms

    Experiment.from_yaml("config.yaml").run()

Per-arm failure isolation: if one sweep arm raises during training, it gets
marked ``status=failed`` on the dashboard and the remaining arms continue.
The experiment finishes ``completed`` if any arm succeeded.

Config carries the project-shaped fields under ``metadata``:

    metadata:
      hypothesis: "..."
      tags: ["sft", "qwen3_4b"]
      project_goal_id: "..."
      success_metric: "pass_rate"   # which metric ranks arms for best_score
      benchmark:                    # post-training eval (optional)
        path: "data/benchmark/<name>"
        id: "<dashboard benchmark id>"
        breakdown_keys: ["toolkit"]

Dependencies are injectable for testing:
  * ``store``: TrajectoryStore (None → skip dashboard records, run locally)
  * ``train_fn``: ``(cfg) -> list[RunResult]`` (default: ``runner.run_experiment``)
  * ``benchmark``: ``Benchmark`` (overrides metadata.benchmark.path)
  * ``inference_factory``: ``(RunResult, RunConfig) -> InferenceClient``
    (called once per completed arm to build the eval client)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .benchmark import Benchmark, BenchmarkScore
from .config import ExperimentConfig, RunConfig
from .protocols import InferenceClient, RunResult
from .step_metrics import forward_step_metrics
from .sweep import expand_runs

logger = logging.getLogger(__name__)


TrainFn = Callable[[ExperimentConfig], list[RunResult]]
InferenceFactory = Callable[[RunResult, RunConfig], InferenceClient]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    """One arm of the experiment (one expanded ``RunConfig``)."""

    name: str
    run_config: RunConfig
    status: str  # "completed" | "failed"
    metrics: dict[str, float] = field(default_factory=dict)
    """Training-side metrics from ``RunResult.metrics``."""
    eval_metrics: dict[str, float] = field(default_factory=dict)
    """Benchmark eval metrics, if a benchmark was scored."""
    eval_breakdowns: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    error: str | None = None
    train_seconds: float | None = None
    eval_seconds: float | None = None
    run_result: RunResult | None = None
    """Raw underlying ``RunResult`` for advanced consumers."""

    def score(self, metric: str) -> float | None:
        """Return ``eval_metrics[metric]`` if present, else ``metrics[metric]``."""
        if metric in self.eval_metrics:
            return self.eval_metrics[metric]
        return self.metrics.get(metric)


@dataclass
class ExperimentResult:
    """What ``Experiment.run`` returns."""

    name: str
    status: str  # "completed" | "failed"
    arms: list[ArmResult]
    best_arm: ArmResult | None
    best_score: float | None
    conclusion: str
    experiment_id: str | None = None
    hypothesis: str | None = None


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


class Experiment:
    """Bundle ExperimentConfig + dashboard writes + per-arm orchestration."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        store: Any | None = None,
        train_fn: TrainFn | None = None,
        benchmark: Benchmark | None = None,
        inference_factory: InferenceFactory | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.train_fn = train_fn or _default_train_fn
        self._benchmark_override = benchmark
        self.inference_factory = inference_factory

    # -- entry points -----------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: Any) -> Experiment:
        from .yaml_loader import load_yaml

        return cls(load_yaml(path), **kwargs)

    def run(self) -> ExperimentResult:
        meta = self.config.metadata or {}
        hypothesis = meta.get("hypothesis")
        tags = list(meta.get("tags") or [])
        success_metric = meta.get("success_metric")
        benchmark = self._resolve_benchmark(meta.get("benchmark") or {})

        experiment_id = self._create_experiment(hypothesis, tags, meta)

        arms: list[ArmResult] = []
        for run_cfg in self._iter_runs():
            arms.append(self._execute_arm(experiment_id, run_cfg, benchmark, meta))

        best_arm = self._pick_best(arms, success_metric) if success_metric else None
        best_score = best_arm.score(success_metric) if (best_arm and success_metric) else None
        conclusion = self._build_conclusion(arms, best_arm, success_metric)
        status = "completed" if any(a.status == "completed" for a in arms) else "failed"

        self._finalize_experiment(experiment_id, status, best_score, conclusion)

        return ExperimentResult(
            name=self.config.name,
            status=status,
            arms=arms,
            best_arm=best_arm,
            best_score=best_score,
            conclusion=conclusion,
            experiment_id=experiment_id,
            hypothesis=hypothesis,
        )

    # -- internals: orchestration steps; safe to override in subclasses ---

    def _iter_runs(self) -> list[RunConfig]:
        """Expand sweep matrix / single-run / multi-run to a flat list."""
        cfg = self.config
        if cfg.run is not None:
            return [cfg.run]
        if cfg.runs is not None:
            return list(cfg.runs)
        if cfg.matrix is not None:
            return expand_runs(cfg.matrix.base_run, cfg.matrix.axes, cfg.matrix.name_template)
        raise RuntimeError(f"experiment {cfg.name!r} has no runs")  # pragma: no cover

    def _resolve_benchmark(self, spec: dict) -> Benchmark | None:
        if self._benchmark_override is not None:
            return self._benchmark_override
        path = spec.get("path")
        if not path:
            return None
        return Benchmark.from_dir(path)

    def _create_experiment(
        self, hypothesis: str | None, tags: list[str], meta: dict
    ) -> str | None:
        if self.store is None:
            return None
        exp = self.store.create_experiment(
            experiment_name=self.config.name,
            hypothesis=hypothesis,
            tags=tags or None,
            project_goal_id=meta.get("project_goal_id"),
        )
        return exp.get("id") if isinstance(exp, dict) else None

    def _execute_arm(
        self,
        experiment_id: str | None,
        run_cfg: RunConfig,
        benchmark: Benchmark | None,
        meta: dict,
    ) -> ArmResult:
        run_id = self._create_run(experiment_id, run_cfg)
        arm = ArmResult(name=run_cfg.name, run_config=run_cfg, status="failed", run_id=run_id)
        try:
            arm = self._train_arm(arm, run_cfg)
            if arm.status == "completed" and benchmark is not None:
                arm = self._eval_arm(arm, run_cfg, benchmark, meta)
            self._mark_run_completed(run_id, arm)
        except Exception as e:
            logger.exception("arm %r failed", run_cfg.name)
            arm.status = "failed"
            arm.error = f"{type(e).__name__}: {e}"
            self._mark_run_failed(run_id, arm.error)
        return arm

    def _train_arm(self, arm: ArmResult, run_cfg: RunConfig) -> ArmResult:
        single_cfg = self.config.model_copy(
            update={"run": run_cfg, "runs": None, "matrix": None}
        )
        t0 = time.time()
        results = self.train_fn(single_cfg)
        arm.train_seconds = time.time() - t0
        if not results:
            raise RuntimeError(f"train_fn returned no results for arm {run_cfg.name!r}")
        result = results[0]
        arm.run_result = result
        arm.metrics = dict(result.metrics)
        self._forward_step_metrics(arm)
        if result.status != "completed":
            raise RuntimeError(result.error or f"train_fn status={result.status}")
        arm.status = "completed"
        return arm

    def _forward_step_metrics(self, arm: ArmResult) -> None:
        """Push the arm's local metrics.jsonl rows to the store.

        Runner-time logging writes locally; this batch-forwards to the
        dashboard so the script doesn't have to call backfill_step_metrics
        manually after training.
        """
        run_dir = self._resolve_run_dir(arm)
        forward_step_metrics(self.store, arm.run_id, run_dir)

    def _resolve_run_dir(self, arm: ArmResult) -> Path | None:
        """Reconstruct the run output dir the runner wrote into."""
        if arm.run_result is not None:
            from_artifact = arm.run_result.artifacts.get("run_dir")
            if from_artifact:
                return Path(from_artifact)
        safe_name = arm.run_config.name.replace("/", "_").replace(" ", "_")
        return Path(self.config.output_dir).expanduser() / safe_name

    def _eval_arm(
        self,
        arm: ArmResult,
        run_cfg: RunConfig,
        benchmark: Benchmark,
        meta: dict,
    ) -> ArmResult:
        if self.inference_factory is None:
            logger.info("no inference_factory; skipping benchmark eval for %r", run_cfg.name)
            return arm
        bench_meta = dict((meta.get("benchmark") or {}))
        assert arm.run_result is not None
        client = self.inference_factory(arm.run_result, run_cfg)
        t0 = time.time()
        score = benchmark.score(
            client,
            max_tokens=int(bench_meta.get("max_tokens", 512)),
            temperature=float(bench_meta.get("temperature", 0.0)),
            breakdown_keys=list(bench_meta.get("breakdown_keys") or []),
        )
        arm.eval_seconds = time.time() - t0
        arm.eval_metrics = dict(score.metrics)
        arm.eval_breakdowns = dict(score.breakdowns)
        self._record_eval(arm, benchmark, bench_meta, score)
        return arm

    # -- store passthroughs (each guarded so store=None is fine) ---------

    def _create_run(self, experiment_id: str | None, run_cfg: RunConfig) -> str | None:
        if self.store is None or experiment_id is None:
            return None
        run = self.store.create_run(
            experiment_id=experiment_id,
            recipe_kind=run_cfg.algorithm.kind,
            run_config=run_cfg.model_dump(),
            seed=run_cfg.seed,
            status="running",
        )
        return run.get("id") if isinstance(run, dict) else None

    def _mark_run_completed(self, run_id: str | None, arm: ArmResult) -> None:
        if self.store is None or run_id is None:
            return
        self.store.update_run(run_id, status="completed")

    def _mark_run_failed(self, run_id: str | None, error: str) -> None:
        if self.store is None or run_id is None:
            return
        try:
            self.store.update_run(run_id, status="failed", error_message=error)
        except Exception:
            logger.exception("failed to mark run %r failed", run_id)

    def _record_eval(
        self,
        arm: ArmResult,
        benchmark: Benchmark,
        bench_meta: dict,
        score: BenchmarkScore,
    ) -> None:
        if self.store is None or arm.run_id is None:
            return
        try:
            self.store.create_eval(
                run_id=arm.run_id,
                benchmark_id=bench_meta.get("id"),
                metrics=dict(score.metrics),
                breakdowns=dict(score.breakdowns) or None,
            )
        except Exception:
            logger.exception("failed to record eval for arm %r", arm.name)

    def _finalize_experiment(
        self,
        experiment_id: str | None,
        status: str,
        best_score: float | None,
        conclusion: str,
    ) -> None:
        if self.store is None or experiment_id is None:
            return
        patch: dict[str, Any] = {"status": status, "conclusion": conclusion}
        if best_score is not None:
            patch["best_score"] = best_score
        try:
            self.store.update_experiment(experiment_id, **patch)
        except Exception:
            logger.exception("failed to finalize experiment %r", experiment_id)

    # -- aggregation -----------------------------------------------------

    def _pick_best(self, arms: list[ArmResult], metric: str) -> ArmResult | None:
        scored = [(a, a.score(metric)) for a in arms if a.status == "completed"]
        scored = [(a, s) for a, s in scored if s is not None]
        if not scored:
            return None
        return max(scored, key=lambda pair: pair[1])[0]

    def _build_conclusion(
        self,
        arms: list[ArmResult],
        best_arm: ArmResult | None,
        success_metric: str | None,
    ) -> str:
        completed = [a for a in arms if a.status == "completed"]
        failed = [a for a in arms if a.status == "failed"]
        if not completed:
            return f"All {len(arms)} arms failed."
        parts = []
        if best_arm is not None and success_metric is not None:
            best_score = best_arm.score(success_metric)
            parts.append(
                f"Best arm: {best_arm.name} at {success_metric}={best_score:.4f}."
            )
        parts.append(f"{len(completed)}/{len(arms)} arms completed.")
        if failed:
            parts.append(f"Failed: {', '.join(a.name for a in failed)}.")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Default train_fn — defers the import so tests don't pull the heavy runner.
# ---------------------------------------------------------------------------


def _default_train_fn(cfg: ExperimentConfig) -> list[RunResult]:
    from .runner import run_experiment

    return run_experiment(cfg)


__all__ = [
    "ArmResult",
    "Experiment",
    "ExperimentResult",
    "TrainFn",
    "InferenceFactory",
]

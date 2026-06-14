"""Experiment — the top-level OOP orchestrator.

What it replaces in researcher scripts: the manual
``create_experiment`` → ``create_group`` → per-arm ``create_run`` →
``run_experiment(cfg)`` → ``create_eval`` → ``set_conclusion`` choreography.
Today every sweep script hand-rolls that loop. An ``Experiment`` collapses it
to one declarative ``.run()`` call.

Usage:

    # experiments/<date>_<slug>/run.py
    from evsys_sdk import Experiment
    import src   # registers custom verifiers/transforms

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
  * ``store``: EvsysStore (None → skip dashboard records, run locally)
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
from .inference.chat_templated import ChatTemplatedInference
from .protocols import InferenceClient, RunResult
from .registry import get_default_inference_factory
from .step_metrics import forward_step_metrics
from .sweep import expand_runs

logger = logging.getLogger(__name__)


TrainFn = Callable[[ExperimentConfig], list[RunResult]]
InferenceFactory = Callable[[RunResult, RunConfig], InferenceClient]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """One benchmark scored against one arm at one moment.

    Multiple ``EvalResult`` entries can attach to an ``ArmResult`` when an
    experiment carries several benchmarks under ``metadata.benchmark``
    (e.g. a `val` set + a `test` set). The ``step`` field disambiguates
    in-loop validation rows (with their training-step value) from a
    single post-training row (``step is None``).
    """

    name: str
    benchmark_id: str | None
    metrics: dict[str, float]
    breakdowns: dict[str, Any]
    eval_seconds: float
    step: int | None = None
    """``None`` → scored once post-training; int → in-loop at that step."""
    tags: list[str] = field(default_factory=list)


@dataclass
class ArmResult:
    """One arm of the experiment (one expanded ``RunConfig``)."""

    name: str
    run_config: RunConfig
    status: str  # "completed" | "failed"
    metrics: dict[str, float] = field(default_factory=dict)
    """Training-side metrics from ``RunResult.metrics``."""
    eval_metrics: dict[str, float] = field(default_factory=dict)
    """Back-compat alias — mirrors ``evals[<primary>].metrics``. ``primary``
    is the first ``test``-tagged post-training eval, or the first
    post-training eval, or the first eval overall."""
    eval_breakdowns: dict[str, Any] = field(default_factory=dict)
    evals: list[EvalResult] = field(default_factory=list)
    """All benchmarks scored against this arm — see ``EvalResult``."""
    run_id: str | None = None
    error: str | None = None
    train_seconds: float | None = None
    eval_seconds: float | None = None
    run_result: RunResult | None = None
    """Raw underlying ``RunResult`` for advanced consumers."""
    group_id: str | None = None
    """Dashboard ``group_id`` when ``n_repeats > 1`` (else None)."""
    group_name: str | None = None
    """Primary's name (= the group's name) when grouped; else None."""

    def eval(self, name: str, *, step: int | None = None) -> EvalResult | None:
        """Look up an eval result by benchmark ``name``.

        ``step=None`` (default) → return the post-training row (``step is None``)
        if present; else the last in-loop row for that benchmark.

        ``step=<int>`` → return the exact in-loop row at that step (or
        ``None`` if no exact match — callers can do their own nearest-step
        lookup over ``arm.evals``).
        """
        matches = [e for e in self.evals if e.name == name]
        if not matches:
            return None
        if step is None:
            post = [e for e in matches if e.step is None]
            if post:
                return post[0]
            in_loop = [e for e in matches if e.step is not None]
            return max(in_loop, key=lambda e: e.step) if in_loop else None
        return next((e for e in matches if e.step == step), None)

    def score(self, metric: str) -> float | None:
        """Return the metric value for ranking by ``success_metric``.

        Three forms supported:

        * ``"pass_rate"`` (bare) — looked up on ``eval_metrics`` (the
          primary post-training eval), then ``metrics`` (training-side).
        * ``"bench.pass_rate"`` (dotted) — looked up on
          ``arm.eval("bench").metrics["pass_rate"]``. The dotted form wins
          when an experiment carries multiple named benchmarks and
          ``success_metric`` picks one explicitly.
        * Returns ``None`` if neither path resolves.
        """
        if "." in metric:
            bench_name, _, key = metric.partition(".")
            ev = self.eval(bench_name)
            if ev is not None and key in ev.metrics:
                return ev.metrics[key]
            return None
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
        self._train_fn_is_default = train_fn is None
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
        benchmarks = self._resolve_benchmarks(meta.get("benchmark"))

        # In-loop entries (run_every > 0) are picked up by the algorithm
        # composer via build_in_loop_evaluators (training/evaluators.py),
        # which reads ctx.config.metadata directly. The list returned here
        # still contains them so the test for "is there ANY benchmark?"
        # works; _eval_arm skips them at post-training time.

        experiment_id = self._create_experiment(hypothesis, tags, meta)

        # When n_repeats > 1, register one dashboard group per primary
        # RunConfig; replicates share the group_id. n_repeats == 1 keeps the
        # previous behavior — no groups, no group_id on runs. 
        # TODO : even when n_repeats == 1, we should create a group.
        primaries = self._iter_runs()
        n_repeats = self.config.n_repeats
        group_id_by_name: dict[str, str | None] = {}
        if n_repeats > 1:
            for p in primaries:
                group_id_by_name[p.name] = self._create_group(experiment_id, p.name)

        arms: list[ArmResult] = []
        for primary in primaries:
            for arm_cfg, group_name in self._replicates_for(primary):
                group_id = group_id_by_name.get(group_name) if group_name else None
                arms.append(self._execute_arm(
                    experiment_id, arm_cfg, benchmarks, meta,
                    group_id=group_id, group_name=group_name,
                ))

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
        """Expand sweep matrix / single-run / multi-run to a flat list of primaries.

        Each entry is a "group" when ``n_repeats > 1``. See
        :meth:`_replicates_for` for the per-primary seeded replicates.
        """
        cfg = self.config
        if cfg.run is not None:
            return [cfg.run]
        if cfg.runs is not None:
            return list(cfg.runs)
        if cfg.matrix is not None:
            return expand_runs(cfg.matrix.base_run, cfg.matrix.axes, cfg.matrix.name_template)
        raise RuntimeError(f"experiment {cfg.name!r} has no runs")  # pragma: no cover

    def _replicates_for(self, primary: RunConfig) -> list[tuple[RunConfig, str | None]]:
        """Per-primary seed replicates.

        For ``n_repeats == 1``: returns ``[(primary, None)]`` — no group.
        For ``n_repeats > 1``: returns N tuples of
        ``(<RunConfig with name=primary.name__s<seed> and seed=<seed>>, primary.name)``.
        Seeds run ``[base_seed, base_seed+1, ...]`` when ``base_seed`` is set,
        else ``[primary.seed, primary.seed+1, ...]``.
        """
        n = self.config.n_repeats
        if n <= 1:
            return [(primary, None)]
        base = self.config.base_seed if self.config.base_seed is not None else primary.seed
        return [
            (
                primary.model_copy(update={"name": f"{primary.name}__s{base + i}", "seed": base + i}),
                primary.name,
            )
            for i in range(n)
        ]

    def _resolve_benchmarks(
        self, raw: dict | list | None
    ) -> list[tuple[Benchmark, dict]]:
        """Normalize ``metadata.benchmark`` to a list of ``(Benchmark, spec)``.

        Accepts three shapes:

        * ``None`` / empty → ``[]`` (no eval).
        * single ``dict`` (legacy single-benchmark form) → wrapped into a
          one-element list with ``name`` defaulting to ``"benchmark"`` (or
          the spec's own ``name`` if present).
        * ``list[dict]`` (new multi-benchmark form) → each entry must carry
          a ``name`` and may carry ``tags`` and ``run_every``.

        ``self._benchmark_override`` (test seam) bypasses everything and
        returns a single-entry list.
        """
        if self._benchmark_override is not None:
            return [(self._benchmark_override, {"name": "benchmark"})]
        if not raw:
            return []
        specs: list[dict] = [raw] if isinstance(raw, dict) else list(raw)
        out: list[tuple[Benchmark, dict]] = []
        for i, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise ValueError(
                    f"metadata.benchmark[{i}] must be a dict (got {type(spec).__name__})"
                )
            bench = self._materialize_benchmark(spec)
            if bench is None:
                continue
            # Default a name when missing — required for list form, harmless for single.
            spec = dict(spec)
            spec.setdefault("name", str(spec.get("id") or spec.get("path") or f"benchmark_{i}"))
            out.append((bench, spec))
        return out

    def _materialize_benchmark(self, spec: dict) -> Benchmark | None:
        """Resolve one benchmark spec dict to a ``Benchmark`` (or ``None``)."""
        path = spec.get("path")
        if path:
            return Benchmark.from_dir(path)
        # Preferred: a dashboard benchmark by id (or name → latest version's
        # id), pulled into the local .evsys/ workspace. path is the
        # offline / dev fallback above.
        bid, dashboard_name = spec.get("id"), spec.get("name")
        if not (bid or dashboard_name):
            return None
        from .data_types import harbor_task_from_dict
        from .workspace import Workspace, read_jsonl_rows
        ws = Workspace(self.store) if self.store is not None else Workspace()
        resolved = str(bid) if bid else ws.benchmark_id_for_name(str(dashboard_name))
        mat = ws.pull_benchmark(resolved)
        tasks = [harbor_task_from_dict(r) for r in read_jsonl_rows(mat.path)]
        return Benchmark.from_iterable(dashboard_name or "benchmark", tasks)

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
        benchmarks: list[tuple[Benchmark, dict]],
        meta: dict,
        *,
        group_id: str | None = None,
        group_name: str | None = None,
    ) -> ArmResult:
        run_id = self._create_run(experiment_id, run_cfg, group_id=group_id)
        arm = ArmResult(
            name=run_cfg.name,
            run_config=run_cfg,
            status="failed",
            run_id=run_id,
            group_id=group_id,
            group_name=group_name,
        )
        try:
            arm = self._train_arm(arm, run_cfg)
            if arm.status == "completed" and benchmarks:
                arm = self._eval_arm(arm, run_cfg, benchmarks, meta)
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
        from .runner import run_experiment
        if self._train_fn_is_default:
            # Hand the dashboard store + run_id down to the runner so in-loop
            # validation (harbor engine) can upload its eval rollouts tagged
            # with this run. Custom train_fns get the plain (cfg) contract.
            results = run_experiment(
                single_cfg,
                extra_context={"store": self.store, "dashboard_run_id": arm.run_id},
            )
        else:
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

    def _resolve_inference_factory(self, run_cfg: RunConfig) -> InferenceFactory | None:
        """User-supplied factory wins; otherwise pick a default by backend kind.

        Falls back to the registry's ``get_default_inference_factory`` so we
        don't have to import backend-specific inference modules here — e.g.
        ``tinker`` registers its own default at module load.
        """
        if self.inference_factory is not None:
            return self.inference_factory
        return get_default_inference_factory(run_cfg.backend.kind)

    def _eval_arm(
        self,
        arm: ArmResult,
        run_cfg: RunConfig,
        benchmarks: list[tuple[Benchmark, dict]],
        meta: dict,
    ) -> ArmResult:
        """Score each post-training benchmark and attach an EvalResult per entry.

        Entries flagged with ``run_every`` are in-loop and skipped here (their
        scoring happens during training in the algorithm wrapper — task
        commit 2). Entries without ``run_every`` get a single post-training
        ``EvalResult`` appended to ``arm.evals``.

        After all benchmarks score, the flat back-compat fields
        (``arm.eval_metrics`` / ``eval_breakdowns`` / ``eval_seconds``)
        mirror the **primary** eval — the first ``test``-tagged
        post-training row, else the first post-training row, else nothing.
        """
        factory = self._resolve_inference_factory(run_cfg)
        if factory is None:
            logger.info(
                "no inference_factory; skipping benchmark eval for %r", run_cfg.name
            )
            return arm
        assert arm.run_result is not None
        for bench, bench_meta in benchmarks:
            if bench_meta.get("run_every"):
                continue  # in-loop entry — scored by the algorithm wrapper
            if str(bench_meta.get("engine", "")).lower() == "harbor":
                # Opt-in: score this benchmark through harbor's rollout engine
                # (and upload the eval rollouts). Default path below is untouched.
                self._eval_arm_harbor(arm, run_cfg, bench, bench_meta)
                continue
            client = factory(arm.run_result, run_cfg)
            # Auto-wrap with chat templating when configured. Lets researchers
            # declare a system_prompt + user_template in YAML instead of
            # writing a per-project ChatTemplatedTinker shim in run.py.
            ct = bench_meta.get("chat_template") or {}
            if ct:
                client = ChatTemplatedInference(client, **ct)
            t0 = time.time()
            score = bench.score(
                client,
                max_tokens=int(bench_meta.get("max_tokens", 512)),
                temperature=float(bench_meta.get("temperature", 0.0)),
                breakdown_keys=list(bench_meta.get("breakdown_keys") or []),
                limit=int(bench_meta["limit"]) if bench_meta.get("limit") is not None else None,
            )
            seconds = time.time() - t0
            arm.evals.append(EvalResult(
                name=str(bench_meta.get("name", "benchmark")),
                benchmark_id=bench_meta.get("id"),
                metrics=dict(score.metrics),
                breakdowns=dict(score.breakdowns),
                eval_seconds=seconds,
                step=None,
                tags=list(bench_meta.get("tags") or []),
            ))
            self._record_eval(arm, bench, bench_meta, score)

        # Pick the primary post-training eval to mirror into the flat fields.
        post = [e for e in arm.evals if e.step is None]
        primary = next((e for e in post if "test" in e.tags), None) or (post[0] if post else None)
        if primary is not None:
            arm.eval_metrics = dict(primary.metrics)
            arm.eval_breakdowns = dict(primary.breakdowns)
            arm.eval_seconds = primary.eval_seconds
        return arm

    def _eval_arm_harbor(
        self, arm: ArmResult, run_cfg: RunConfig, bench: Benchmark, bench_meta: dict,
    ) -> None:
        """Score one benchmark through harbor's rollout engine and upload the
        eval rollouts (kind='eval'). Opt-in via ``benchmark.engine: harbor``."""
        import asyncio
        import tempfile

        from .training.harbor_eval import (
            eval_metrics,
            eval_predictions,
            score_via_harbor,
            upload_eval_rollouts,
        )

        model_path = self._final_checkpoint(arm)
        limit = int(bench_meta["limit"]) if bench_meta.get("limit") is not None else None
        tasks = bench.tasks if limit is None else bench.tasks[: max(0, limit)]
        ct = bench_meta.get("chat_template") or {}
        workspace = Path(tempfile.mkdtemp(prefix="evsys_eval_"))

        t0 = time.time()
        groups = asyncio.run(score_via_harbor(
            tasks,
            model_name=run_cfg.model.name,
            model_path=model_path,
            workspace_dir=workspace,
            num_samples=int(bench_meta.get("num_samples", 1)),
            max_tokens=int(bench_meta.get("max_tokens", 512)),
            temperature=float(bench_meta.get("temperature", 0.0)),
            renderer_name=run_cfg.model.renderer_name,
            system_prompt=ct.get("system_prompt"),
        ))
        seconds = time.time() - t0

        metrics = eval_metrics(groups)
        arm.evals.append(EvalResult(
            name=str(bench_meta.get("name", "benchmark")),
            benchmark_id=bench_meta.get("id"),
            metrics=metrics,
            breakdowns={},
            eval_seconds=seconds,
            step=None,
            tags=list(bench_meta.get("tags") or []),
        ))
        eval_id = self._record_eval(
            arm, bench, bench_meta, BenchmarkScore(metrics=metrics, per_task=[], breakdowns={}),
        )
        # Upload eval rollouts only (training rollouts are never uploaded), and
        # only once they have an eval_id to hang off of — orphan predictions
        # can't be told apart from other evals on the same run.
        if self.store is not None and arm.run_id:
            if eval_id is None:
                logger.warning(
                    "skipping eval rollout upload for arm %r: create_eval gave no id",
                    arm.name,
                )
            else:
                preds = eval_predictions(tasks, groups, eval_id=eval_id, step=None)
                upload_eval_rollouts(self.store, arm.run_id, preds)

    @staticmethod
    def _final_checkpoint(arm: ArmResult) -> str | None:
        """The trained sampler checkpoint URI from the arm's artifacts."""
        arts = (arm.run_result.artifacts if arm.run_result else {}) or {}
        return (
            arts.get("checkpoint-final")
            or arts.get("sampler_path")
            or next((v for k, v in arts.items()
                     if "sampler" in str(k) or "checkpoint" in str(k)), None)
        )

    # -- store passthroughs (each guarded so store=None is fine) ---------

    def _create_group(self, experiment_id: str | None, name: str) -> str | None:
        """Register a run group for variance studies; returns its id (or None)."""
        if self.store is None or experiment_id is None:
            return None
        try:
            grp = self.store.create_group(experiment_id, name)
        except Exception:
            logger.exception("failed to create group %r", name)
            return None
        return grp.get("id") if isinstance(grp, dict) else None

    def _create_run(
        self,
        experiment_id: str | None,
        run_cfg: RunConfig,
        *,
        group_id: str | None = None,
    ) -> str | None:
        if self.store is None or experiment_id is None:
            return None
        run = self.store.create_run(
            experiment_id=experiment_id,
            group_id=group_id,
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
            return None
        try:
            ev = self.store.create_eval(
                run_id=arm.run_id,
                benchmark_id=bench_meta.get("id"),
                metrics=dict(score.metrics),
                breakdowns=dict(score.breakdowns) or None,
            )
            return ev.get("id") if isinstance(ev, dict) else None
        except Exception:
            logger.exception("failed to record eval for arm %r", arm.name)
            return None

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
    "EvalResult",
    "Experiment",
    "ExperimentResult",
    "TrainFn",
    "InferenceFactory",
]

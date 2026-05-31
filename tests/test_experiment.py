"""Tests for `trajectory_labs.experiment.Experiment`.

Experiment is the OOP orchestrator that replaces the manual
``create_experiment`` → per-arm ``create_run`` → ``run_experiment`` →
``create_eval`` → ``set_conclusion`` choreography in researcher scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from trajectory_labs.benchmark import Benchmark
from trajectory_labs.config import (
    AlgorithmConfig,
    BackendConfig,
    DataConfig,
    ExperimentConfig,
    MatrixSpec,
    ModelConfig,
    RunConfig,
)
from trajectory_labs.experiment import (
    ArmResult,
    Experiment,
    ExperimentResult,
)
from trajectory_labs.protocols import RunResult


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeStore:
    """Records every store call. Returns deterministic ids."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._next_id = 0

    def _id(self) -> str:
        self._next_id += 1
        return f"id-{self._next_id}"

    def create_experiment(self, **kw: Any) -> dict:
        self.calls.append(("create_experiment", kw))
        return {"id": self._id(), **kw}

    def create_run(self, **kw: Any) -> dict:
        self.calls.append(("create_run", kw))
        return {"id": self._id(), **kw}

    def update_run(self, run_id: str, **patch: Any) -> dict:
        self.calls.append(("update_run", {"run_id": run_id, **patch}))
        return {"id": run_id, **patch}

    def update_experiment(self, experiment_id: str, **patch: Any) -> dict:
        self.calls.append(("update_experiment", {"experiment_id": experiment_id, **patch}))
        return {"id": experiment_id, **patch}

    def create_eval(self, **kw: Any) -> dict:
        self.calls.append(("create_eval", kw))
        return {"id": self._id(), **kw}


class _ScriptedInference:
    name: ClassVar[str] = "scripted"

    def __init__(self, completions: list[str]) -> None:
        self._iter = iter(completions)

    def generate(self, *, prompt: str, max_tokens: int = 256, temperature: float = 0.0,
                 stop: list[str] | None = None) -> str:
        return next(self._iter)


def _make_train_fn(
    metric_by_arm: dict[str, dict[str, float]] | None = None,
    fail_arms: set[str] | None = None,
):
    """Build a train_fn that returns canned RunResult per arm name."""
    metric_by_arm = metric_by_arm or {}
    fail_arms = fail_arms or set()

    def train_fn(cfg: ExperimentConfig) -> list[RunResult]:
        assert cfg.run is not None, "Experiment._train_arm should hand us a single-run cfg"
        name = cfg.run.name
        if name in fail_arms:
            raise RuntimeError(f"boom on arm {name!r}")
        return [RunResult(
            run_id=name,
            status="completed",
            metrics=dict(metric_by_arm.get(name, {"loss": 0.5})),
            artifacts={},
        )]

    return train_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_run() -> RunConfig:
    return RunConfig(
        name="base",
        data=DataConfig(source_kind="in_memory", rows=[{"q": "Q", "a": "A"}]),
        model=ModelConfig(name="m"),
        algorithm=AlgorithmConfig(kind="mock_sft", params={"lora_rank": 0}),
        backend=BackendConfig(kind="mock"),
    )


@pytest.fixture()
def sweep_config(base_run: RunConfig) -> ExperimentConfig:
    return ExperimentConfig(
        name="rank_sweep",
        matrix=MatrixSpec(
            base_run=base_run,
            axes={"algorithm.params.lora_rank": [1, 4, 16]},
        ),
        metadata={
            "hypothesis": "higher rank → higher reward",
            "tags": ["sft", "test"],
            "success_metric": "reward",
        },
    )


@pytest.fixture()
def single_run_config(base_run: RunConfig) -> ExperimentConfig:
    return ExperimentConfig(
        name="single",
        run=base_run,
        metadata={"hypothesis": "h", "success_metric": "reward"},
    )


@pytest.fixture()
def benchmark_dir(tmp_path: Path) -> Path:
    root = tmp_path / "toy"
    root.mkdir()
    rows = [
        {"task_id": "t1", "instruction": "Q1",
         "verifier": {"kind": "in_process", "fn_name": "exact_match", "expected": "A"},
         "metadata": {}},
        {"task_id": "t2", "instruction": "Q2",
         "verifier": {"kind": "in_process", "fn_name": "exact_match", "expected": "B"},
         "metadata": {}},
    ]
    (root / "tasks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (root / "metadata.yaml").write_text(yaml.safe_dump({"name": "toy"}))
    return root


# ---------------------------------------------------------------------------
# _iter_runs / from_yaml / construction
# ---------------------------------------------------------------------------


def test_iter_runs_single(base_run: RunConfig):
    cfg = ExperimentConfig(name="x", run=base_run)
    e = Experiment(cfg)
    runs = e._iter_runs()
    assert len(runs) == 1 and runs[0].name == "base"


def test_iter_runs_multi(base_run: RunConfig):
    other = base_run.model_copy(update={"name": "other"})
    cfg = ExperimentConfig(name="x", runs=[base_run, other])
    e = Experiment(cfg)
    runs = e._iter_runs()
    assert [r.name for r in runs] == ["base", "other"]


def test_iter_runs_matrix_expands(sweep_config: ExperimentConfig):
    e = Experiment(sweep_config)
    runs = e._iter_runs()
    assert len(runs) == 3
    assert [r.algorithm.params["lora_rank"] for r in runs] == [1, 4, 16]


def test_from_yaml_round_trip(tmp_path: Path, base_run: RunConfig):
    yaml_path = tmp_path / "config.yaml"
    cfg = ExperimentConfig(name="from_yaml", run=base_run)
    from trajectory_labs.yaml_loader import dump_yaml
    dump_yaml(cfg, path=yaml_path)
    e = Experiment.from_yaml(yaml_path)
    assert e.config.name == "from_yaml"
    assert e.config.run.name == "base"


# ---------------------------------------------------------------------------
# Happy path: sweep with store, no benchmark
# ---------------------------------------------------------------------------


def test_run_sweep_happy_path_writes_to_store(sweep_config: ExperimentConfig):
    store = _FakeStore()
    train_fn = _make_train_fn(metric_by_arm={
        "base__lora_rank1":  {"reward": 0.1},
        "base__lora_rank4":  {"reward": 0.5},
        "base__lora_rank16": {"reward": 0.3},
    })
    exp = Experiment(sweep_config, store=store, train_fn=train_fn)
    res = exp.run()

    assert isinstance(res, ExperimentResult)
    assert res.status == "completed"
    assert len(res.arms) == 3
    assert all(a.status == "completed" for a in res.arms)

    # Best score picks rank=4 (reward 0.5).
    assert res.best_arm is not None
    assert res.best_arm.name == "base__lora_rank4"
    assert res.best_score == pytest.approx(0.5)
    assert "base__lora_rank4" in res.conclusion
    assert "Best" in res.conclusion

    # Store interaction: exactly 1 create_experiment + 3 create_run +
    # 3 update_run(completed) + 1 update_experiment.
    kinds = [c[0] for c in store.calls]
    assert kinds.count("create_experiment") == 1
    assert kinds.count("create_run") == 3
    assert kinds.count("update_run") == 3
    assert kinds.count("update_experiment") == 1

    exp_call = [c for c in store.calls if c[0] == "create_experiment"][0][1]
    assert exp_call["hypothesis"] == "higher rank → higher reward"
    assert exp_call["tags"] == ["sft", "test"]

    final = [c for c in store.calls if c[0] == "update_experiment"][-1][1]
    assert final["status"] == "completed"
    assert final["best_score"] == pytest.approx(0.5)


def test_run_records_seed_and_recipe_kind_on_create_run(sweep_config: ExperimentConfig):
    store = _FakeStore()
    Experiment(sweep_config, store=store, train_fn=_make_train_fn()).run()
    create_runs = [c[1] for c in store.calls if c[0] == "create_run"]
    assert create_runs[0]["recipe_kind"] == "mock_sft"
    assert create_runs[0]["seed"] == 42
    # run_config includes the swept value
    assert create_runs[0]["run_config"]["algorithm"]["params"]["lora_rank"] == 1


# ---------------------------------------------------------------------------
# Per-arm failure isolation
# ---------------------------------------------------------------------------


def test_partial_failure_isolated_others_run(sweep_config: ExperimentConfig):
    """When one arm raises, the other arms still complete; experiment finishes."""
    store = _FakeStore()
    train_fn = _make_train_fn(
        metric_by_arm={
            "base__lora_rank1":  {"reward": 0.2},
            "base__lora_rank16": {"reward": 0.4},
        },
        fail_arms={"base__lora_rank4"},
    )
    res = Experiment(sweep_config, store=store, train_fn=train_fn).run()

    assert res.status == "completed"
    statuses = {a.name: a.status for a in res.arms}
    assert statuses == {
        "base__lora_rank1": "completed",
        "base__lora_rank4": "failed",
        "base__lora_rank16": "completed",
    }
    # Best is the highest among completed (rank 16 at 0.4).
    assert res.best_arm.name == "base__lora_rank16"
    assert res.best_score == pytest.approx(0.4)

    # The failed run got a failed-status patch.
    failed_patches = [c[1] for c in store.calls
                      if c[0] == "update_run" and c[1].get("status") == "failed"]
    assert len(failed_patches) == 1
    assert "boom" in failed_patches[0]["error_message"]

    # The other two got completed-status patches.
    completed_patches = [c[1] for c in store.calls
                         if c[0] == "update_run" and c[1].get("status") == "completed"]
    assert len(completed_patches) == 2


def test_all_arms_fail_marks_experiment_failed(sweep_config: ExperimentConfig):
    train_fn = _make_train_fn(fail_arms={
        "base__lora_rank1", "base__lora_rank4", "base__lora_rank16"
    })
    res = Experiment(sweep_config, train_fn=train_fn).run()
    assert res.status == "failed"
    assert res.best_arm is None
    assert "All" in res.conclusion and "failed" in res.conclusion


# ---------------------------------------------------------------------------
# train_fn semantics — empty results & non-completed status are failures
# ---------------------------------------------------------------------------


def test_train_fn_returning_empty_is_failure(single_run_config: ExperimentConfig):
    def empty_train(_cfg):
        return []
    res = Experiment(single_run_config, train_fn=empty_train).run()
    assert res.arms[0].status == "failed"
    assert "no results" in res.arms[0].error


def test_train_fn_returning_failed_status_is_failure(single_run_config: ExperimentConfig):
    def failed_train(_cfg):
        return [RunResult(run_id="x", status="failed", error="kaboom")]
    res = Experiment(single_run_config, train_fn=failed_train).run()
    assert res.arms[0].status == "failed"
    assert "kaboom" in res.arms[0].error


# ---------------------------------------------------------------------------
# Offline mode (store=None)
# ---------------------------------------------------------------------------


def test_no_store_runs_offline(sweep_config: ExperimentConfig):
    res = Experiment(sweep_config, train_fn=_make_train_fn(
        metric_by_arm={"base__lora_rank1": {"reward": 0.9}},
    )).run()
    # No exception, arms processed.
    assert res.experiment_id is None
    assert any(a.status == "completed" for a in res.arms)


def test_no_success_metric_skips_best_arm(base_run: RunConfig):
    cfg = ExperimentConfig(name="x", run=base_run, metadata={"hypothesis": "h"})
    res = Experiment(cfg, train_fn=_make_train_fn()).run()
    assert res.best_arm is None
    assert res.best_score is None


def test_success_metric_with_no_completed_arms_returns_none(single_run_config: ExperimentConfig):
    """`success_metric` set but every arm failed → best is None."""
    res = Experiment(single_run_config, train_fn=_make_train_fn(
        fail_arms={"base"}
    )).run()
    assert res.best_arm is None
    assert res.status == "failed"


# ---------------------------------------------------------------------------
# Benchmark integration
# ---------------------------------------------------------------------------


def test_benchmark_eval_per_arm_uses_factory(
    benchmark_dir: Path, single_run_config: ExperimentConfig
):
    bench = Benchmark.from_dir(benchmark_dir)
    # benchmark expects answers "A" then "B"
    inference_factory = lambda result, run_cfg: _ScriptedInference(["A", "B"])
    store = _FakeStore()
    res = Experiment(
        single_run_config, store=store, train_fn=_make_train_fn(),
        benchmark=bench, inference_factory=inference_factory,
    ).run()

    arm = res.arms[0]
    assert arm.eval_metrics["pass_rate"] == 1.0
    assert arm.eval_metrics["n_tasks"] == 2.0
    # create_eval was called with that arm's run id.
    eval_calls = [c[1] for c in store.calls if c[0] == "create_eval"]
    assert len(eval_calls) == 1
    assert eval_calls[0]["metrics"]["pass_rate"] == 1.0


def test_benchmark_eval_metrics_used_for_best_score(
    benchmark_dir: Path, base_run: RunConfig
):
    cfg = ExperimentConfig(
        name="bench_sweep",
        matrix=MatrixSpec(
            base_run=base_run,
            axes={"algorithm.params.lora_rank": [1, 4]},
        ),
        metadata={"success_metric": "pass_rate"},
    )
    bench = Benchmark.from_dir(benchmark_dir)

    # arm rank=1 gets 50% (one correct), arm rank=4 gets 100%
    completions = iter([["A", "X"], ["A", "B"]])
    def factory(_result, _run_cfg):
        return _ScriptedInference(next(completions))

    res = Experiment(
        cfg, train_fn=_make_train_fn(),
        benchmark=bench, inference_factory=factory,
    ).run()

    assert res.best_arm.name == "base__lora_rank4"
    assert res.best_score == pytest.approx(1.0)


def test_benchmark_from_metadata_path_is_loaded(
    benchmark_dir: Path, single_run_config: ExperimentConfig
):
    """If config.metadata.benchmark.path is set, Experiment loads it."""
    single_run_config.metadata["benchmark"] = {"path": str(benchmark_dir)}
    factory = lambda r, c: _ScriptedInference(["A", "B"])
    res = Experiment(
        single_run_config, train_fn=_make_train_fn(),
        inference_factory=factory,
    ).run()
    assert res.arms[0].eval_metrics["pass_rate"] == 1.0


def test_benchmark_loaded_but_no_factory_skips_eval(
    benchmark_dir: Path, single_run_config: ExperimentConfig
):
    """A benchmark is configured but no inference_factory was provided —
    the eval step is skipped (no exception, no eval_metrics)."""
    bench = Benchmark.from_dir(benchmark_dir)
    res = Experiment(single_run_config, train_fn=_make_train_fn(),
                     benchmark=bench).run()
    arm = res.arms[0]
    assert arm.status == "completed"
    assert arm.eval_metrics == {}


def test_benchmark_breakdown_keys_propagate(
    benchmark_dir: Path, single_run_config: ExperimentConfig
):
    single_run_config.metadata["benchmark"] = {
        "path": str(benchmark_dir),
        "breakdown_keys": ["toolkit"],  # toy tasks have empty metadata so all → __missing__
    }
    factory = lambda r, c: _ScriptedInference(["A", "B"])
    res = Experiment(single_run_config, train_fn=_make_train_fn(),
                     inference_factory=factory).run()
    assert "toolkit" in res.arms[0].eval_breakdowns


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def test_arm_result_score_prefers_eval_metric():
    arm = ArmResult(name="a", run_config=None, status="completed",  # type: ignore[arg-type]
                    metrics={"reward": 0.2}, eval_metrics={"reward": 0.8})
    assert arm.score("reward") == 0.8


def test_arm_result_score_falls_back_to_train_metric():
    arm = ArmResult(name="a", run_config=None, status="completed",  # type: ignore[arg-type]
                    metrics={"reward": 0.5})
    assert arm.score("reward") == 0.5


def test_arm_result_score_missing_returns_none():
    arm = ArmResult(name="a", run_config=None, status="completed")  # type: ignore[arg-type]
    assert arm.score("reward") is None


# ---------------------------------------------------------------------------
# Conclusion text
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Store-flake resilience — store errors don't kill the experiment
# ---------------------------------------------------------------------------


class _FlakyStore(_FakeStore):
    """Fakes a backend that throws on selected methods."""

    def __init__(self, methods_that_throw: set[str]) -> None:
        super().__init__()
        self.throws = methods_that_throw

    def create_eval(self, **kw: Any) -> dict:
        if "create_eval" in self.throws:
            raise RuntimeError("eval write failed")
        return super().create_eval(**kw)

    def update_experiment(self, experiment_id: str, **patch: Any) -> dict:
        if "update_experiment" in self.throws:
            raise RuntimeError("exp finalize failed")
        return super().update_experiment(experiment_id, **patch)

    def update_run(self, run_id: str, **patch: Any) -> dict:
        if "update_run_failed" in self.throws and patch.get("status") == "failed":
            raise RuntimeError("failed-status write failed")
        return super().update_run(run_id, **patch)


def test_store_eval_write_failure_does_not_kill_arm(
    benchmark_dir: Path, single_run_config: ExperimentConfig
):
    store = _FlakyStore({"create_eval"})
    factory = lambda r, c: _ScriptedInference(["A", "B"])
    res = Experiment(
        single_run_config, store=store, train_fn=_make_train_fn(),
        benchmark=Benchmark.from_dir(benchmark_dir),
        inference_factory=factory,
    ).run()
    # Arm still completes even though create_eval threw.
    assert res.arms[0].status == "completed"
    assert res.status == "completed"


def test_store_finalize_failure_does_not_kill_run(sweep_config: ExperimentConfig):
    store = _FlakyStore({"update_experiment"})
    res = Experiment(sweep_config, store=store, train_fn=_make_train_fn()).run()
    # All arms still complete; we only fail to *record* completion.
    assert all(a.status == "completed" for a in res.arms)


def test_store_failed_status_write_failure_swallowed(single_run_config: ExperimentConfig):
    """If marking a run failed throws on the dashboard, we still continue
    and return the arm result locally."""
    store = _FlakyStore({"update_run_failed"})
    res = Experiment(
        single_run_config, store=store,
        train_fn=_make_train_fn(fail_arms={"base"}),
    ).run()
    assert res.arms[0].status == "failed"
    # Experiment as a whole is failed (single arm failed).
    assert res.status == "failed"


# ---------------------------------------------------------------------------
# Default train_fn shim — wired up correctly
# ---------------------------------------------------------------------------


def test_default_train_fn_is_used_when_none_passed(single_run_config: ExperimentConfig):
    """Smoke: construction without train_fn doesn't bind to None."""
    exp = Experiment(single_run_config)
    assert exp.train_fn is not None


def test_default_train_fn_routes_to_runner(monkeypatch, single_run_config: ExperimentConfig):
    """The shim imports run_experiment lazily and forwards the config."""
    from trajectory_labs import experiment as exp_mod

    captured: dict = {}

    def fake_run_experiment(cfg):
        captured["cfg"] = cfg
        return [RunResult(run_id="x", status="completed", metrics={"loss": 0.0})]

    monkeypatch.setattr("trajectory_labs.runner.run_experiment", fake_run_experiment)
    out = exp_mod._default_train_fn(single_run_config)
    assert captured["cfg"] is single_run_config
    assert out[0].status == "completed"


def test_conclusion_includes_failed_arm_names(sweep_config: ExperimentConfig):
    train_fn = _make_train_fn(
        metric_by_arm={
            "base__lora_rank1": {"reward": 0.1},
            "base__lora_rank16": {"reward": 0.4},
        },
        fail_arms={"base__lora_rank4"},
    )
    res = Experiment(sweep_config, train_fn=train_fn).run()
    assert "base__lora_rank4" in res.conclusion
    assert "Failed" in res.conclusion

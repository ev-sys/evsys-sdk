"""Tests for `evsys_sdk.experiment.Experiment`.

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

from evsys_sdk.benchmark import Benchmark
from evsys_sdk.config import (
    AlgorithmConfig,
    BackendConfig,
    DataConfig,
    ExperimentConfig,
    MatrixSpec,
    ModelConfig,
    RunConfig,
)
from evsys_sdk.experiment import (
    ArmResult,
    Experiment,
    ExperimentResult,
)
from evsys_sdk.protocols import RunResult

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

    def create_group(self, experiment_id: str, name: str, *,
                     description: str | None = None) -> dict:
        self.calls.append(("create_group", {"experiment_id": experiment_id,
                                            "name": name, "description": description}))
        return {"id": self._id(), "experiment_id": experiment_id, "name": name}

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
    from evsys_sdk.yaml_loader import dump_yaml
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

    exp_call = next(c for c in store.calls if c[0] == "create_experiment")[1]
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
    def inference_factory(result, run_cfg):
        return _ScriptedInference(["A", "B"])
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
    def factory(r, c):
        return _ScriptedInference(["A", "B"])
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
    def factory(r, c):
        return _ScriptedInference(["A", "B"])
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
# Run groups (n_repeats + base_seed)
# ---------------------------------------------------------------------------


def test_n_repeats_zero_rejected(base_run: RunConfig):
    with pytest.raises(ValueError, match="n_repeats must be >= 1"):
        ExperimentConfig(name="x", run=base_run, n_repeats=0)


def test_n_repeats_default_one_no_groups(sweep_config: ExperimentConfig):
    """Backward compat: default n_repeats=1 → no create_group, no group ids."""
    store = _FakeStore()
    res = Experiment(sweep_config, store=store, train_fn=_make_train_fn()).run()
    assert [c[0] for c in store.calls].count("create_group") == 0
    assert all(a.group_id is None and a.group_name is None for a in res.arms)
    # create_run still called per arm but without a group_id
    create_runs = [c[1] for c in store.calls if c[0] == "create_run"]
    assert all(cr.get("group_id") is None for cr in create_runs)


def test_n_repeats_replicates_single_run(base_run: RunConfig):
    """n_repeats=3 with a single primary → 3 arms with seeds [42, 43, 44]."""
    cfg = ExperimentConfig(name="x", run=base_run, n_repeats=3,
                           metadata={"success_metric": "reward"})
    store = _FakeStore()
    res = Experiment(cfg, store=store, train_fn=_make_train_fn()).run()
    assert len(res.arms) == 3
    assert sorted(a.run_config.seed for a in res.arms) == [42, 43, 44]
    assert sorted(a.name for a in res.arms) == ["base__s42", "base__s43", "base__s44"]
    assert all(a.group_name == "base" for a in res.arms)
    # One group; all three arms share its id.
    create_groups = [c[1] for c in store.calls if c[0] == "create_group"]
    assert len(create_groups) == 1 and create_groups[0]["name"] == "base"
    next(c[1] for c in store.calls if c[0] == "create_group"
                    and c[1]["name"] == "base")
    # group_id_by_arm matches the created group's id
    next(c for c in store.calls if c[0] == "create_group")
    # the fake store returns the assigned id; pull from the create_run calls
    create_runs = [c[1] for c in store.calls if c[0] == "create_run"]
    assert len({cr["group_id"] for cr in create_runs}) == 1
    assert all(a.group_id == create_runs[0]["group_id"] for a in res.arms)


def test_n_repeats_with_runs_list_groups_per_entry(base_run: RunConfig):
    """runs: [A, B] with n_repeats=3 → 2 groups, 6 arms total."""
    other = base_run.model_copy(update={"name": "other"})
    cfg = ExperimentConfig(name="x", runs=[base_run, other], n_repeats=3,
                           metadata={"success_metric": "reward"})
    store = _FakeStore()
    res = Experiment(cfg, store=store, train_fn=_make_train_fn()).run()
    assert len(res.arms) == 6
    group_names_seen = {a.group_name for a in res.arms}
    assert group_names_seen == {"base", "other"}
    # 2 create_group + 6 create_run
    kinds = [c[0] for c in store.calls]
    assert kinds.count("create_group") == 2
    assert kinds.count("create_run") == 6
    # each arm's group_id matches its group_name's id
    {c[1]["name"]: None for c in store.calls if c[0] == "create_group"}
    # the fake store assigns ids in order; reverse-engineer mapping from the
    # arm-side group_id (which came from the create_group return)
    by_name = {a.group_name: a.group_id for a in res.arms}
    assert len(by_name) == 2
    assert all(by_name[a.group_name] == a.group_id for a in res.arms)


def test_n_repeats_with_matrix_groups_per_cell(sweep_config: ExperimentConfig):
    """matrix (3 cells) with n_repeats=2 → 3 groups, 6 arms total."""
    sweep_config = sweep_config.model_copy(update={"n_repeats": 2})
    store = _FakeStore()
    res = Experiment(sweep_config, store=store, train_fn=_make_train_fn()).run()
    assert len(res.arms) == 6
    group_names = {a.group_name for a in res.arms}
    assert group_names == {"base__lora_rank1", "base__lora_rank4", "base__lora_rank16"}
    kinds = [c[0] for c in store.calls]
    assert kinds.count("create_group") == 3
    assert kinds.count("create_run") == 6


def test_base_seed_overrides_primary_seed(base_run: RunConfig):
    """When base_seed is set, replicate seeds are [base_seed, base_seed+1, ...]."""
    cfg = ExperimentConfig(name="x", run=base_run, n_repeats=3, base_seed=100,
                           metadata={"success_metric": "reward"})
    res = Experiment(cfg, train_fn=_make_train_fn()).run()
    assert sorted(a.run_config.seed for a in res.arms) == [100, 101, 102]


def test_base_seed_none_uses_primary_seed(base_run: RunConfig):
    """When base_seed is None, seeds start at primary.seed."""
    primary = base_run.model_copy(update={"seed": 17})
    cfg = ExperimentConfig(name="x", run=primary, n_repeats=2,
                           metadata={"success_metric": "reward"})
    res = Experiment(cfg, train_fn=_make_train_fn()).run()
    assert sorted(a.run_config.seed for a in res.arms) == [17, 18]


def test_n_repeats_groups_run_offline(base_run: RunConfig):
    """No store: replication still produces N arms with right seeds; group_id stays None."""
    cfg = ExperimentConfig(name="x", run=base_run, n_repeats=2,
                           metadata={"success_metric": "reward"})
    res = Experiment(cfg, train_fn=_make_train_fn()).run()
    assert len(res.arms) == 2
    assert sorted(a.run_config.seed for a in res.arms) == [42, 43]
    assert all(a.group_id is None for a in res.arms)
    # group_name is still set so consumers can bucket client-side
    assert all(a.group_name == "base" for a in res.arms)


def test_n_repeats_create_group_failure_does_not_kill_arms(base_run: RunConfig):
    """If store.create_group throws, arms still run with group_id=None."""

    class _BrokenGroupStore(_FakeStore):
        def create_group(self, experiment_id, name, *, description=None):
            raise RuntimeError("group write failed")

    cfg = ExperimentConfig(name="x", run=base_run, n_repeats=2,
                           metadata={"success_metric": "reward"})
    res = Experiment(cfg, store=_BrokenGroupStore(), train_fn=_make_train_fn()).run()
    assert len(res.arms) == 2
    assert all(a.status == "completed" for a in res.arms)
    assert all(a.group_id is None for a in res.arms)
    assert all(a.group_name == "base" for a in res.arms)


# ---------------------------------------------------------------------------
# Eval auto-wrap with ChatTemplatedInference (config-driven)
# ---------------------------------------------------------------------------


class _TokenizedInference:
    """Eval client that carries a tokenizer; required by ChatTemplatedInference."""

    name: ClassVar[str] = "tokenized"

    def __init__(self, completion: str) -> None:
        self._tokenizer = _Tok()
        self._completion = completion

    def generate(self, *, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.0, stop: list[str] | None = None) -> str:
        # When wrapped, prompt will be the tokenizer's templated string.
        return self._completion


class _Tok:
    """Tokenizer stand-in that returns a sentinel string when templated."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False):
        self.calls.append({"messages": messages})
        return "<TEMPLATED>"


def test_eval_arm_wraps_client_when_chat_template_configured(
    benchmark_dir: Path, single_run_config: ExperimentConfig
):
    """metadata.benchmark.chat_template → ChatTemplatedInference wrapping."""
    single_run_config.metadata["benchmark"] = {
        "path": str(benchmark_dir),
        "chat_template": {
            "system_prompt": "You answer.",
            "user_template": "Q: {prompt}",
        },
    }
    client = _TokenizedInference("A")
    res = Experiment(
        single_run_config, train_fn=_make_train_fn(),
        inference_factory=lambda r, c: client,
    ).run()
    # The wrapped client routed every prompt through the templated path,
    # leaving its tokenizer with one call per benchmark task.
    assert res.arms[0].status == "completed"
    assert len(client._tokenizer.calls) == len(Benchmark.from_dir(benchmark_dir).tasks)
    # Message shape from the wrapper
    sample = client._tokenizer.calls[0]["messages"]
    assert sample[0] == {"role": "system", "content": "You answer."}
    assert sample[1]["role"] == "user"
    assert sample[1]["content"].startswith("Q: ")


def test_eval_arm_skips_wrap_when_chat_template_absent(
    benchmark_dir: Path, single_run_config: ExperimentConfig
):
    """No chat_template block → client is passed through unwrapped."""
    single_run_config.metadata["benchmark"] = {"path": str(benchmark_dir)}
    client = _TokenizedInference("A")
    Experiment(
        single_run_config, train_fn=_make_train_fn(),
        inference_factory=lambda r, c: client,
    ).run()
    # No template call: Benchmark.score handed the raw task.instruction
    # straight to client.generate without touching the tokenizer.
    assert client._tokenizer.calls == []


# ---------------------------------------------------------------------------
# Default inference-factory resolution (via the registry)
# ---------------------------------------------------------------------------


def test_resolve_inference_factory_user_supplied_wins(sweep_config: ExperimentConfig):
    """When the caller passes inference_factory=…, the registry default is ignored."""
    def sentinel(rr, rc):
        return object()
    exp = Experiment(sweep_config, inference_factory=sentinel)
    # Pick any run config to test against; factory is independent of run_cfg.
    run_cfg = next(iter(exp._iter_runs()))
    assert exp._resolve_inference_factory(run_cfg) is sentinel


def test_resolve_inference_factory_falls_back_to_registry(sweep_config: ExperimentConfig, monkeypatch):
    """When no inference_factory is passed, fall back to the registered default
    for the run's backend kind. We register a fake for 'mock' to verify the
    plumbing without needing the tinker module."""
    from evsys_sdk import registry

    def fake(rr, rc):
        return object()
    monkeypatch.setitem(registry._DEFAULT_INFERENCE_FACTORIES, "mock", fake)

    exp = Experiment(sweep_config)
    run_cfg = next(iter(exp._iter_runs()))
    assert exp._resolve_inference_factory(run_cfg) is fake


def test_resolve_inference_factory_none_when_no_default_registered(sweep_config: ExperimentConfig, monkeypatch):
    """No user-supplied factory + no registered default → None (eval skipped)."""
    from evsys_sdk import registry

    monkeypatch.setitem(registry._DEFAULT_INFERENCE_FACTORIES, "mock", None)
    # explicitly clear instead of None to mimic the "never registered" case
    monkeypatch.delitem(registry._DEFAULT_INFERENCE_FACTORIES, "mock", raising=False)

    exp = Experiment(sweep_config)
    run_cfg = next(iter(exp._iter_runs()))
    assert exp._resolve_inference_factory(run_cfg) is None


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
    def factory(r, c):
        return _ScriptedInference(["A", "B"])
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
    from evsys_sdk import experiment as exp_mod

    captured: dict = {}

    def fake_run_experiment(cfg):
        captured["cfg"] = cfg
        return [RunResult(run_id="x", status="completed", metrics={"loss": 0.0})]

    monkeypatch.setattr("evsys_sdk.runner.run_experiment", fake_run_experiment)
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


# ---------------------------------------------------------------------------
# Step-metric auto-forwarding
# ---------------------------------------------------------------------------


class _StoreWithLogMetrics(_FakeStore):
    def log_metrics(self, *, run_id: str, step: int, metrics: dict) -> dict:
        self.calls.append(("log_metrics", {"run_id": run_id, "step": step, "metrics": metrics}))
        return {"ok": True}





def _make_bench_dir(tmp_path: Path, name: str, expected_a: str, expected_b: str) -> Path:
    """Two-task harbor benchmark; lets each ScriptedInference produce 2 strs."""
    root = tmp_path / name
    root.mkdir()
    rows = [
        {"task_id": f"{name}-t1", "instruction": "Q1",
         "verifier": {"kind": "in_process", "fn_name": "exact_match", "expected": expected_a},
         "metadata": {}},
        {"task_id": f"{name}-t2", "instruction": "Q2",
         "verifier": {"kind": "in_process", "fn_name": "exact_match", "expected": expected_b},
         "metadata": {}},
    ]
    (root / "tasks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (root / "metadata.yaml").write_text(yaml.safe_dump({"name": name}))
    return root


def test_single_dict_benchmark_still_works(
    benchmark_dir: Path, single_run_config: ExperimentConfig
):
    """Back-compat: legacy single-dict `metadata.benchmark` keeps producing
    one EvalResult and mirrors into eval_metrics/eval_breakdowns."""
    single_run_config.metadata["benchmark"] = {"path": str(benchmark_dir)}
    res = Experiment(
        single_run_config, train_fn=_make_train_fn(),
        inference_factory=lambda r, c: _ScriptedInference(["A", "B"]),
    ).run()
    arm = res.arms[0]
    assert len(arm.evals) == 1
    assert arm.evals[0].step is None
    assert arm.evals[0].metrics["pass_rate"] == 1.0
    # Flat back-compat fields mirror the single eval.
    assert arm.eval_metrics["pass_rate"] == 1.0


def test_list_benchmark_runs_each_entry(
    tmp_path: Path, single_run_config: ExperimentConfig
):
    """List form: two post-training entries → arm.evals has 2 rows in order,
    each named and tagged per the spec."""
    val_dir = _make_bench_dir(tmp_path, "val", "A", "B")
    test_dir = _make_bench_dir(tmp_path, "test", "A", "B")
    single_run_config.metadata["benchmark"] = [
        {"name": "val_set",  "path": str(val_dir),  "tags": ["val"]},
        {"name": "test_set", "path": str(test_dir), "tags": ["test"]},
    ]
    store = _FakeStore()
    completions = iter([["A", "B"], ["A", "X"]])  # val perfect, test 1/2
    res = Experiment(
        single_run_config, store=store, train_fn=_make_train_fn(),
        inference_factory=lambda r, c: _ScriptedInference(next(completions)),
    ).run()
    arm = res.arms[0]

    assert [e.name for e in arm.evals] == ["val_set", "test_set"]
    assert arm.evals[0].metrics["pass_rate"] == 1.0
    assert arm.evals[1].metrics["pass_rate"] == 0.5
    assert arm.evals[0].tags == ["val"]
    assert arm.evals[1].tags == ["test"]
    # Each benchmark round-trips to create_eval on the dashboard.
    eval_calls = [c[1] for c in store.calls if c[0] == "create_eval"]
    assert len(eval_calls) == 2


def test_list_benchmark_primary_mirrors_first_test_tagged(
    tmp_path: Path, single_run_config: ExperimentConfig
):
    """eval_metrics mirrors the FIRST `test`-tagged entry, even if it isn't
    the first overall — that's the post-training metric researchers care about."""
    val_dir = _make_bench_dir(tmp_path, "val", "A", "B")
    test_dir = _make_bench_dir(tmp_path, "test", "A", "B")
    single_run_config.metadata["benchmark"] = [
        {"name": "val_set",  "path": str(val_dir),  "tags": ["val"]},
        {"name": "test_set", "path": str(test_dir), "tags": ["test"]},
    ]
    # val gets 100%, test gets 50% → flat field should mirror test (0.5).
    completions = iter([["A", "B"], ["A", "X"]])
    res = Experiment(
        single_run_config, train_fn=_make_train_fn(),
        inference_factory=lambda r, c: _ScriptedInference(next(completions)),
    ).run()
    arm = res.arms[0]
    assert arm.eval_metrics["pass_rate"] == 0.5


def test_arm_eval_lookup_by_name(
    tmp_path: Path, single_run_config: ExperimentConfig
):
    """arm.eval(name) returns the matching EvalResult; None for misses."""
    val_dir = _make_bench_dir(tmp_path, "val", "A", "B")
    single_run_config.metadata["benchmark"] = [
        {"name": "v", "path": str(val_dir), "tags": ["val"]},
    ]
    res = Experiment(
        single_run_config, train_fn=_make_train_fn(),
        inference_factory=lambda r, c: _ScriptedInference(["A", "B"]),
    ).run()
    arm = res.arms[0]
    assert arm.eval("v") is not None
    assert arm.eval("v").metrics["pass_rate"] == 1.0
    assert arm.eval("does_not_exist") is None


def test_dotted_success_metric_picks_named_benchmark(
    tmp_path: Path, base_run: RunConfig
):
    """`success_metric: <bench>.pass_rate` ranks arms by the named eval,
    not by the (mirrored) first-test default."""
    val_dir = _make_bench_dir(tmp_path, "val", "A", "B")
    test_dir = _make_bench_dir(tmp_path, "test", "A", "B")
    cfg = ExperimentConfig(
        name="dotted",
        matrix=MatrixSpec(
            base_run=base_run,
            axes={"algorithm.params.lora_rank": [1, 4]},
        ),
        metadata={
            "benchmark": [
                {"name": "val_set",  "path": str(val_dir),  "tags": ["val"]},
                {"name": "test_set", "path": str(test_dir), "tags": ["test"]},
            ],
            "success_metric": "test_set.pass_rate",
        },
    )
    # rank=1 gets 100% on val, 0% on test. rank=4 gets 50% on val, 100% on test.
    # If we ranked by val, rank=1 wins; by test, rank=4 wins.
    completions = iter([
        ["A", "B"], ["X", "Y"],   # arm 1: val=100%, test=0%
        ["A", "X"], ["A", "B"],   # arm 4: val=50%,  test=100%
    ])
    res = Experiment(
        cfg, train_fn=_make_train_fn(),
        inference_factory=lambda r, c: _ScriptedInference(next(completions)),
    ).run()
    assert res.best_arm.name == "base__lora_rank4"
    assert res.best_score == pytest.approx(1.0)


def test_run_every_entries_are_skipped_post_training(
    tmp_path: Path, single_run_config: ExperimentConfig
):
    """Entries with ``run_every`` are picked up by the algorithm composer
    (via ``training.evaluators.build_in_loop_evaluators``) and scored
    during training, NOT here in ``_eval_arm``. Confirm ``arm.evals`` stays
    empty when the only configured benchmark is an in-loop one."""
    val_dir = _make_bench_dir(tmp_path, "val", "A", "B")
    single_run_config.metadata["benchmark"] = [
        {"name": "val_set", "path": str(val_dir), "tags": ["val"], "run_every": 100},
    ]
    res = Experiment(
        single_run_config, train_fn=_make_train_fn(),
        inference_factory=lambda r, c: _ScriptedInference(["A", "B"]),
    ).run()
    # in-loop entry is the algorithm composer's job; _eval_arm does NOT
    # attach an EvalResult for it.
    assert res.arms[0].evals == []


def test_evsys_logger_warns_when_configured_without_key(monkeypatch, caplog, single_run_config):
    """An evsys_logger with no store + no EVSYS_API_KEY warns up front (so the
    user knows dashboard logging is disabled before the run, not mid-run)."""
    import logging
    from evsys_sdk.config import CallbackSpec

    monkeypatch.delenv("EVSYS_API_KEY", raising=False)
    monkeypatch.setattr(logging.getLogger("evsys_sdk"), "propagate", True)  # SDK logger is propagate=False
    cfg = single_run_config.model_copy(update={"callbacks": [CallbackSpec(kind="evsys_logger")]})
    with caplog.at_level(logging.WARNING, logger="evsys_sdk.experiment"):
        Experiment(cfg)
    assert any("EVSYS_API_KEY is not set" in r.getMessage() for r in caplog.records)


def test_no_warning_when_evsys_logger_has_key(monkeypatch, caplog, single_run_config):
    import logging
    from evsys_sdk.config import CallbackSpec

    monkeypatch.setenv("EVSYS_API_KEY", "sk-test")
    monkeypatch.setattr(logging.getLogger("evsys_sdk"), "propagate", True)
    cfg = single_run_config.model_copy(update={"callbacks": [CallbackSpec(kind="evsys_logger")]})
    with caplog.at_level(logging.WARNING, logger="evsys_sdk.experiment"):
        Experiment(cfg)
    assert not any("EVSYS_API_KEY is not set" in r.getMessage() for r in caplog.records)

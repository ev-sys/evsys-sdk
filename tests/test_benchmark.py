"""Tests for `evsys_sdk.benchmark.Benchmark`.

The Benchmark class loads harbor-format eval suites from disk
(`tasks.jsonl` + optional `metadata.yaml`) and scores a model against
them via the SDK's in-process verifier-fn registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from evsys_sdk.benchmark import (
    Benchmark,
    BenchmarkScore,
    BenchmarkTaskResult,
)
from evsys_sdk.data_types import HarborTask, InProcessVerifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _row(task_id: str, instruction: str, fn_name: str, expected: Any, **meta) -> dict:
    return {
        "task_id": task_id,
        "instruction": instruction,
        "verifier": {"kind": "in_process", "fn_name": fn_name, "expected": expected, "params": {}},
        "metadata": meta,
    }


@pytest.fixture()
def benchmark_dir(tmp_path: Path) -> Path:
    """A 5-task benchmark dir with mixed exact_match + contains verifiers."""
    root = tmp_path / "toy"
    root.mkdir()
    rows = [
        _row("t1", "Q1", "exact_match", "A", toolkit="SLACK"),
        _row("t2", "Q2", "exact_match", "B", toolkit="SLACK"),
        _row("t3", "Q3", "exact_match", "C", toolkit="GITHUB"),
        _row("t4", "Q4", "contains", "needle", toolkit="GITHUB"),
        _row("t5", "Q5", "exact_match", "E", toolkit="OUTLOOK"),
    ]
    (root / "tasks.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (root / "metadata.yaml").write_text(
        yaml.safe_dump({"name": "toy", "version": "v1", "description": "fixture"})
    )
    return root


class _ScriptedInference:
    """Returns a pre-baked completion per prompt."""

    name: ClassVar[str] = "scripted"

    def __init__(self, completions: list[str]) -> None:
        self._iter = iter(completions)
        self.calls: list[dict] = []

    def generate(
        self,
        *,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens,
                           "temperature": temperature, "stop": stop})
        return next(self._iter)


# ---------------------------------------------------------------------------
# from_dir
# ---------------------------------------------------------------------------


def test_from_dir_round_trip(benchmark_dir: Path):
    bench = Benchmark.from_dir(benchmark_dir)
    assert bench.name == "toy"  # from metadata.yaml
    assert len(bench.tasks) == 5
    assert bench.metadata == {"name": "toy", "version": "v1", "description": "fixture"}
    assert bench.root == benchmark_dir

    t1, *_ = bench.tasks
    assert isinstance(t1, HarborTask)
    assert t1.task_id == "t1"
    assert t1.instruction == "Q1"
    assert isinstance(t1.verifier, InProcessVerifier)
    assert t1.verifier.fn_name == "exact_match"
    assert t1.verifier.expected == "A"


def test_from_dir_no_metadata_yaml_uses_dir_name(tmp_path: Path):
    root = tmp_path / "no_meta"
    root.mkdir()
    (root / "tasks.jsonl").write_text(json.dumps(_row("a", "?", "exact_match", "x")) + "\n")
    bench = Benchmark.from_dir(root)
    assert bench.name == "no_meta"
    assert bench.metadata == {}
    assert len(bench.tasks) == 1


def test_from_dir_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="benchmark dir not found"):
        Benchmark.from_dir(tmp_path / "nope")


def test_from_dir_missing_tasks_jsonl(tmp_path: Path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match=r"missing tasks\.jsonl"):
        Benchmark.from_dir(root)


def test_from_dir_malformed_json_line(tmp_path: Path):
    root = tmp_path / "bad_json"
    root.mkdir()
    (root / "tasks.jsonl").write_text("not json\n")
    with pytest.raises(ValueError, match="malformed json"):
        Benchmark.from_dir(root)


def test_from_dir_unknown_verifier_kind(tmp_path: Path):
    root = tmp_path / "bad_verifier"
    root.mkdir()
    (root / "tasks.jsonl").write_text(json.dumps({
        "task_id": "x", "instruction": "?",
        "verifier": {"kind": "bogus"}, "metadata": {},
    }) + "\n")
    with pytest.raises(ValueError, match="unknown verifier kind"):
        Benchmark.from_dir(root)


def test_from_dir_blank_lines_ignored(tmp_path: Path):
    root = tmp_path / "blanks"
    root.mkdir()
    (root / "tasks.jsonl").write_text(
        "\n"  # leading blank
        + json.dumps(_row("a", "?", "exact_match", "x")) + "\n"
        + "\n"  # middle blank
        + json.dumps(_row("b", "?", "exact_match", "y")) + "\n"
    )
    bench = Benchmark.from_dir(root)
    assert [t.task_id for t in bench.tasks] == ["a", "b"]


def test_from_dir_metadata_yaml_not_a_mapping(tmp_path: Path):
    root = tmp_path / "bad_meta"
    root.mkdir()
    (root / "tasks.jsonl").write_text(json.dumps(_row("a", "?", "exact_match", "x")) + "\n")
    (root / "metadata.yaml").write_text("- a\n- b\n")  # a list, not a dict
    with pytest.raises(ValueError, match="must be a mapping"):
        Benchmark.from_dir(root)


def test_from_dir_empty_metadata_yaml_ok(tmp_path: Path):
    root = tmp_path / "empty_meta"
    root.mkdir()
    (root / "tasks.jsonl").write_text(json.dumps(_row("a", "?", "exact_match", "x")) + "\n")
    (root / "metadata.yaml").write_text("")  # empty file → safe_load returns None
    bench = Benchmark.from_dir(root)
    assert bench.metadata == {}


# ---------------------------------------------------------------------------
# from_iterable
# ---------------------------------------------------------------------------


def test_from_iterable_dicts():
    bench = Benchmark.from_iterable(
        "mem", [_row("a", "?", "exact_match", "x")], metadata={"src": "test"}
    )
    assert bench.name == "mem"
    assert bench.metadata == {"src": "test"}
    assert bench.root is None
    assert len(bench.tasks) == 1


def test_from_iterable_already_typed():
    task = HarborTask(task_id="a", instruction="?",
                     verifier=InProcessVerifier(fn_name="exact_match", expected="x"),
                     metadata={})
    bench = Benchmark.from_iterable("mem", [task])
    assert bench.tasks[0] is task


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


def test_score_aggregates(benchmark_dir: Path):
    bench = Benchmark.from_dir(benchmark_dir)
    # Mock model returns: t1=A (pass), t2=Z (fail), t3=C (pass),
    #                     t4="this has a needle inside" (pass), t5=E (pass)
    client = _ScriptedInference(["A", "Z", "C", "this has a needle inside", "E"])
    score = bench.score(client)

    assert isinstance(score, BenchmarkScore)
    assert score.metrics["n_tasks"] == 5.0
    assert score.metrics["pass_rate"] == pytest.approx(4 / 5)
    assert score.metrics["mean_reward"] == pytest.approx(4 / 5)
    assert [r.reward for r in score.per_task] == [1.0, 0.0, 1.0, 1.0, 1.0]
    # per-task carries the model output + expected for downstream prediction rows
    assert score.per_task[0].model_output == "A"
    assert score.per_task[0].expected == "A"


def test_score_num_samples_drives_pass_at_k(benchmark_dir: Path):
    bench = Benchmark.from_dir(benchmark_dir)
    # 2 tasks (limit=2), 3 samples each. t1 expects "A", t2 expects "B".
    #   t1 samples: A, A, X  -> any pass (pass@3), not all (pass^3)
    #   t2 samples: B, B, B  -> all pass
    client = _ScriptedInference(["A", "A", "X", "B", "B", "B"])
    score = bench.score(
        client, limit=2, num_samples=3, metrics=["pass@3", "pass^3", "avg"]
    )
    assert score.metrics["n_tasks"] == 2.0
    assert score.metrics["pass@3"] == pytest.approx(1.0)   # both have ≥1 pass
    assert score.metrics["pass^3"] == pytest.approx(0.5)   # only t2 all-pass
    assert score.metrics["avg"] == pytest.approx((2 / 3 + 1.0) / 2)
    # 3 generations per task were issued.
    assert len(client.calls) == 6


def test_score_empty_benchmark(tmp_path: Path):
    root = tmp_path / "empty"
    root.mkdir()
    (root / "tasks.jsonl").write_text("")
    bench = Benchmark.from_dir(root)
    score = bench.score(_ScriptedInference([]))
    assert score.metrics == {"mean_reward": 0.0, "pass_rate": 0.0, "n_tasks": 0.0}
    assert score.per_task == []


def test_score_with_breakdown_keys(benchmark_dir: Path):
    bench = Benchmark.from_dir(benchmark_dir)
    # SLACK: 1/2 pass, GITHUB: 2/2 pass, OUTLOOK: 1/1 pass
    client = _ScriptedInference(["A", "Z", "C", "needle here", "E"])
    score = bench.score(client, breakdown_keys=["toolkit"])

    tk = score.breakdowns["toolkit"]
    assert tk["SLACK"]["n"] == 2.0
    assert tk["SLACK"]["pass_rate"] == pytest.approx(0.5)
    assert tk["GITHUB"]["n"] == 2.0
    assert tk["GITHUB"]["pass_rate"] == 1.0
    assert tk["OUTLOOK"]["n"] == 1.0
    assert tk["OUTLOOK"]["pass_rate"] == 1.0


def test_score_breakdown_missing_field(tmp_path: Path):
    """Tasks without the breakdown field land in a `__missing__` bucket."""
    root = tmp_path / "mixed"
    root.mkdir()
    rows = [
        _row("a", "?", "exact_match", "x", toolkit="A"),
        _row("b", "?", "exact_match", "x"),  # no toolkit
    ]
    (root / "tasks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    bench = Benchmark.from_dir(root)
    score = bench.score(_ScriptedInference(["x", "x"]), breakdown_keys=["toolkit"])
    tk = score.breakdowns["toolkit"]
    assert set(tk) == {"A", "__missing__"}


def test_score_with_prompt_builder(benchmark_dir: Path):
    bench = Benchmark.from_dir(benchmark_dir)
    client = _ScriptedInference(["A"] * 5)
    bench.score(client, prompt_builder=lambda t: f"SYSTEM:\n{t.instruction}")
    assert client.calls[0]["prompt"] == "SYSTEM:\nQ1"


def test_score_passes_generation_params(benchmark_dir: Path):
    bench = Benchmark.from_dir(benchmark_dir)
    client = _ScriptedInference(["A"] * 5)
    bench.score(client, max_tokens=42, temperature=0.7, stop=["</end>"])
    assert client.calls[0]["max_tokens"] == 42
    assert client.calls[0]["temperature"] == 0.7
    assert client.calls[0]["stop"] == ["</end>"]


def test_score_limit_caps_n_tasks(benchmark_dir: Path):
    """`limit=N` scores only the first N tasks (deterministic order)."""
    bench = Benchmark.from_dir(benchmark_dir)
    assert len(bench.tasks) >= 3, "fixture should expose at least 3 tasks"
    client = _ScriptedInference(["A"] * 10)
    out = bench.score(client, limit=2)
    assert out.metrics["n_tasks"] == 2.0
    assert len(out.per_task) == 2
    assert [r.task_id for r in out.per_task] == [t.task_id for t in bench.tasks[:2]]
    # client only invoked twice — the rest of self.tasks weren't sampled
    assert len(client.calls) == 2


def test_score_limit_none_scores_all(benchmark_dir: Path):
    bench = Benchmark.from_dir(benchmark_dir)
    out = bench.score(_ScriptedInference(["A"] * 10), limit=None)
    assert out.metrics["n_tasks"] == float(len(bench.tasks))


def test_score_limit_zero_yields_empty(benchmark_dir: Path):
    """`limit=0` is a no-op score: no client calls, empty per_task."""
    bench = Benchmark.from_dir(benchmark_dir)
    client = _ScriptedInference([])
    out = bench.score(client, limit=0)
    assert out.metrics["n_tasks"] == 0.0
    assert out.per_task == []
    assert client.calls == []


def test_score_limit_larger_than_n_tasks_scores_all(benchmark_dir: Path):
    bench = Benchmark.from_dir(benchmark_dir)
    out = bench.score(_ScriptedInference(["A"] * 10), limit=999)
    assert out.metrics["n_tasks"] == float(len(bench.tasks))


def test_score_unknown_fn_name(tmp_path: Path):
    root = tmp_path / "unknown_fn"
    root.mkdir()
    (root / "tasks.jsonl").write_text(json.dumps({
        "task_id": "a", "instruction": "?",
        "verifier": {"kind": "in_process", "fn_name": "no_such_fn", "expected": "x"},
        "metadata": {},
    }) + "\n")
    bench = Benchmark.from_dir(root)
    with pytest.raises(ValueError, match="Unknown verifier fn"):
        bench.score(_ScriptedInference(["x"]))


def test_score_e2b_verifier_not_implemented():
    bench = Benchmark.from_iterable("e2b", [{
        "task_id": "a", "instruction": "?",
        "verifier": {"kind": "e2b", "dockerfile": "FROM x", "test_sh": "pytest"},
        "metadata": {},
    }])
    with pytest.raises(NotImplementedError, match="E2BVerifier"):
        bench.score(_ScriptedInference(["out"]))


def test_score_llm_judge_not_implemented():
    bench = Benchmark.from_iterable("judge", [{
        "task_id": "a", "instruction": "?",
        "verifier": {"kind": "llm_judge", "judge_model": "claude", "rubric": "..."},
        "metadata": {},
    }])
    with pytest.raises(NotImplementedError, match="LLMJudgeVerifier"):
        bench.score(_ScriptedInference(["out"]))


def test_score_completely_unknown_verifier_object_raises_type_error():
    """Defensive: if someone hand-constructs a HarborTask with an alien verifier
    object that's none of the three known payload types, score() complains
    clearly rather than silently returning a 0."""
    task = HarborTask(task_id="weird", instruction="?",
                     verifier=object(),  # type: ignore[arg-type]
                     metadata={})
    bench = Benchmark.from_iterable("alien", [task])
    with pytest.raises(TypeError, match="unknown verifier type"):
        bench.score(_ScriptedInference(["x"]))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


def test_task_result_dataclass_shape():
    r = BenchmarkTaskResult(task_id="a", instruction="?", model_output="o",
                            expected="x", reward=0.5, metadata={"k": "v"})
    assert r.reward == 0.5
    assert r.metadata == {"k": "v"}


# ---------------------------------------------------------------------------
# score_via_harbor (harbor engine; rollout mocked at run_harbor_rollouts)
# ---------------------------------------------------------------------------


def test_score_via_harbor_builds_benchmarkscore_with_rollouts(monkeypatch, tmp_path):
    import asyncio

    # score_via_harbor pulls in evsys_sdk.training (loop/data_processing/...), which
    # imports tinker at module level. CI runs without the [tinker] extra, so skip
    # there — consistent with every other tinker-dependent test in the suite.
    pytest.importorskip("tinker")

    from evsys_sdk.training.harbor_eval import eval_predictions
    from evsys_sdk.training.trajectory import Trajectory, TrajectoryGroup, Turn

    bench = Benchmark.from_iterable("b", [
        _row("t0", "P0", "exact_match", "x", toolkit="A"),
        _row("t1", "P1", "exact_match", "x", toolkit="B"),
    ])

    async def _fake_rollouts(items, **kwargs):
        # The runner is adapter-aware, so (mocked here) it receives the
        # benchmark's HarborTasks directly. 2 samples/task; t0 both pass, t1 both fail.
        per = {"t0": [1.0, 1.0], "t1": [0.0, 0.0]}
        return [
            TrajectoryGroup(trajectories=[
                Trajectory(
                    turns=[Turn(prompt_tokens=[1], completion_tokens=[2, 3])],
                    reward=r,
                )
                for r in per[task.task_id]
            ])
            for task in items
        ]

    monkeypatch.setattr(
        "evsys_sdk.training.harbor_engine.run_harbor_rollouts", _fake_rollouts
    )

    score = asyncio.run(bench.score_via_harbor(
        model_name="m", model_path="ckpt", workspace_dir=tmp_path,
        num_samples=2, breakdown_keys=["toolkit"], metrics=["mean_reward", "pass_rate"],
    ))

    assert isinstance(score, BenchmarkScore)
    assert score.metrics["n_tasks"] == 2.0
    assert score.metrics["mean_reward"] == 0.5          # (1.0 + 0.0) per-task means
    # per_task: one row per task (mean reward), in task order
    assert [r.task_id for r in score.per_task] == ["t0", "t1"]
    assert score.per_task[0].reward == 1.0 and score.per_task[1].reward == 0.0
    # rollouts carries the raw per-(task, sample) groups
    assert len(score.rollouts) == 2
    assert all(len(g.trajectories) == 2 for g in score.rollouts)
    # breakdown by toolkit metadata
    assert score.breakdowns["toolkit"]["A"]["mean_reward"] == 1.0
    assert score.breakdowns["toolkit"]["B"]["mean_reward"] == 0.0
    # eval_predictions builds one row per (task, sample) from score.rollouts
    rows = eval_predictions(bench.tasks, score.rollouts, eval_id="e1")
    assert len(rows) == 4
    assert rows[0]["completion_token_ids"] == [2, 3]
    assert all(r["eval_id"] == "e1" and r["kind"] == "eval" for r in rows)

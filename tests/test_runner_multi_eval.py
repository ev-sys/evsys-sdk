"""Multiple evals per training run + best-effort persistence routing.

Uses the mock backend/algorithm and an in-test log store that captures the
`log_eval` calls, so we can assert evals are run, namespaced and persisted
without touching a real backend.
"""

from __future__ import annotations

from pathlib import Path

from trajectory_labs import (
    AlgorithmConfig,
    BackendConfig,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    InferenceSpec,
    LogStoreSpec,
    MetricSpec,
    ModelConfig,
    RunConfig,
    TransformSpec,
    run_experiment,
)
from trajectory_labs.registry import register_log_store

# Captures eval persistence calls across runs in this test module.
CAPTURED_EVALS: list[dict] = []


@register_log_store("capture")
class _CaptureLogStore:
    name = "capture"
    Config = type("C", (), {})

    def __init__(self, **_: object) -> None:
        pass

    def log_scalar(self, key, value, step): ...
    def log_metrics(self, metrics, step): ...
    def log_hyperparams(self, params): ...
    def log_artifact(self, name, path, *, kind="file"): ...
    def close(self): ...

    def log_eval(self, *, name, metrics, step=None, benchmark_id=None, model_ref=None):
        CAPTURED_EVALS.append(
            {"name": name, "metrics": dict(metrics), "step": step,
             "benchmark_id": benchmark_id}
        )


def _answer(slug: str) -> str:
    return f"<think>t</think>\n<answer>{slug}</answer>"


def _cfg(tmp_path: Path, rows: list[dict]) -> ExperimentConfig:
    return ExperimentConfig(
        name="multi_eval",
        output_dir=str(tmp_path / "out"),
        log_store=LogStoreSpec(kind="capture"),
        run=RunConfig(
            name="r",
            data=DataConfig(source_kind="in_memory", rows=rows,
                            transforms=[TransformSpec(kind="identity")]),
            model=ModelConfig(name="tiny/fake"),
            algorithm=AlgorithmConfig(kind="mock_sft", params={"num_epochs": 1, "batch_size": 1}),
            backend=BackendConfig(kind="mock"),
            # Primary eval: always answers OUTLOOK_CREATE_CONTACT -> 1/3 exact match.
            eval=EvalConfig(
                metrics=[MetricSpec(kind="exact_match")],
                inference=InferenceSpec(kind="mock", params={"template": _answer("OUTLOOK_CREATE_CONTACT")}),
            ),
            # Two additional named evals against different benchmarks.
            evals=[
                EvalConfig(
                    name="slack_topic", benchmark_id="bench-1",
                    metrics=[MetricSpec(kind="exact_match")],
                    inference=InferenceSpec(kind="mock", params={"template": _answer("SLACK_SET_THE_TOPIC_OF_A_CONVERSATION")}),
                ),
                EvalConfig(
                    name="toolkits",
                    metrics=[MetricSpec(kind="toolkit_match")],
                    inference=InferenceSpec(kind="mock", params={"template": _answer("SLACK_X")}),
                ),
            ],
        ),
    )


def test_runner_runs_and_namespaces_multiple_evals(tmp_path: Path, sample_rows):
    CAPTURED_EVALS.clear()
    results = run_experiment(_cfg(tmp_path, sample_rows))
    r = results[0]
    assert r.status == "completed"

    # Primary eval keeps the historical un-named key.
    assert "eval/exact_match" in r.metrics
    assert 0.30 < r.metrics["eval/exact_match"] < 0.35
    # Named evals are namespaced by eval name.
    assert "eval/slack_topic/exact_match" in r.metrics
    assert "eval/toolkits/toolkit_match" in r.metrics
    # slack_topic answers the 3rd row's slug -> 1/3.
    assert 0.30 < r.metrics["eval/slack_topic/exact_match"] < 0.35
    # SLACK_X startswith "SLACK_" matches both SLACK rows -> 2/3.
    assert 0.60 < r.metrics["eval/toolkits/toolkit_match"] < 0.70


def test_each_eval_is_persisted_once(tmp_path: Path, sample_rows):
    CAPTURED_EVALS.clear()
    run_experiment(_cfg(tmp_path, sample_rows))
    # Three evals -> three persisted rows, with names + benchmark ids preserved.
    names = sorted(e["name"] for e in CAPTURED_EVALS)
    assert names == ["default", "slack_topic", "toolkits"]
    by_name = {e["name"]: e for e in CAPTURED_EVALS}
    assert by_name["slack_topic"]["benchmark_id"] == "bench-1"
    assert by_name["default"]["benchmark_id"] is None
    # Metrics passed to persistence are un-prefixed (raw metric kind keys).
    assert "exact_match" in by_name["default"]["metrics"]
    assert all(e["step"] is not None for e in CAPTURED_EVALS)

"""Tests for the validation-dataset feature.

Covers: the typed ``run.validation`` config block, ``upload_validation_dataset``
(+ ``evsys validation upload`` CLI), the pure metrics.py scoring core
(``compute_validation_metrics``), the tinker evaluator-builder bridge, the
runner's validation loader, and split-aware step-metric forwarding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from evsys_sdk.cli import main as cli_main
from evsys_sdk.config import MetricSpec, RunConfig, ValidationConfig
from evsys_sdk.data_types import HarborTask, InProcessVerifier
from evsys_sdk.algorithms.validation_evaluator import (
    build_validation_evaluator_builders,
    compute_validation_metrics,
)
from evsys_sdk.step_metrics import forward_step_metrics
from evsys_sdk.validation_upload import (
    VALIDATION_FORMAT,
    ValidationUploadResult,
    upload_validation_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _row(task_id: str, expected: str, toolkit: str = "") -> dict:
    return {
        "task_id": task_id,
        "instruction": f"Q-{task_id}",
        "verifier": {"kind": "in_process", "fn_name": "exact_match",
                     "expected": expected, "params": {}},
        "metadata": {"toolkit": toolkit} if toolkit else {},
    }


@pytest.fixture()
def val_dir(tmp_path: Path) -> Path:
    root = tmp_path / "toy_val"
    root.mkdir()
    (root / "tasks.jsonl").write_text(
        "\n".join(json.dumps(_row(t, "A")) for t in ["t1", "t2", "t3"]) + "\n"
    )
    (root / "metadata.yaml").write_text(yaml.safe_dump({"name": "toy_val"}))
    return root


class _FakeStore:
    """Tracks create/add/list for validation datasets (mirrors benchmark fake)."""

    def __init__(self) -> None:
        self.records: list[dict] = []
        self.rows_by_id: dict[str, list[dict]] = {}
        self._next_id = 0

    def _id(self) -> str:
        self._next_id += 1
        return f"val-{self._next_id}"

    def list_validation_datasets(self, project_id: str | None = None) -> list[dict]:
        return list(self.records)

    def create_validation_dataset(self, **kw: Any) -> dict:
        record = {"id": self._id(), **kw}
        self.records.append(record)
        return record

    def add_validation_dataset_rows(self, validation_dataset_id: str, rows: list[dict],
                                    *, start_idx: int = 0) -> list[dict]:
        self.rows_by_id.setdefault(validation_dataset_id, []).extend(rows)
        return rows


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _run_dict(**validation: Any) -> dict:
    d = {
        "name": "r1",
        "data": {"source_kind": "in_memory", "rows": [{"x": 1}]},
        "model": {"name": "tiny/fake"},
        "algorithm": {"kind": "mock_sft"},
        "backend": {"kind": "mock"},
    }
    if validation:
        d["validation"] = validation
    return d


def test_validation_block_validates():
    run = RunConfig.model_validate(_run_dict(
        dataset_id="val-1", eval_for_every=10,
        metrics=[{"kind": "exact_match"}], max_tokens=128,
    ))
    assert run.validation.dataset_id == "val-1"
    assert run.validation.eval_for_every == 10
    assert run.validation.metrics[0].kind == "exact_match"
    assert run.validation.max_tokens == 128


def test_validation_defaults_to_disabled_no_cadence():
    run = RunConfig.model_validate(_run_dict())
    assert isinstance(run.validation, ValidationConfig)
    assert run.validation.eval_for_every == 0
    assert run.validation.metrics == []


def test_validation_rejects_unknown_key():
    with pytest.raises(Exception):
        RunConfig.model_validate(_run_dict(eval_evry=10))  # typo'd key


# ---------------------------------------------------------------------------
# Pure metrics.py scoring core
# ---------------------------------------------------------------------------


def _tasks() -> list[HarborTask]:
    return [
        HarborTask(task_id="t1", instruction="q1",
                   verifier=InProcessVerifier(fn_name="x", expected="paris")),
        HarborTask(task_id="t2", instruction="q2",
                   verifier=InProcessVerifier(fn_name="x", expected="berlin")),
    ]


def test_compute_validation_metrics_exact_match():
    completions = ["the answer is <answer>paris</answer>", "<answer>rome</answer>"]
    m = compute_validation_metrics(_tasks(), completions, [MetricSpec(kind="exact_match")])
    assert m == {"val/exact_match": 0.5}


def test_compute_validation_metrics_all_correct():
    completions = ["<answer>paris</answer>", "<answer>berlin</answer>"]
    m = compute_validation_metrics(_tasks(), completions, [MetricSpec(kind="exact_match")])
    assert m == {"val/exact_match": 1.0}


def test_compute_validation_metrics_bad_metric_skipped():
    # Unknown metric kind is logged + skipped, not raised.
    m = compute_validation_metrics(_tasks(), ["x", "y"], [MetricSpec(kind="no_such_metric")])
    assert m == {}


# ---------------------------------------------------------------------------
# Evaluator-builder bridge
# ---------------------------------------------------------------------------


class _Ctx:
    def __init__(self, extras: dict) -> None:
        self.extras = extras
        self.log_store = None


def test_builder_no_validation_returns_empty():
    builders, eval_every = build_validation_evaluator_builders(_Ctx({}), tokenizer=None)
    assert builders == []
    assert eval_every is None


def test_builder_with_validation_returns_cadence():
    extras = {"validation": {
        "tasks": _tasks(),
        "eval_for_every": 5,
        "metric_specs": [MetricSpec(kind="exact_match")],
        "gen": {"max_tokens": 64, "temperature": 0.0},
        "dataset_id": "val-1",
    }}
    builders, eval_every = build_validation_evaluator_builders(_Ctx(extras), tokenizer=object())
    assert eval_every == 5
    assert len(builders) == 1
    evaluator = builders[0]()  # building must not require tinker
    assert evaluator.eval_for_every == 5
    assert len(evaluator.tasks) == 2


# ---------------------------------------------------------------------------
# upload_validation_dataset
# ---------------------------------------------------------------------------


def test_first_upload_creates_v1(val_dir: Path):
    store = _FakeStore()
    result = upload_validation_dataset(store, val_dir)
    assert isinstance(result, ValidationUploadResult)
    assert result.status == "created"
    assert result.version == 1
    assert result.n_tasks == 3
    assert result.name == "toy_val"
    rec = store.records[0]
    assert rec["format"] == VALIDATION_FORMAT
    assert rec["metadata"]["content_hash"] == result.content_hash
    assert len(store.rows_by_id[result.validation_dataset_id]) == 3


def test_idempotent_reupload_unchanged(val_dir: Path):
    store = _FakeStore()
    first = upload_validation_dataset(store, val_dir)
    second = upload_validation_dataset(store, val_dir)
    assert second.status == "unchanged"
    assert second.validation_dataset_id == first.validation_dataset_id
    assert len(store.records) == 1


def test_changed_content_new_version(val_dir: Path):
    store = _FakeStore()
    first = upload_validation_dataset(store, val_dir)
    (val_dir / "tasks.jsonl").write_text(
        "\n".join(json.dumps(_row(t, "B")) for t in ["t1", "t2"]) + "\n"
    )
    second = upload_validation_dataset(store, val_dir)
    assert second.status == "updated"
    assert second.version == 2
    assert second.validation_dataset_id != first.validation_dataset_id


def test_cli_validation_upload_happy(val_dir: Path, monkeypatch, capsys):
    fake = _FakeStore()
    monkeypatch.setattr("evsys_sdk.store.EvsysStore", lambda *a, **kw: fake)
    rc = cli_main(["validation", "upload", str(val_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out.split("\n\n")[0])
    assert payload["status"] == "created"
    assert payload["n_tasks"] == 3
    assert "run.validation.dataset_id" in out


# ---------------------------------------------------------------------------
# Runner loader
# ---------------------------------------------------------------------------


def test_load_validation_from_path(val_dir: Path):
    from evsys_sdk.runner import _load_validation

    run = RunConfig.model_validate(_run_dict(
        path=str(val_dir), eval_for_every=2, metrics=[{"kind": "exact_match"}],
    ))
    out = _load_validation(run)
    assert out is not None
    assert len(out["tasks"]) == 3
    assert out["eval_for_every"] == 2
    assert out["metric_specs"][0].kind == "exact_match"


def test_load_validation_disabled_returns_none(val_dir: Path):
    from evsys_sdk.runner import _load_validation

    # eval_for_every defaults to 0 → disabled
    run = RunConfig.model_validate(_run_dict(path=str(val_dir),
                                             metrics=[{"kind": "exact_match"}]))
    assert _load_validation(run) is None


def test_load_validation_no_metrics_returns_none(val_dir: Path):
    from evsys_sdk.runner import _load_validation

    run = RunConfig.model_validate(_run_dict(path=str(val_dir), eval_for_every=2))
    assert _load_validation(run) is None


def test_load_validation_n_samples_caps(val_dir: Path):
    from evsys_sdk.runner import _load_validation

    run = RunConfig.model_validate(_run_dict(
        path=str(val_dir), eval_for_every=2, n_samples=1,
        metrics=[{"kind": "exact_match"}],
    ))
    out = _load_validation(run)
    assert out is not None and len(out["tasks"]) == 1


def test_validation_accepts_dataset_name():
    run = RunConfig.model_validate(_run_dict(
        dataset_name="val_set", eval_for_every=2, metrics=[{"kind": "exact_match"}],
    ))
    assert run.validation.dataset_name == "val_set"


def test_validation_dataset_id_for_name_picks_latest(tmp_path):
    from evsys_sdk.workspace import Workspace

    class _S:
        def list_validation_datasets(self, project_id=None):
            return [{"id": "v1", "name": "val", "version": 1},
                    {"id": "v2", "name": "val", "version": 4}]

    ws = Workspace(_S(), root=str(tmp_path))
    assert ws.validation_dataset_id_for_name("val") == "v2"


def test_load_validation_by_dataset_name(tmp_path, monkeypatch):
    from evsys_sdk import runner
    from evsys_sdk.workspace import MaterializedDataset

    jsonl = tmp_path / "v.jsonl"
    jsonl.write_text(json.dumps(_row("t1", "A")) + "\n")

    class _FakeWS:
        last: str | None = None

        def __init__(self, *a, **k) -> None:
            pass

        def validation_dataset_id_for_name(self, name: str) -> str:
            return "vid-resolved"

        def pull_validation_dataset(self, vid: str, *, force: bool = False):
            _FakeWS.last = vid
            return MaterializedDataset(str(jsonl), "harbor_task", None, 1, cached=False)

    monkeypatch.setattr("evsys_sdk.workspace.Workspace", _FakeWS)
    run = RunConfig.model_validate(_run_dict(
        dataset_name="val_set", eval_for_every=2, metrics=[{"kind": "exact_match"}],
    ))
    out = runner._load_validation(run)
    assert _FakeWS.last == "vid-resolved"          # name → id → pull
    assert out is not None
    assert out["dataset_id"] == "vid-resolved"     # resolved id carried for create_eval
    assert len(out["tasks"]) == 1


# ---------------------------------------------------------------------------
# Split-aware forwarding
# ---------------------------------------------------------------------------


class _SplitStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def log_metrics(self, *, run_id: str, step: int, metrics: dict, split: str = "train") -> dict:
        self.calls.append({"step": step, "metrics": dict(metrics), "split": split})
        return {"ok": True}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_val_prefixed_metrics_forwarded_as_val_split(tmp_path: Path):
    _write(tmp_path / "logs" / "metrics.jsonl", [
        {"step": 10, "metrics": {"loss": 0.5, "val/exact_match": 0.7}},
    ])
    store = _SplitStore()
    sent = forward_step_metrics(store, "run-A", tmp_path)
    # one train call (loss) + one val call (exact_match)
    assert sent == 2
    by_split = {c["split"]: c["metrics"] for c in store.calls}
    assert by_split["train"] == {"loss": 0.5}
    assert by_split["val"] == {"val/exact_match": 0.7}


def test_val_only_row_forwards_single_val_call(tmp_path: Path):
    _write(tmp_path / "logs" / "metrics.jsonl", [
        {"step": 20, "metrics": {"val/exact_match": 1.0}},
    ])
    store = _SplitStore()
    sent = forward_step_metrics(store, "run-A", tmp_path)
    assert sent == 1
    assert store.calls[0]["split"] == "val"
    assert store.calls[0]["step"] == 20

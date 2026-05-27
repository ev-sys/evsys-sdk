"""ModalMiles dispatcher — unit-tested with a fake `modal` module.

We can't hit real Modal/GPU here, so we inject a fake `modal` into sys.modules
and stub the volume upload. This exercises engine routing, env building, the
spawn→poll→ingest-checkpoint path, and the failure paths — everything except
the remote loop itself (which runs in the deployed Modal app).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import ClassVar

from trajectory_labs import (
    AlgorithmConfig,
    BackendConfig,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LogStoreSpec,
    ModelConfig,
    RunConfig,
    run_experiment,
)
from trajectory_labs.algorithms import modal_miles as mm


class _FakeCall:
    object_id = "fake-call-1"

    def __init__(self, result):
        self._result = result

    def get(self, timeout=None):
        return self._result


class _FakeFn:
    last_spawn_kwargs: ClassVar[dict] = {}

    def __init__(self, result):
        self._result = result

    def spawn(self, **kwargs):
        _FakeFn.last_spawn_kwargs = kwargs
        return _FakeCall(self._result)


def _install_fake_modal(monkeypatch, *, result):
    fake = types.ModuleType("modal")

    class _Function:
        @staticmethod
        def from_name(app_name, fn_name):
            _Function.looked_up = (app_name, fn_name)
            return _FakeFn(result)

    fake.Function = _Function
    monkeypatch.setitem(sys.modules, "modal", fake)
    # Stub the volume upload so no real `modal` CLI runs.
    monkeypatch.setattr(mm, "_volume_put", lambda **kw: None)
    return _Function


def _cfg(tmp_path: Path, *, recipe_kind="sft", engine="auto", model="Qwen/Qwen3-4B") -> ExperimentConfig:
    return ExperimentConfig(
        name="modal_e2e",
        output_dir=str(tmp_path / "out"),
        log_store=LogStoreSpec(kind="jsonl"),
        run=RunConfig(
            name="r",
            data=DataConfig(source_kind="in_memory", rows=[{"messages": [{"role": "user", "content": "hi"}]}]),
            model=ModelConfig(name=model),
            algorithm=AlgorithmConfig(kind="modal_miles", params={"recipe_kind": recipe_kind, "num_steps": 5}),
            backend=BackendConfig(kind="modal", params={"engine": engine}),
            eval=EvalConfig(enabled=False),
        ),
    )


def test_dispatch_completes_and_ingests_checkpoint(tmp_path, monkeypatch):
    fn = _install_fake_modal(monkeypatch, result={
        "exit_code": 0, "final_checkpoint_path": "/checkpoints/r/0/final",
        "stdout_tail": "ok", "stderr_tail": "",
    })
    results = run_experiment(_cfg(tmp_path, recipe_kind="rl", model="Qwen/Qwen3-32B"))
    r = results[0]
    assert r.status == "completed"
    assert r.artifacts.get("final_checkpoint") == "/checkpoints/r/0/final"
    # 32B + RL -> miles engine -> train_remote_miles.
    assert fn.looked_up == ("trajectory-training", "train_remote_miles")
    assert _FakeFn.last_spawn_kwargs["script_name"] == "run_miles_rl.sh"
    assert _FakeFn.last_spawn_kwargs["env_overrides"]["BASE_MODEL"] == "Qwen/Qwen3-32B"


def test_small_sft_routes_to_unsloth(tmp_path, monkeypatch):
    fn = _install_fake_modal(monkeypatch, result={"exit_code": 0, "final_checkpoint_path": None})
    run_experiment(_cfg(tmp_path, recipe_kind="sft", model="Qwen/Qwen3-4B"))
    assert fn.looked_up == ("trajectory-training", "train_remote_unsloth")
    assert _FakeFn.last_spawn_kwargs["script_name"] == "run_unsloth_sft.py"


def test_engine_override_forces_miles(tmp_path, monkeypatch):
    fn = _install_fake_modal(monkeypatch, result={"exit_code": 0, "final_checkpoint_path": None})
    run_experiment(_cfg(tmp_path, recipe_kind="sft", engine="miles", model="Qwen/Qwen3-4B"))
    assert fn.looked_up == ("trajectory-training", "train_remote_miles")
    assert _FakeFn.last_spawn_kwargs["script_name"] == "run_miles_sft.sh"


def test_nonzero_exit_is_failure(tmp_path, monkeypatch):
    _install_fake_modal(monkeypatch, result={"exit_code": 1, "stderr_tail": "boom"})
    r = run_experiment(_cfg(tmp_path))[0]
    assert r.status == "failed"
    assert "boom" in (r.error or "")


def test_missing_modal_fails_gracefully(tmp_path, monkeypatch):
    # Simulate modal not installed: block the import.
    monkeypatch.setitem(sys.modules, "modal", None)
    r = run_experiment(_cfg(tmp_path))[0]
    assert r.status == "failed"
    assert "modal not installed" in (r.error or "")

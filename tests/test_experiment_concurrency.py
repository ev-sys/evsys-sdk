"""Concurrent arms — Experiment runs arms in a capped thread pool.

Uses an injected train_fn that records how many arms are in-flight at once, so we
can assert the cap is honored and that arms genuinely overlap (no tinker needed).
"""

from __future__ import annotations

import threading
import time

from evsys_sdk import RunResult
from evsys_sdk.config import AlgorithmConfig, DataConfig, ExperimentConfig, ModelConfig, RunConfig
from evsys_sdk.experiment import Experiment


def _runs(n: int) -> list[RunConfig]:
    return [
        RunConfig(name=f"arm{i}", data=DataConfig(path="d.jsonl"),
                  model=ModelConfig(name="m"), algorithm=AlgorithmConfig(kind="sft"))
        for i in range(n)
    ]


def _tracking_train_fn(state: dict, lock: threading.Lock, hold: float = 0.15):
    def train_fn(cfg: ExperimentConfig):
        name = cfg.run.name
        with lock:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(hold)              # hold the slot so overlap is observable
        with lock:
            state["active"] -= 1
        return [RunResult(run_id=name, status="completed", metrics={"score": float(len(name))})]
    return train_fn


def _run(n_arms: int, cap: int) -> tuple:
    state = {"active": 0, "max": 0}
    lock = threading.Lock()
    cfg = ExperimentConfig(name="exp", runs=_runs(n_arms), max_concurrent_arms=cap)
    exp = Experiment(cfg, store=None, train_fn=_tracking_train_fn(state, lock))
    result = exp.run()
    return result, state


def test_arms_run_concurrently_up_to_cap():
    result, state = _run(n_arms=4, cap=2)
    assert len(result.arms) == 4
    assert all(a.status == "completed" for a in result.arms)
    assert state["max"] == 2          # exactly the cap: 2 arms overlapped, never more


def test_cap_one_is_sequential():
    result, state = _run(n_arms=3, cap=1)
    assert all(a.status == "completed" for a in result.arms)
    assert state["max"] == 1          # never more than one arm in flight


def test_arm_result_order_is_preserved_regardless_of_completion():
    # Vary hold so completion order != submission order; result order must still match.
    state = {"active": 0, "max": 0}
    lock = threading.Lock()

    def train_fn(cfg):
        name = cfg.run.name
        time.sleep(0.05 * (5 - int(name[-1])))   # arm0 finishes last
        return [RunResult(run_id=name, status="completed", metrics={})]

    cfg = ExperimentConfig(name="exp", runs=_runs(4), max_concurrent_arms=4)
    result = Experiment(cfg, store=None, train_fn=train_fn).run()
    assert [a.name for a in result.arms] == ["arm0", "arm1", "arm2", "arm3"]


def test_one_arm_failure_does_not_kill_the_others():
    def train_fn(cfg):
        if cfg.run.name == "arm1":
            raise RuntimeError("boom")
        return [RunResult(run_id=cfg.run.name, status="completed", metrics={})]

    cfg = ExperimentConfig(name="exp", runs=_runs(3), max_concurrent_arms=3)
    result = Experiment(cfg, store=None, train_fn=train_fn).run()
    by_name = {a.name: a.status for a in result.arms}
    assert by_name == {"arm0": "completed", "arm1": "failed", "arm2": "completed"}


def test_default_max_concurrent_arms_is_4():
    assert ExperimentConfig(name="x", run=_runs(1)[0]).max_concurrent_arms == 4

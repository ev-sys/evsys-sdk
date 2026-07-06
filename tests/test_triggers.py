"""Layer-2 trigger — the cheap deterministic gate: state/policy, built-in fns,
the driver's eval cadence + escalation events + logging, live-policy retune, and
error isolation."""

from __future__ import annotations

import json

from evsys_sdk.protocols import TriggerDecision
from evsys_sdk.registry import register_trigger
from evsys_sdk.trace_types import Trace
from evsys_sdk.triggers import (
    LocalTriggerStore,
    TriggerDriver,
    TriggerPolicy,
    TriggerState,
    build_trigger,
    resolve_hook,
)


def mk_trace(i: int, *, status: str = "success", reward: float | None = None, inp: str = "q") -> Trace:
    fb = [] if reward is None else [{"key": "correct", "score": reward, "comment": None, "source": "x", "turn": None}]
    return Trace(
        trace_id=f"t{i}",
        messages=[{"role": "user", "content": f"{inp}{i}"}, {"role": "assistant", "content": "a"}],
        feedback=fb,
        metadata={"source": "test", "status": status, "timestamp": f"2026-07-06T00:00:{i:02d}"},
    )


def _log(store: LocalTriggerStore) -> list[dict]:
    p = store.log_path()
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# 1. State push + eval cadence ---------------------------------------------

def test_state_push_bounds_and_aggregates():
    st = TriggerState()
    pol = TriggerPolicy(window=3)
    for i in range(5):
        st.push(mk_trace(i, status="error" if i % 2 else "success", reward=0.2 if i % 2 else 0.9), pol)
    assert len(st.window) == 3  # bounded to policy.window
    assert st.counters["n_seen"] == 5  # lifetime counter unbounded
    assert st.aggregates["window_size"] == 3
    assert set(st.aggregates) == {"window_size", "avg_reward", "n_failed", "status_hist"}


def test_driver_eval_cadence(tmp_path):
    store = LocalTriggerStore(tmp_path)
    drv = TriggerDriver(store, seed_policy=TriggerPolicy(kind="volume", params={"every": 999}, every_n=10))
    for i in range(25):
        drv(mk_trace(i), None)
    evals = [r for r in _log(store) if r["event"] == "evaluate"]
    assert len(evals) == 2  # fired at trace 10 and 20, not every trace
    assert store.read_state().counters["n_seen"] == 25


# 2. Built-ins escalate -----------------------------------------------------

def test_failure_rate_escalates():
    trig = build_trigger(TriggerPolicy(kind="failure_rate", params={"threshold": 0.4, "min_traces": 5}))
    st = TriggerState()
    pol = TriggerPolicy(window=100)
    for i in range(10):  # 6/10 fail
        st.push(mk_trace(i, status="error" if i < 6 else "success", reward=0.1 if i < 6 else 0.9), pol)
    d = trig.evaluate(st)
    assert d.escalate and d.signal["failure_rate"] == 0.6 and len(d.trace_ids) == 6


def test_feedback_drop_and_volume_and_novelty():
    st = TriggerState()
    pol = TriggerPolicy(window=100)
    for i in range(10):
        st.push(mk_trace(i, reward=0.2, inp=f"unique{i}"), pol)  # low reward + all-distinct inputs
    assert build_trigger(TriggerPolicy(kind="feedback_drop", params={"floor": 0.5, "min_traces": 5})).evaluate(st).escalate
    assert build_trigger(TriggerPolicy(kind="novelty", params={"threshold": 0.8, "min_traces": 5})).evaluate(st).escalate
    st.counters["since_last_escalation"] = 12
    assert build_trigger(TriggerPolicy(kind="volume", params={"every": 10})).evaluate(st).escalate


def test_driver_writes_escalation_event(tmp_path):
    store = LocalTriggerStore(tmp_path)
    drv = TriggerDriver(store, seed_policy=TriggerPolicy(
        kind="failure_rate", params={"threshold": 0.4, "min_traces": 5}, every_n=10))
    for i in range(10):  # all fail → escalates at the 10th
        drv(mk_trace(i, status="error", reward=0.0), None)
    esc = list(store.escalations_dir().glob("*.json"))
    assert len(esc) == 1
    event = json.loads(esc[0].read_text())
    assert event["kind"] == "failure_rate" and event["signal"]["failure_rate"] == 1.0
    assert any(r["event"] == "escalation" for r in _log(store))


# 3. Live policy: seed doesn't clobber; edits retune the cadence ------------

def test_seed_does_not_clobber_existing_policy(tmp_path):
    store = LocalTriggerStore(tmp_path)
    store.write_policy(TriggerPolicy(kind="novelty", every_n=7))  # agent-edited policy already there
    TriggerDriver(store, seed_policy=TriggerPolicy(kind="failure_rate", every_n=20))  # seed must NOT overwrite
    live = store.read_policy()
    assert live.kind == "novelty" and live.every_n == 7


def test_live_policy_retune_changes_cadence(tmp_path):
    store = LocalTriggerStore(tmp_path)
    drv = TriggerDriver(store, seed_policy=TriggerPolicy(kind="volume", params={"every": 999}, every_n=100))
    for i in range(5):
        drv(mk_trace(i), None)
    assert not [r for r in _log(store) if r["event"] == "evaluate"]  # every_n=100 → no eval yet
    store.write_policy(TriggerPolicy(kind="volume", params={"every": 999}, every_n=3))  # retune mid-run
    drv(mk_trace(99), None)  # since_last_eval=6 ≥ 3 → fires now
    assert [r for r in _log(store) if r["event"] == "evaluate"]


# 4. No trigger configured → Layer-1 no-op ---------------------------------

def test_resolve_hook_none():
    assert resolve_hook(None) is None


def test_resolve_hook_builds_working_gate(tmp_path):
    from evsys_sdk.config import TriggerConfig

    cfg = TriggerConfig(kind="failure_rate", params={"threshold": 0.4, "min_traces": 5},
                        every_n=5, state_dir=str(tmp_path))
    hook = resolve_hook(cfg)  # the TraceHook Layer-1 ingestion would fire
    for i in range(5):
        hook(mk_trace(i, status="error", reward=0.0), None)
    store = LocalTriggerStore(tmp_path)
    assert store.read_policy().kind == "failure_rate"  # policy.json seeded from config
    assert list(store.escalations_dir().glob("*.json"))  # gate fired end-to-end


def test_builtins_below_threshold_do_not_escalate():
    st = TriggerState()
    for _ in range(10):  # all healthy, all the SAME input (no drift)
        st.window.append({"trace_id": "t", "reward": 0.95, "status": "success", "input_sig": "same"})
    st._recompute()
    assert not build_trigger(TriggerPolicy(kind="failure_rate", params={"min_traces": 5})).evaluate(st).escalate
    assert not build_trigger(TriggerPolicy(kind="feedback_drop", params={"min_traces": 5})).evaluate(st).escalate
    assert not build_trigger(TriggerPolicy(kind="novelty", params={"min_traces": 5})).evaluate(st).escalate
    assert not build_trigger(TriggerPolicy(kind="volume", params={"every": 10})).evaluate(st).escalate
    # too few traces → all abstain
    empty = TriggerState()
    for k in ("failure_rate", "feedback_drop", "novelty"):
        assert not build_trigger(TriggerPolicy(kind=k, params={"min_traces": 5})).evaluate(empty).escalate


# 5. Error isolation: a raising fn never kills ingestion -------------------

def test_raising_trigger_is_isolated(tmp_path):
    @register_trigger("boom")
    class Boom:
        name = "boom"

        class Config:
            def __init__(self, **kw):
                pass

        def __init__(self, **params):
            self.cfg = self.Config(**params)

        def evaluate(self, state) -> TriggerDecision:
            raise RuntimeError("kaboom")

    store = LocalTriggerStore(tmp_path)
    drv = TriggerDriver(store, seed_policy=TriggerPolicy(kind="boom", every_n=1))
    drv(mk_trace(1), None)  # must not raise
    assert store.state_path().exists()  # state still persisted
    assert any(r["event"] == "error" for r in _log(store))

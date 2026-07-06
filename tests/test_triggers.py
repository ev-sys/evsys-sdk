"""Layer-2 trigger — the cheap deterministic gate: state/policy, the driver's
eval cadence + escalation events + logging, live-policy retune, and error
isolation.

No fns ship with the SDK, so the tests register their own example triggers
(exactly how a researcher or the trigger agent would) to exercise the mechanism.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict

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

# -- example triggers, registered the way a user/agent would ---------------

@register_trigger("ex_failrate")
class ExFailRate:
    """Escalate when the failing fraction of the window crosses ``threshold``."""

    name = "ex_failrate"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        threshold: float = 0.4
        min_traces: int = 5

    def __init__(self, **params):
        self.cfg = self.Config(**params)

    def evaluate(self, state: TriggerState) -> TriggerDecision:
        w = state.window  # raw traces: {trace_id, messages, feedback, metadata}
        if len(w) < self.cfg.min_traces:
            return TriggerDecision(False, f"only {len(w)} traces")
        bad = [t for t in w if t["metadata"].get("status") == "error"]  # fn does its OWN reduction
        rate = len(bad) / len(w)
        if rate >= self.cfg.threshold:
            return TriggerDecision(True, f"failure {rate:.0%}", {"failure_rate": rate},
                                   [t["trace_id"] for t in bad])
        return TriggerDecision(False, f"failure {rate:.0%}", {"failure_rate": rate})


@register_trigger("ex_quiet")
class ExQuiet:
    """Never escalates — used to probe the eval cadence."""

    name = "ex_quiet"
    Config = BaseModel

    def __init__(self, **params):
        pass

    def evaluate(self, state: TriggerState) -> TriggerDecision:
        return TriggerDecision(False, "quiet")


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
    return [] if not p.exists() else [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# 1. State push + eval cadence ---------------------------------------------

def test_state_push_keeps_raw_traces_bounded():
    st = TriggerState()
    pol = TriggerPolicy(window=3)
    for i in range(5):
        st.push(mk_trace(i, status="error" if i % 2 else "success", reward=0.2 if i % 2 else 0.9), pol)
    assert len(st.window) == 3  # bounded to policy.window
    assert st.counters["n_seen"] == 5  # lifetime counter unbounded
    # the window holds the RAW traces verbatim — nothing extracted, no aggregates
    last = st.window[-1]
    assert set(last) == {"trace_id", "messages", "feedback", "metadata"}
    assert last["messages"][0]["role"] == "user"
    assert not hasattr(st, "aggregates")


def test_driver_eval_cadence(tmp_path):
    store = LocalTriggerStore(tmp_path)
    drv = TriggerDriver(store, seed_policy=TriggerPolicy(kind="ex_quiet", every_n=10))
    for i in range(25):
        drv(mk_trace(i), None)
    evals = [r for r in _log(store) if r["event"] == "evaluate"]
    assert len(evals) == 2  # fired at trace 10 and 20, not every trace
    assert store.read_state().counters["n_seen"] == 25


# 2. Escalation event + logging --------------------------------------------

def test_example_trigger_escalates():
    trig = build_trigger(TriggerPolicy(kind="ex_failrate", params={"threshold": 0.4, "min_traces": 5}))
    st = TriggerState()
    pol = TriggerPolicy(window=100)
    for i in range(10):  # 6/10 fail
        st.push(mk_trace(i, status="error" if i < 6 else "success"), pol)
    d = trig.evaluate(st)
    assert d.escalate and d.signal["failure_rate"] == 0.6 and len(d.trace_ids) == 6


def test_driver_writes_escalation_event(tmp_path):
    store = LocalTriggerStore(tmp_path)
    drv = TriggerDriver(store, seed_policy=TriggerPolicy(
        kind="ex_failrate", params={"threshold": 0.4, "min_traces": 5}, every_n=10))
    for i in range(10):  # all fail → escalates at the 10th
        drv(mk_trace(i, status="error"), None)
    esc = list(store.escalations_dir().glob("*.json"))
    assert len(esc) == 1
    event = json.loads(esc[0].read_text())
    assert event["kind"] == "ex_failrate" and event["signal"]["failure_rate"] == 1.0
    assert any(r["event"] == "escalation" for r in _log(store))


# 3. Live policy: seed doesn't clobber; edits retune the cadence ------------

def test_seed_does_not_clobber_existing_policy(tmp_path):
    store = LocalTriggerStore(tmp_path)
    store.write_policy(TriggerPolicy(kind="ex_quiet", every_n=7))  # agent-edited policy already there
    TriggerDriver(store, seed_policy=TriggerPolicy(kind="ex_failrate", every_n=20))  # seed must NOT overwrite
    live = store.read_policy()
    assert live.kind == "ex_quiet" and live.every_n == 7


def test_live_policy_retune_changes_cadence(tmp_path):
    store = LocalTriggerStore(tmp_path)
    drv = TriggerDriver(store, seed_policy=TriggerPolicy(kind="ex_quiet", every_n=100))
    for i in range(5):
        drv(mk_trace(i), None)
    assert not [r for r in _log(store) if r["event"] == "evaluate"]  # every_n=100 → no eval yet
    store.write_policy(TriggerPolicy(kind="ex_quiet", every_n=3))  # retune mid-run
    drv(mk_trace(99), None)  # since_last_eval=6 ≥ 3 → fires now
    assert [r for r in _log(store) if r["event"] == "evaluate"]


# 4. Hook resolution -------------------------------------------------------

def test_resolve_hook_none():
    assert resolve_hook(None) is None


def test_resolve_hook_builds_working_gate(tmp_path):
    from evsys_sdk.config import TriggerConfig

    cfg = TriggerConfig(kind="ex_failrate", params={"threshold": 0.4, "min_traces": 5},
                        every_n=5, state_dir=str(tmp_path))
    hook = resolve_hook(cfg)  # the TraceHook Layer-1 ingestion would fire
    for i in range(5):
        hook(mk_trace(i, status="error"), None)
    store = LocalTriggerStore(tmp_path)
    assert store.read_policy().kind == "ex_failrate"  # policy.json seeded from config
    assert list(store.escalations_dir().glob("*.json"))  # gate fired end-to-end


# 5. Error isolation: a raising / unregistered fn never kills ingestion -----

def test_raising_trigger_is_isolated(tmp_path):
    @register_trigger("boom")
    class Boom:
        name = "boom"
        Config = BaseModel

        def __init__(self, **params):
            pass

        def evaluate(self, state) -> TriggerDecision:
            raise RuntimeError("kaboom")

    store = LocalTriggerStore(tmp_path)
    drv = TriggerDriver(store, seed_policy=TriggerPolicy(kind="boom", every_n=1))
    drv(mk_trace(1), None)  # must not raise
    assert store.state_path().exists()  # state still persisted
    assert any(r["event"] == "error" for r in _log(store))


def test_unregistered_kind_is_isolated(tmp_path):
    store = LocalTriggerStore(tmp_path)
    drv = TriggerDriver(store, seed_policy=TriggerPolicy(kind="does_not_exist", every_n=1))
    drv(mk_trace(1), None)  # unresolved kind is caught, logged, ingestion survives
    assert any(r["event"] == "error" for r in _log(store))

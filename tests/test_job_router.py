"""Tests for the autonomous JobRouter: queue -> place -> track snapshots ->
preempt -> requeue -> restart (reuse or cross-provider stream), no operator."""
import pytest

from evsys_sdk.compute import availability as av
from evsys_sdk.compute.checkpoint_map import CheckpointMap
from evsys_sdk.compute.job_router import JobRouter, NodeHandle
from evsys_sdk.compute.queue import DONE, QUEUED, RUNNING, Queue


def _cap(provider="verda", gpu="H100", usd=2.0, sku="1H100", region="r1"):
    return av.Capacity(provider=provider, gpu=gpu, count=1, state="available",
                       region=region, sku=sku, spot=True, usd_hr=usd)


class FakeProvisioner:
    """In-memory cluster. Machines live until the test kills them."""

    def __init__(self):
        self.machines = {}          # machine_id -> alive?
        self.calls = []             # (job_id, plan_mode, interval)
        self.fail_next = False
        self._n = 0

    def provision(self, job, capacity, plan, snapshot_interval_s):
        self.calls.append((job.id, plan.mode, snapshot_interval_s, plan))
        if self.fail_next:
            self.fail_next = False
            return None
        self._n += 1
        mid = f"m{self._n}"
        self.machines[mid] = True
        return NodeHandle(job_id=job.id, provider=capacity.provider,
                          machine_id=mid, region=capacity.region or "",
                          gpu=capacity.gpu,
                          volume=plan.volume or f"vol-{mid}",
                          usd_hr=capacity.usd_hr or 0)

    def alive(self, handle):
        return self.machines.get(handle.machine_id, False)

    def kill(self, machine_id):
        self.machines[machine_id] = False

    def terminate(self, handle, *, keep_volume=True):
        self.machines[handle.machine_id] = False


@pytest.fixture
def rig(tmp_path, monkeypatch):
    q = Queue(path=str(tmp_path / "q.jsonl"))
    m = CheckpointMap(path=str(tmp_path / "map.jsonl"))
    prov = FakeProvisioner()
    events = []
    caps = [_cap()]
    monkeypatch.setattr(av, "scan", lambda *a, **k: list(caps))
    router = JobRouter(q, m, prov, events=lambda: events.pop_all()
                       if hasattr(events, "pop_all") else _drain(events),
                       handles_path=str(tmp_path / "handles.json"),
                       snapshot_cost_s=45.0, mtbf_s=3600.0)
    return router, q, m, prov, events, caps


def _drain(events):
    out = list(events)
    events.clear()
    return out


def test_submit_and_place_fresh(rig):
    router, q, m, prov, events, caps = rig
    job = router.submit("configs/a.yaml", model="Qwen/Qwen3.5-9B",
                        gpus=["H100"])
    out = router.tick()
    assert out["placed"] == [(job.id, "fresh")]
    assert q.jobs()[0].state == RUNNING
    # Young/Daly interval was computed and handed to the node
    interval = prov.calls[0][2]
    assert 500 < interval < 600           # sqrt(2*45*3600) ~= 569s


def test_agent_checkpoints_maintain_map_automatically(rig):
    router, q, m, prov, events, caps = rig
    job = router.submit("c.yaml", model="m", gpus=["H100"])
    router.tick()
    events.append({"kind": "checkpoint", "job_id": job.id, "step": 100,
                   "store": {"kind": "local_dir", "provider": "verda",
                             "volume": "vol-m1", "path": "/data/store"},
                   "base_key": "base.evd", "delta_key": "step-100.evd",
                   "sha256": {"base.evd": "a" * 64}})
    out = router.tick()
    assert out["events"] == 1
    ck = m.latest(job.id)
    assert ck is not None and ck.step == 100
    assert ck.store.volume == "vol-m1"


def test_preemption_auto_requeues_and_reuses_volume(rig):
    router, q, m, prov, events, caps = rig
    job = router.submit("c.yaml", model="m", gpus=["H100"])
    router.tick()
    events.append({"kind": "checkpoint", "job_id": job.id, "step": 200,
                   "store": {"kind": "local_dir", "provider": "verda",
                             "volume": "vol-m1", "path": "/data/store"},
                   "base_key": "base.evd", "delta_key": "step-200.evd"})
    router.tick()
    # the machine dies -- nobody tells the router; it notices
    prov.kill("m1")
    out = router.tick()
    assert out["preempted"] == [job.id]
    # the SAME tick re-places it with a zero-copy volume reuse
    assert out["placed"] == [(job.id, "reuse")]
    plan = prov.calls[-1][3]
    assert plan.volume == "vol-m1"
    assert plan.checkpoint.step == 200
    fresh = q.jobs()[0]
    assert fresh.state == RUNNING
    assert fresh.attempts == 1
    assert "resume:" in fresh.error


def test_cross_provider_restart_streams(rig):
    router, q, m, prov, events, caps = rig
    from evsys_sdk.compute.portability import register_storage_caps
    register_storage_caps("othercloud", True)
    try:
        job = router.submit("c.yaml", model="m", gpus=["H100"])
        router.tick()
        events.append({"kind": "checkpoint", "job_id": job.id, "step": 300,
                       "store": {"kind": "local_dir", "provider": "verda",
                                 "volume": "vol-m1", "path": "/data/store"},
                       "base_key": "base.evd", "delta_key": "step-300.evd"})
        router.tick()
        prov.kill("m1")
        caps[:] = [_cap(provider="othercloud", region="us-1")]   # verda gone
        out = router.tick()
        assert out["placed"] == [(job.id, "stream")]
        plan = prov.calls[-1][3]
        assert plan.src.provider == "verda"
        assert plan.src.volume == "vol-m1"
        assert plan.checkpoint.step == 300
    finally:
        register_storage_caps("othercloud", False)


def test_done_event_terminates_and_keeps_volume(rig):
    router, q, m, prov, events, caps = rig
    job = router.submit("c.yaml", model="m", gpus=["H100"])
    router.tick()
    events.append({"kind": "done", "job_id": job.id})
    router.tick()
    assert q.jobs()[0].state == DONE
    assert job.id not in router.handles
    assert prov.machines["m1"] is False       # terminated


def test_failed_launch_leaves_job_queued(rig):
    router, q, m, prov, events, caps = rig
    router.submit("c.yaml", model="m", gpus=["H100"])
    prov.fail_next = True
    out = router.tick()
    assert out["placed"] == []
    assert q.jobs()[0].state == QUEUED        # retried next tick
    out2 = router.tick()
    assert len(out2["placed"]) == 1


def test_router_restart_recovers_handles(rig, tmp_path, monkeypatch):
    router, q, m, prov, events, caps = rig
    job = router.submit("c.yaml", model="m", gpus=["H100"])
    router.tick()
    # a new router process over the same durable state
    router2 = JobRouter(q, m, prov, events=lambda: [],
                        handles_path=str(router._path))
    assert job.id in router2.handles
    prov.kill("m1")
    out = router2.tick()
    assert out["preempted"] == [job.id]       # reconcile works post-restart


def test_run_exits_when_all_terminal(rig):
    router, q, m, prov, events, caps = rig
    job = router.submit("c.yaml", model="m", gpus=["H100"])
    router.tick()
    events.append({"kind": "done", "job_id": job.id})
    router.poll_s = 0.01
    router.run(timeout_s=5)                   # returns via all-terminal, not timeout
    assert q.jobs()[0].state == DONE


# ---------------------------------------------------------------------------
# Multi-provider routing + the provisioner registry
# ---------------------------------------------------------------------------


def test_multi_provider_dict_routes_by_capacity_provider(tmp_path, monkeypatch):
    q = Queue(path=str(tmp_path / "q.jsonl"))
    m = CheckpointMap(path=str(tmp_path / "map.jsonl"))
    pv, pw = FakeProvisioner(), FakeProvisioner()
    monkeypatch.setattr(av, "scan", lambda *a, **k: [_cap(provider="verda")])
    router = JobRouter(q, m, {"verda": pv, "vast": pw}, events=lambda: [],
                       handles_path=str(tmp_path / "handles.json"))
    job = router.submit("c.yaml", model="m", gpus=["H100"])
    router.tick()
    assert [c[0] for c in pv.calls] == [job.id]      # verda got the launch
    assert pw.calls == []                            # vast untouched
    # liveness + teardown route by the HANDLE's provider on later ticks
    router.tick()
    assert pv.machines and not pw.machines


def test_prov_for_unknown_provider_raises(tmp_path):
    q = Queue(path=str(tmp_path / "q.jsonl"))
    m = CheckpointMap(path=str(tmp_path / "map.jsonl"))
    router = JobRouter(q, m, {"verda": FakeProvisioner()}, events=lambda: [],
                       handles_path=str(tmp_path / "handles.json"))
    with pytest.raises(KeyError, match="prime"):
        router._prov_for("prime")


def test_build_provisioner_resolves_registry_and_validates(tmp_path):
    import evsys_sdk.compute  # noqa: F401  registers the built-ins
    from evsys_sdk.compute.job_router import build_provisioner
    from evsys_sdk.compute.provisioner_verda import VerdaProvisioner
    from evsys_sdk.registry import get_provisioner, list_provisioners

    assert "verda" in list_provisioners()
    assert get_provisioner("verda") is VerdaProvisioner

    calls = []
    p = build_provisioner("verda", {"ssh_key_id": "k1", "volume_gb": 100},
                          call=lambda *a, **k: calls.append(a) or "x")
    assert isinstance(p, VerdaProvisioner)
    assert p.ssh_key_id == "k1" and p.volume_gb == 100

    with pytest.raises(Exception, match="ssh_key_idd|extra|validation"):
        build_provisioner("verda", {"ssh_key_idd": "typo"},
                          call=lambda *a, **k: "x")


def test_submit_defaults_to_spot(tmp_path):
    q = Queue(path=str(tmp_path / "q.jsonl"))
    m = CheckpointMap(path=str(tmp_path / "map.jsonl"))
    router = JobRouter(q, m, FakeProvisioner(), events=lambda: [],
                       handles_path=str(tmp_path / "handles.json"))
    assert router.submit("c.yaml", model="m").spot is True      # the default
    assert router.submit("c.yaml", model="m", spot=None).spot is None
    assert router.submit("c.yaml", model="m", spot=False).spot is False

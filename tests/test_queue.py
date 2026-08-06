"""The SkyRL job queue: durability, packing, and not losing work to capacity.

Every test here pins a behaviour that costs either money (renting a GPU when a
live one had a free adapter slot) or a run (a job silently dropped because the
scheduler crashed mid-launch).
"""

from __future__ import annotations

import json
import time

import pytest

from evsys_sdk.compute import availability as av
from evsys_sdk.compute import queue as q


@pytest.fixture
def qq(tmp_path):
    return q.Queue(path=str(tmp_path / "queue.jsonl"))


def cap(provider="verda", gpu="H200", count=1, usd=2.0, region="FIN-03"):
    return av.Capacity(provider=provider, gpu=gpu, count=count,
                       state=av.AVAILABLE, region=region, usd_hr=usd)


def cluster(name="c1", model="m", max_adapters=8, used=0):
    return q.Cluster(name=name, model=model, max_adapters=max_adapters,
                     used_adapters=used)


# -- durability --------------------------------------------------------------


def test_a_submitted_job_survives_a_restart(qq, tmp_path):
    """The queue is the record. A scheduler crash must not lose work."""
    qq.submit("a.yaml", model="Qwen/Qwen3-4B")
    reopened = q.Queue(path=str(tmp_path / "queue.jsonl"))
    assert [j.config for j in reopened.jobs()] == ["a.yaml"]


def test_latest_record_wins(qq):
    j = qq.submit("a.yaml", model="m")
    qq.update(j, q.RUNNING, cluster="c1")
    got = qq.jobs()
    assert len(got) == 1 and got[0].state == q.RUNNING and got[0].cluster == "c1"


def test_a_torn_final_line_does_not_break_the_queue(qq):
    """A process killed mid-write leaves half a line. Reverting that job to its
    previous state is safe; guessing at the truncated record is not."""
    j = qq.submit("a.yaml", model="m")
    qq.update(j, q.RUNNING)
    with qq.path.open("a") as f:
        f.write('{"id": "broke", "config": "b.ya')
    got = qq.jobs()
    assert [x.id for x in got] == [j.id] and got[0].state == q.RUNNING


def test_unknown_fields_are_ignored_on_read(qq):
    """A queue written by a newer SDK must still load."""
    with qq.path.open("a") as f:
        f.write(json.dumps({"id": "x", "config": "a.yaml", "model": "m",
                            "state": q.QUEUED, "from_the_future": 1}) + "\n")
    assert [j.id for j in qq.jobs()] == ["x"]


def test_compact_collapses_history_atomically(qq):
    j = qq.submit("a.yaml", model="m")
    for s in (q.PLACING, q.RUNNING, q.DONE):
        qq.update(j, s)
    assert len(qq.path.read_text().splitlines()) == 4
    assert qq.compact() == 1
    assert len(qq.path.read_text().splitlines()) == 1
    assert qq.jobs()[0].state == q.DONE


# -- ordering ----------------------------------------------------------------


def test_priority_wins_and_ties_stay_fifo(qq):
    a = qq.submit("a.yaml", model="m")
    time.sleep(0.01)
    b = qq.submit("b.yaml", model="m")
    time.sleep(0.01)
    c = qq.submit("c.yaml", model="m", priority=5)
    assert [j.id for j in qq.pending()] == [c.id, a.id, b.id]


def test_only_queued_jobs_are_pending(qq):
    a = qq.submit("a.yaml", model="m")
    qq.submit("b.yaml", model="m")
    qq.update(a, q.RUNNING)
    assert [j.config for j in qq.pending()] == ["b.yaml"]


# -- packing: the whole economic argument ------------------------------------


def test_a_job_packs_onto_a_live_cluster_instead_of_renting(qq, monkeypatch):
    """Packing beats renting on cost: one H100 with 8 adapters measured
    $0.0429/M against $0.0471/M for eight separate ones. The win is ~9%, not
    8x — adapters share the card — but it is still the right default."""
    monkeypatch.setattr(av, "scan", lambda *a, **k: pytest.fail("rented!"))
    qq.submit("a.yaml", model="Qwen/Qwen3-4B")
    s = q.Scheduler(qq, clusters=lambda: [cluster(model="Qwen/Qwen3-4B")])
    [p] = s.plan()
    assert p.action == "pack" and p.cluster == "c1"


def test_a_different_model_never_packs(qq, monkeypatch):
    """Adapters on one server share a base model. Placing a 9B job on a 4B
    server trains against the wrong weights and looks fine until it doesn't."""
    monkeypatch.setattr(av, "scan", lambda *a, **k: [cap()])
    qq.submit("a.yaml", model="Qwen/Qwen3.5-9B")
    s = q.Scheduler(qq, clusters=lambda: [cluster(model="Qwen/Qwen3-4B")])
    [p] = s.plan()
    assert p.action == "rent"


def test_a_full_cluster_does_not_take_more(qq, monkeypatch):
    """Megatron allocates adapter slots up front; overflowing fails inside
    swap_to_adapter with a CUDA error that mentions nothing about adapters."""
    monkeypatch.setattr(av, "scan", lambda *a, **k: [cap()])
    qq.submit("a.yaml", model="m")
    s = q.Scheduler(qq, clusters=lambda: [cluster(max_adapters=4, used=4)])
    assert s.plan()[0].action == "rent"


def test_two_jobs_in_one_tick_cannot_share_the_last_slot(qq, monkeypatch):
    """A race that only appears under load: both jobs get promised the same
    free slot, and the second dies deep inside the run."""
    monkeypatch.setattr(av, "scan", lambda *a, **k: [cap()])
    qq.submit("a.yaml", model="m")
    qq.submit("b.yaml", model="m")
    s = q.Scheduler(qq, clusters=lambda: [cluster(max_adapters=1)])
    actions = [p.action for p in s.plan()]
    assert actions.count("pack") == 1 and actions.count("rent") == 1


def test_a_job_needing_several_slots_needs_them_all_free(qq, monkeypatch):
    monkeypatch.setattr(av, "scan", lambda *a, **k: [cap()])
    qq.submit("a.yaml", model="m", adapters=4)
    s = q.Scheduler(qq, clusters=lambda: [cluster(max_adapters=8, used=6)])
    assert s.plan()[0].action == "rent"


# -- routing across vendors --------------------------------------------------


def test_rents_the_cheapest_per_gpu_hour_across_vendors(qq, monkeypatch):
    table = {"H200": [cap(provider="pricey", gpu="H200", usd=3.0)],
             "H100": [cap(provider="cheap", gpu="H100", usd=1.1)]}
    monkeypatch.setattr(av, "scan", lambda gpu, n, **k: table.get(gpu, []))
    qq.submit("a.yaml", model="m", gpus=["H200", "H100"])
    [p] = q.Scheduler(qq).plan()
    assert p.action == "rent" and p.capacity.provider == "cheap"


def test_price_is_compared_per_gpu_not_per_machine(qq, monkeypatch):
    table = {"H200": [cap(provider="bulk", gpu="H200", count=8, usd=11.2)],
             "H100": [cap(provider="single", gpu="H100", count=8, usd=24.0)]}
    monkeypatch.setattr(av, "scan", lambda gpu, n, **k: table.get(gpu, []))
    qq.submit("a.yaml", model="m", gpus=["H200", "H100"], count=8)
    [p] = q.Scheduler(qq).plan()
    assert p.capacity.provider == "bulk"


def test_unavailable_capacity_is_never_rented(qq, monkeypatch):
    dead = av.Capacity("v", "H200", 1, av.UNAVAILABLE, usd_hr=0.1)
    monkeypatch.setattr(av, "scan", lambda *a, **k: [dead])
    qq.submit("a.yaml", model="m")
    [p] = q.Scheduler(qq).plan()
    assert p.action == "wait"


def test_no_capacity_waits_rather_than_failing(qq, monkeypatch):
    """The reason this module exists: spot capacity that is gone now is
    usually back within the hour. A job must not die of it."""
    monkeypatch.setattr(av, "scan", lambda *a, **k: [])
    qq.submit("a.yaml", model="m")
    s = q.Scheduler(qq)
    s.tick()
    assert [j.state for j in qq.jobs()] == [q.QUEUED]


def test_only_enabled_vendors_are_asked(qq, monkeypatch):
    seen = []

    def fake(gpu, n, **k):
        seen.append(k.get("clouds_"))
        return []

    monkeypatch.setattr(av, "scan", fake)
    qq.submit("a.yaml", model="m", gpus=["H200"])
    q.Scheduler(qq, vendors=["verda"]).plan()
    assert seen == [["verda"]]


# -- executing ---------------------------------------------------------------


def test_placing_then_running_is_recorded(qq, monkeypatch):
    monkeypatch.setattr(av, "scan", lambda *a, **k: [cap()])
    qq.submit("a.yaml", model="m")
    s = q.Scheduler(qq, launch=lambda job, c: cluster(name="new"))
    s.tick()
    j = qq.jobs()[0]
    assert j.state == q.RUNNING and j.cluster == "new"


def test_capacity_vanishing_between_probe_and_launch_requeues(qq, monkeypatch):
    """Routine, not exceptional — availability answers expire in ~30s and
    every SKU that probed free was refused seconds later at some point."""
    monkeypatch.setattr(av, "scan", lambda *a, **k: [cap()])
    qq.submit("a.yaml", model="m")
    q.Scheduler(qq, launch=lambda job, c: None).tick()
    j = qq.jobs()[0]
    assert j.state == q.QUEUED and j.attempts == 1


def test_a_launch_that_raises_requeues_with_the_reason(qq, monkeypatch):
    monkeypatch.setattr(av, "scan", lambda *a, **k: [cap()])
    qq.submit("a.yaml", model="m")

    def boom(job, c):
        raise RuntimeError("no capacity")

    q.Scheduler(qq, launch=boom).tick()
    j = qq.jobs()[0]
    assert j.state == q.QUEUED and "no capacity" in j.error


def test_a_job_that_fails_forever_stops_being_retried(qq, monkeypatch):
    """Visible rather than silently looping — a job stuck on an unlaunchable
    shape should be findable, not quietly burning probe calls."""
    monkeypatch.setattr(av, "scan", lambda *a, **k: [cap()])
    j = qq.submit("a.yaml", model="m")
    qq.update(j, q.QUEUED, attempts=10)
    [p] = q.Scheduler(qq, max_attempts=10).plan()
    assert p.action == "wait" and "failed attempts" in p.reason


def test_planning_without_a_launcher_rents_nothing(qq, monkeypatch):
    """Dry-run: the plan is the useful artifact, and planning must not move
    the job — a dry run that mutates state is not a dry run."""
    monkeypatch.setattr(av, "scan", lambda *a, **k: [cap()])
    qq.submit("a.yaml", model="m")
    [p] = q.Scheduler(qq, launch=None).tick()
    assert p.action == "rent"
    assert qq.jobs()[0].state == q.QUEUED


def test_run_returns_once_the_queue_drains(qq, monkeypatch):
    monkeypatch.setattr(av, "scan", lambda *a, **k: [cap()])
    monkeypatch.setattr(q.time, "sleep", lambda s: None)
    qq.submit("a.yaml", model="m")
    q.Scheduler(qq, launch=lambda job, c: cluster(name="n")).run()
    assert qq.jobs()[0].state == q.RUNNING


def test_run_respects_a_deadline_with_work_left(qq, monkeypatch):
    """Must not spin forever when capacity never arrives."""
    monkeypatch.setattr(av, "scan", lambda *a, **k: [])
    monkeypatch.setattr(q.time, "sleep", lambda s: None)
    qq.submit("a.yaml", model="m")
    q.Scheduler(qq, poll_s=0).run(timeout_s=0.01)
    assert qq.jobs()[0].state == q.QUEUED


def test_run_once_does_a_single_pass(qq, monkeypatch):
    """One pass asks each acceptable accelerator exactly once, then stops —
    it does not loop waiting for capacity."""
    seen = []
    monkeypatch.setattr(av, "scan", lambda gpu, n, **k: seen.append(gpu) or [])
    monkeypatch.setattr(q.time, "sleep", lambda s: pytest.fail("slept"))
    qq.submit("a.yaml", model="m", gpus=["H200", "H100"])
    q.Scheduler(qq).run(once=True)
    assert seen == ["H200", "H100"]


def test_a_cheaper_accelerator_short_circuits_the_rest(qq, monkeypatch):
    """The first accelerator with stock ends that gpu's scan — scan() is
    already price-sorted, so reading past its first hit buys nothing."""
    seen = []

    def fake(gpu, n, **k):
        seen.append(gpu)
        return [cap(gpu=gpu), cap(gpu=gpu, usd=99.0)]

    monkeypatch.setattr(av, "scan", fake)
    qq.submit("a.yaml", model="m", gpus=["H200", "H100"])
    [p] = q.Scheduler(qq).plan()
    assert p.capacity.usd_hr == 2.0


def test_requeue_clears_the_cluster(qq):
    j = qq.submit("a.yaml", model="m")
    qq.update(j, q.RUNNING, cluster="c1")
    qq.requeue(j, "preempted")
    got = qq.jobs()[0]
    assert got.state == q.QUEUED and got.cluster is None and got.error == "preempted"


def test_default_gpu_order_leads_with_the_measured_cheapest():
    """RTX PRO 6000 measured $0.0320/M — cheapest per token of anything
    tested, and 23x under Tinker. It leads because of that, not its size."""
    assert q.DEFAULT_GPUS[0] == "RTXPRO6000"


class TestSkypilotLauncher:
    """The launcher must honour what the probe found, not re-plan."""

    def _fake_compute(self, monkeypatch, url="http://1.2.3.4:8000"):
        seen = {}

        class Fake:
            def __init__(self, **kw):
                seen.update(kw)

            def up(self):
                return url

        import evsys_sdk.compute.skypilot as sp
        monkeypatch.setattr(sp, "SkyPilotCompute", Fake)
        return seen

    def test_launches_where_the_probe_found_capacity(self, monkeypatch):
        """SkyPilot would otherwise re-plan from its catalog and pick a region
        we never checked for stock."""
        seen = self._fake_compute(monkeypatch)
        job = q.Job(config="a.yaml", model="Qwen/Qwen3-4B")
        c = q.skypilot_launcher()(job, cap(provider="verda", gpu="H200",
                                           region="FIN-03"))
        assert seen["infra"] == "verda/FIN-03"
        assert seen["accelerators"] == "H200:1"
        assert c is not None and c.url.startswith("http")

    def test_gpu_names_with_spaces_become_valid_accelerators(self, monkeypatch):
        """Verda reports "RTX PRO 6000"; SkyPilot needs RTXPRO6000:4."""
        seen = self._fake_compute(monkeypatch)
        job = q.Job(config="a.yaml", model="m")
        q.skypilot_launcher()(job, cap(gpu="RTX PRO 6000", count=4))
        assert seen["accelerators"] == "RTXPRO6000:4"

    def test_multi_lora_is_sized_for_the_adapters_it_will_host(self, monkeypatch):
        seen = self._fake_compute(monkeypatch)
        job = q.Job(config="a.yaml", model="m", adapters=8)
        q.skypilot_launcher()(job, cap())
        assert seen["multi_lora"] is True and seen["max_adapters"] == 8

    def test_a_single_adapter_job_does_not_pay_for_multi_lora(self, monkeypatch):
        seen = self._fake_compute(monkeypatch)
        q.skypilot_launcher()(q.Job(config="a.yaml", model="m"), cap())
        assert seen["multi_lora"] is False

    def test_the_queue_owns_retrying_not_the_launcher(self, monkeypatch):
        """Two retry loops would multiply: the queue must see each failure."""
        seen = self._fake_compute(monkeypatch)
        q.skypilot_launcher()(q.Job(config="a.yaml", model="m"), cap())
        assert seen["retry_until_up"] is False

    def test_no_endpoint_means_no_cluster(self, monkeypatch):
        self._fake_compute(monkeypatch, url="")
        assert q.skypilot_launcher()(q.Job(config="a.yaml", model="m"), cap()) is None

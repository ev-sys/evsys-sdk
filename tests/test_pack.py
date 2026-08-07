"""Pack: several experiments on one node, with the measured placement rules."""
from __future__ import annotations

import threading
import time

import pytest

from evsys_sdk.compute.pack import Pack, PackResult


class FakeCompute:
    """up()/down() with counters — the whole contract Pack needs."""

    def __init__(self, accelerators="H100:1", max_adapters=8):
        self.ups = 0
        self.downs = 0
        from types import SimpleNamespace
        self.cfg = SimpleNamespace(accelerators=accelerators,
                                   max_adapters=max_adapters)

    def up(self):
        self.ups += 1
        return "http://fake:8000/"

    def down(self):
        self.downs += 1


def test_one_up_one_down_for_n_experiments():
    """The economics of packing: N experiments, ONE provision, ONE teardown."""
    c = FakeCompute()
    Pack(c, model="m").run([lambda u: 1, lambda u: 2, lambda u: 3])
    assert (c.ups, c.downs) == (1, 1)


def test_results_in_submission_order_with_values():
    c = FakeCompute()
    out = Pack(c, model="m").run([lambda u: "a", lambda u: "b"])
    assert [r.value for r in out] == ["a", "b"]
    assert all(r.ok for r in out)


def test_every_experiment_sees_the_shared_url(monkeypatch):
    import os
    seen = []
    Pack(FakeCompute(), model="m").run([lambda u: seen.append(
        (u, os.environ.get("TINKER_BASE_URL")))])
    assert seen == [("http://fake:8000", "http://fake:8000")]


def test_one_failure_costs_one_result_not_four():
    """Four runs sharing a node are four independent experiments."""
    def boom(u):
        raise RuntimeError("bad config")

    out = Pack(FakeCompute(), model="m").run(
        [lambda u: 1, boom, lambda u: 3, lambda u: 4])
    assert [r.ok for r in out] == [True, False, True, True]
    assert isinstance(out[1].error, RuntimeError)


def test_teardown_runs_even_when_everything_fails():
    """A failed run that leaks a GPU bills until someone notices."""
    c = FakeCompute()

    def boom(u):
        raise RuntimeError("x")

    Pack(c, model="m").run([boom, boom])
    assert c.downs == 1


def test_raise_on_error_raises_after_teardown():
    c = FakeCompute()

    def boom(u):
        raise ValueError("cfg")

    with pytest.raises(ValueError):
        Pack(c, model="m").run([boom], raise_on_error=True)
    assert c.downs == 1, "the node must be down before the error propagates"


class TestPlacementRules:
    """Each rule here was measured, and violating it fails opaquely."""

    def _concurrent_peak(self, pack, n_jobs=8):
        active, peak, lock = [0], [0], threading.Lock()

        def job(u):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.03)
            with lock:
                active[0] -= 1

        pack.run([job] * n_jobs)
        return peak[0]

    def test_sft_defaults_to_the_measured_peak_of_4(self):
        p = Pack(FakeCompute(), model="m")
        assert self._concurrent_peak(p) <= 4

    def test_rl_on_one_gpu_is_serial(self):
        """Colocated multi-adapter RL fails outright: n=2 and n=4 error where
        n=1 succeeds. This is a correctness bound, not a tuning default."""
        p = Pack(FakeCompute(accelerators="H100:1"), model="m", rl=True)
        assert self._concurrent_peak(p, n_jobs=4) == 1

    def test_rl_on_one_gpu_caps_even_an_explicit_request(self):
        p = Pack(FakeCompute(accelerators="H100:1"), model="m", rl=True,
                 concurrency=4)
        assert self._concurrent_peak(p, n_jobs=4) == 1

    def test_rl_on_two_gpus_packs(self):
        p = Pack(FakeCompute(accelerators="A100:2"), model="m", rl=True)
        assert 1 < self._concurrent_peak(p, n_jobs=6) <= 4

    def test_adapter_slots_bound_concurrency(self):
        """max_cpu_loras is an LRU with no reload: an evicted adapter 404s.
        More experiments than slots must wait, not evict."""
        p = Pack(FakeCompute(max_adapters=2), model="m", concurrency=8)
        assert self._concurrent_peak(p, n_jobs=6) <= 2

    def test_gpu_count_inferred_from_accelerators(self):
        assert Pack(FakeCompute(accelerators="H200:2"), model="m").gpus == 2
        assert Pack(FakeCompute(accelerators="H100:1"), model="m").gpus == 1
        assert Pack(FakeCompute(accelerators=["A100:2", "H100:1"]),
                    model="m").gpus == 2

"""SkyPilot compute target — provisioning the machine a SkyRL server runs on.

The point of the split: a backend says which protocol to speak, a compute
target says whose hardware speaks it. Together they let the same config.yaml
run on someone else's infrastructure.

`sky` is faked throughout — these tests assert the orchestration, not
SkyPilot itself.
"""

from __future__ import annotations

import sys
import types

import pytest

from evsys_sdk.compute import ComputeError, build_compute
from evsys_sdk.compute.skypilot import SERVER_PORT, SkyPilotCompute
from evsys_sdk.registry import get_compute, list_computes

URL = f"http://1.2.3.4:{SERVER_PORT}"


class _FakeSky:
    """Enough of the sky SDK to drive the provider."""

    def __init__(self, *, endpoint_after: int = 1):
        self.launched: list[dict] = []
        self.downed: list[str] = []
        self._calls = 0
        self._endpoint_after = endpoint_after
        self.Resources = lambda **kw: ("resources", kw)
        self.jobs = _FakeSky._Jobs()
        self.Task = _FakeTask

    def launch(self, task, **kw):
        self.launched.append({"task": task, **kw})
        return "req-launch"

    class _Jobs:
        def __init__(self): self.launched = []
        def launch(self, task, **kw):
            self.launched.append({"task": task, **kw})
            return "req-job"

    def stream_and_get(self, rid):
        return rid

    def reset_endpoint(self):
        """Simulate the instance vanishing: the port stops answering."""
        self._calls = 0

    def endpoints(self, cluster, port=None):
        return ("endpoints", cluster, port)

    def get(self, rid):
        # `get` serves two callers: resolving the launch request, and reading
        # the endpoint. Only the latter is what "not ready yet" refers to.
        if rid == "req-launch":
            return (1, "handle")
        self._calls += 1
        if self._calls <= self._endpoint_after:
            raise RuntimeError("cluster not up yet")
        return {SERVER_PORT: f"1.2.3.4:{SERVER_PORT}"}

    def down(self, cluster, **kw):
        self.downed.append(cluster)
        return "req-down"


class _FakeTask:
    def __init__(self, name=None, setup=None, run=None, **kw):
        self.name, self.setup, self.run = name, setup, run
        self.resources = None

    def set_resources(self, r):
        self.resources = r


def _compute(monkeypatch, sky, *, answers=True, **params):
    params.setdefault("model", "Qwen/Qwen3-0.6B")
    c = SkyPilotCompute(**params)
    monkeypatch.setattr(c, "_sky", lambda: sky)
    monkeypatch.setattr(SkyPilotCompute, "_answers", staticmethod(lambda url, timeout=10: answers))
    monkeypatch.setattr("evsys_sdk.compute.skypilot.time.sleep", lambda s: None)
    return c


class TestRegistration:
    def test_selectable_from_yaml(self):
        assert "skypilot" in list_computes()
        assert get_compute("skypilot") is SkyPilotCompute

    def test_build_validates_params(self):
        c = build_compute({"kind": "skypilot", "params": {"model": "m", "infra": "aws"}})
        assert c.cfg.infra == "aws"
        with pytest.raises(Exception):
            build_compute({"kind": "skypilot", "params": {"model": "m", "typo": 1}})

    def test_missing_skypilot_says_how_to_install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sky", None)
        c = SkyPilotCompute(model="m")
        monkeypatch.setattr("builtins.__import__",
                            _raising_import("sky", monkeypatch))
        with pytest.raises(ComputeError, match="skypilot"):
            c.up()


def _raising_import(name, monkeypatch):
    real = __import__

    def _imp(mod, *a, **k):
        if mod == name:
            raise ImportError("no sky")
        return real(mod, *a, **k)
    return _imp


class TestUp:
    def test_launches_and_returns_the_server_url(self, monkeypatch):
        sky = _FakeSky()
        c = _compute(monkeypatch, sky, infra="aws", accelerators="L4:1")
        assert c.up() == URL

        (call,) = sky.launched
        assert call["cluster_name"] == "evsys-skyrl"
        # the run command IS the SkyRL tinker server, on the exposed port
        assert "skyrl.tinker.api" in call["task"].run
        assert f"--port {SERVER_PORT}" in call["task"].run
        assert "Qwen/Qwen3-0.6B" in call["task"].run

    def test_the_port_is_exposed_or_nothing_can_reach_it(self, monkeypatch):
        sky = _FakeSky()
        c = _compute(monkeypatch, sky)
        c.up()
        # Resources is now an ordered candidate LIST, even for a single ask.
        (_, kw), = sky.launched[0]["task"].resources
        assert kw["ports"] == [SERVER_PORT]

    def test_the_extra_follows_the_backend_not_the_gpu(self, monkeypatch):
        """Multi-tenant LoRA only exists on megatron; picking the extra from
        "is there a GPU?" installed the wrong stack and silently gave you a
        single-tenant server."""
        sky = _FakeSky()
        c = _compute(monkeypatch, sky, accelerators="H200:1", server_backend="megatron")
        c.up()
        task = sky.launched[0]["task"]
        assert "--extra megatron" in task.setup and "--extra megatron" in task.run

    def test_multi_tenant_knobs_reach_the_command_line(self, monkeypatch):
        """backend_config was absent from the Config entirely, so merge_lora /
        max_loras / max_cpu_loras could not be set at all."""
        sky = _FakeSky()
        c = _compute(monkeypatch, sky, accelerators="H200:1", server_backend="megatron",
                     backend_config={"trainer.policy.model.lora.max_loras": 4,
                                     "trainer.policy.megatron_config.lora_config.merge_lora": False})
        c.up()
        run = sky.launched[0]["task"].run
        assert "--backend-config" in run
        assert '"trainer.policy.model.lora.max_loras": 4' in run
        assert '"trainer.policy.megatron_config.lora_config.merge_lora": false' in run

    def test_cpu_only_falls_back_to_the_jax_extra(self, monkeypatch):
        """No accelerator is a legitimate config — the JAX backend runs on CPU,
        which is how a smoke test costs cents."""
        sky = _FakeSky()
        c = _compute(monkeypatch, sky, accelerators=None)
        c.up()
        task = sky.launched[0]["task"]
        assert "--extra jax" in task.run

    def test_waits_until_the_endpoint_actually_answers(self, monkeypatch):
        sky = _FakeSky(endpoint_after=3)   # not addressable for three polls
        c = _compute(monkeypatch, sky)
        assert c.up() == URL

    def test_a_running_cluster_is_reused_not_relaunched(self, monkeypatch):
        """Idempotent by contract — otherwise a second run pays for a second GPU."""
        sky = _FakeSky(endpoint_after=0)   # already addressable
        c = _compute(monkeypatch, sky)
        assert c.up() == URL
        assert sky.launched == []          # endpoint answered before launching

    def test_repeat_calls_do_not_relaunch(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky)
        c.up()
        n = len(sky.launched)
        assert c.up() == URL
        assert len(sky.launched) == n

    def test_a_server_that_never_answers_fails_loudly(self, monkeypatch):
        sky = _FakeSky(endpoint_after=0)
        c = _compute(monkeypatch, sky, answers=False, startup_timeout_s=0.05)
        with pytest.raises(ComputeError, match="did not answer"):
            c.up()


class TestCostSafety:
    """A leaked cluster bills by the hour; these are the guards."""

    def test_autostop_and_termination_are_on_by_default(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky)
        c.up()
        call = sky.launched[0]
        assert call["idle_minutes_to_autostop"] == 30
        assert call["down"] is True

    def test_autostop_cannot_be_disabled(self):
        with pytest.raises(Exception):
            SkyPilotCompute(model="m", idle_minutes_to_autostop=0)

    def test_credentials_are_not_uploaded_by_default(self):
        """Agent-authored code runs on this box."""
        assert SkyPilotCompute(model="m").cfg.remote_identity == "NO_UPLOAD"

    def test_down_terminates_the_cluster(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky)
        c.up()
        c.down()
        assert sky.downed == ["evsys-skyrl"]

    def test_teardown_false_keeps_it_for_reuse(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky, teardown=False)
        c.up()
        c.down()
        assert sky.downed == []

    def test_a_failing_teardown_never_raises(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky)
        c.up()
        monkeypatch.setattr(sky, "down", _boom)
        c.down()          # logged, not raised — teardown must not mask the run

    def test_context_manager_releases_on_error(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky)
        with pytest.raises(ValueError):
            with c:
                raise ValueError("run blew up")
        assert sky.downed == ["evsys-skyrl"]


def _boom(*a, **k):
    raise RuntimeError("api unreachable")


class TestBackendIntegration:
    """`compute:` on the backend is the whole user-facing story."""

    def test_the_backend_brings_compute_up_and_trains_against_it(self, monkeypatch):
        pytest.importorskip("tinker")
        from evsys_sdk.backends.skyrl import SkyRLBackend

        built: dict = {}

        class _Fake:
            def __init__(self, **p):
                built.update(p)

            def up(self):
                return URL

            def down(self):
                built["down"] = True

        monkeypatch.setattr("evsys_sdk.compute.build_compute", lambda spec: _Fake(**spec["params"]))
        monkeypatch.setattr("evsys_sdk.backends.skyrl._connect", lambda kw: kw)

        b = SkyRLBackend(compute={"kind": "skypilot", "params": {"infra": "k8s"}},
                         health_check=False)
        h = b.prepare(model={"name": "Qwen/Qwen3-0.6B"}, run_dir="/tmp/r")

        import os
        assert h["base_url"] == URL
        assert os.environ["TINKER_BASE_URL"] == URL       # the whole run follows
        # the run's model is passed down, so it cannot drift from the server's
        assert built["model"] == "Qwen/Qwen3-0.6B"
        assert built["infra"] == "k8s"

        b.teardown(h)
        assert built.get("down") is True

    def test_without_compute_it_uses_the_url_you_gave(self, monkeypatch):
        pytest.importorskip("tinker")
        from evsys_sdk.backends.skyrl import SkyRLBackend

        monkeypatch.setattr("evsys_sdk.backends.skyrl._connect", lambda kw: kw)
        b = SkyRLBackend(base_url="http://my-box:8000", health_check=False)
        h = b.prepare(model={"name": "m"}, run_dir="/tmp/r")
        assert h["base_url"] == "http://my-box:8000"
        b.teardown(h)      # nothing to release, must not explode


class TestAgainstTheRealSkyPilotApi:
    """Mocks accept anything. This checks the arguments we actually pass exist.

    `detach_run=True` looked fine against a `**kw` fake and would have raised
    TypeError on the first real launch.
    """

    def test_every_kwarg_we_pass_is_a_real_parameter(self):
        sky = pytest.importorskip("sky")
        import inspect

        launch = inspect.signature(sky.launch).parameters
        for kw in ("cluster_name", "idle_minutes_to_autostop", "down", "fast"):
            assert kw in launch, f"sky.launch has no '{kw}'"

        res = inspect.signature(sky.Resources.__init__).parameters
        for kw in ("infra", "accelerators", "cpus", "memory", "use_spot", "ports"):
            assert kw in res, f"sky.Resources has no '{kw}'"

        for fn in ("get", "endpoints", "down", "stream_and_get"):
            assert hasattr(sky, fn), f"sky.{fn} is gone"
        assert hasattr(sky.Task, "set_resources")

    def test_the_provider_passes_nothing_else(self):
        """Guards the inverse: a kwarg added here without checking the API."""
        import re

        from evsys_sdk.compute import skypilot as mod

        src = inspect_source(mod.SkyPilotCompute.up)
        passed = set(re.findall(r"^\s*(\w+)=", src, re.M))
        assert passed <= {"cluster_name", "idle_minutes_to_autostop", "down", "fast",
                          "retry_until_up", "name"}


def inspect_source(fn):
    import inspect
    return inspect.getsource(fn)


class TestCredentialUpload:
    """`remote_identity` was decorative until it was wired to SkyPilot's config
    — it read as a security setting while SkyPilot uploaded cloud credentials
    to a VM running agent-authored code."""

    def test_it_is_applied_around_the_launch(self, monkeypatch):
        sky = _FakeSky()
        c = _compute(monkeypatch, sky)
        seen: dict = {}

        import contextlib

        @contextlib.contextmanager
        def fake_override(overrides, *a, **k):
            seen["overrides"] = overrides
            yield

        monkeypatch.setattr("sky.skypilot_config.override_skypilot_config", fake_override)
        c.up()
        assert seen["overrides"]["aws"]["remote_identity"] == "NO_UPLOAD"
        assert seen["overrides"]["kubernetes"]["remote_identity"] == "NO_UPLOAD"

    def test_it_is_a_real_skypilot_config_key(self):
        """Guards against the setting being silently ignored by SkyPilot."""
        pytest.importorskip("sky")
        from sky.utils import schemas

        schema = schemas.get_config_schema()
        aws = schema["properties"]["aws"]["properties"]
        assert "remote_identity" in aws


class TestLifetimeDeadline:
    """PrimeIntellect supports no autostop, no autodown and no stop, so a
    cluster there bills until something explicitly terminates it. `down()` in
    teardown covers the happy path; these cover the host dying."""

    def test_a_deadline_is_armed_after_up(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky, max_lifetime_s=3600)
        c.up()
        assert c._timer is not None and c._armed

    def test_the_deadline_tears_the_cluster_down(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky, max_lifetime_s=0.05)
        c.up()
        # `_compute` stubs time.sleep on the real time module, so spin on the
        # clock instead of sleeping — a no-op sleep would race the timer.
        import time
        end = time.monotonic() + 5
        while not sky.downed and time.monotonic() < end:
            pass
        assert sky.downed == ["evsys-skyrl"]

    def test_down_cancels_the_timer_so_it_cannot_fire_later(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky, max_lifetime_s=3600)
        c.up()
        c.down()
        assert c._timer is None
        assert sky.downed == ["evsys-skyrl"]

    def test_default_lifetime_is_bounded(self):
        """A None default would reintroduce the leak on no-autostop clouds."""
        assert SkyPilotCompute(model="m").cfg.max_lifetime_s == 6 * 3600


class TestReclaimKwargs:
    """PrimeIntellect reports AUTOSTOP/AUTODOWN/STOP unsupported. Passing the
    autostop kwargs there does not get ignored — every candidate resource is
    rejected and the launch dies with ResourcesUnavailableError."""

    def test_autostop_is_dropped_on_a_cloud_without_it(self, monkeypatch):
        c = _compute(monkeypatch, _FakeSky(), infra="primeintellect")
        assert c._autostop_supported() is False
        assert c._reclaim_kwargs() == {"idle_minutes_to_autostop": None, "down": False}

    def test_autostop_is_kept_where_it_works(self, monkeypatch):
        c = _compute(monkeypatch, _FakeSky(), infra="aws", idle_minutes_to_autostop=15)
        assert c._autostop_supported() is True
        assert c._reclaim_kwargs() == {"idle_minutes_to_autostop": 15, "down": True}

    def test_unknown_infra_keeps_the_safe_default(self, monkeypatch):
        c = _compute(monkeypatch, _FakeSky(), infra="not-a-cloud")
        assert c._autostop_supported() is True

    def test_launch_uses_the_capability_aware_kwargs(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky, infra="primeintellect")
        c.up()
        assert sky.launched[0]["idle_minutes_to_autostop"] is None
        assert sky.launched[0]["down"] is False


class TestDeadlineOverridesTeardownFalse:
    """`teardown=False` means "leave it up for the next run". On a cloud with
    no autostop that would otherwise mean "leave it up forever" — the exact
    config the benchmark run uses."""

    def test_deadline_terminates_even_with_teardown_false(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky, teardown=False, infra="primeintellect",
                     max_lifetime_s=0.05)
        c.up()
        import time
        end = time.monotonic() + 5
        while not sky.downed and time.monotonic() < end:
            pass
        assert sky.downed == ["evsys-skyrl"]

    def test_ordinary_down_still_respects_teardown_false(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky, teardown=False, max_lifetime_s=3600)
        c.up()
        c.down()
        assert sky.downed == []


class TestRetryUntilUp:
    """Spot capacity comes and goes — one failed attempt says little."""

    def test_off_by_default(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky)
        c.up()
        assert sky.launched[0]["retry_until_up"] is False

    def test_passed_through_when_asked(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky, retry_until_up=True)
        c.up()
        assert sky.launched[0]["retry_until_up"] is True


class TestSurvivingPreemption:
    """A spot A100 was reclaimed mid-benchmark with no notice, taking its
    local checkpoints and SQLite metadata with it. These pin the parts that
    would have made that survivable — and the warning that would have said so
    beforehand."""

    def test_durable_paths_reach_the_server(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky,
                     checkpoints_path="s3://bucket/ckpt",
                     database_url="postgresql://h/db")
        run = c._task(sky).run
        assert "--checkpoints-base s3://bucket/ckpt" in run
        assert "--database-url postgresql://h/db" in run

    def test_nothing_is_passed_when_unset(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        run = _compute(monkeypatch, sky)._task(sky).run
        assert "--checkpoints-base" not in run and "--database-url" not in run

    @staticmethod
    def _warnings(monkeypatch):
        """Watch the module logger directly — the SDK logger stops propagating
        to root once configured, so caplog silently sees nothing."""
        seen: list[str] = []
        monkeypatch.setattr(
            "evsys_sdk.compute.skypilot.log.warning",
            lambda msg, *a, **k: seen.append(str(msg) % a if a else str(msg)))
        return seen

    def test_spot_without_durable_state_warns(self, monkeypatch):
        seen = self._warnings(monkeypatch)
        c = _compute(monkeypatch, _FakeSky(endpoint_after=1), use_spot=True,
                     price_check=False)
        c.up()
        assert any("loses the entire run" in m for m in seen), seen

    def test_no_warning_when_state_is_durable(self, monkeypatch):
        seen = self._warnings(monkeypatch)
        c = _compute(monkeypatch, _FakeSky(endpoint_after=1), use_spot=True,
                     price_check=False, checkpoints_path="s3://b/c",
                     database_url="postgresql://h/db")
        c.up()
        assert not any("loses the entire run" in m for m in seen), seen

    def test_on_demand_is_not_nagged(self, monkeypatch):
        seen = self._warnings(monkeypatch)
        c = _compute(monkeypatch, _FakeSky(endpoint_after=1), use_spot=False,
                     price_check=False)
        c.up()
        assert not any("loses the entire run" in m for m in seen), seen

    def test_healthy_is_false_once_the_instance_is_gone(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky)
        c.up()
        assert c.healthy() is True
        monkeypatch.setattr(SkyPilotCompute, "_answers", staticmethod(lambda u, timeout=10: False))
        assert c.healthy() is False

    def test_recover_relaunches(self, monkeypatch):
        """sky.launch builds an UNMANAGED cluster — SkyPilot does not bring it
        back by itself, so recovery has to relaunch."""
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky)
        c.up()
        assert len(sky.launched) == 1
        sky.reset_endpoint()
        assert c.recover()
        assert len(sky.launched) == 2


class TestFallbackAcrossGpusAndProviders:
    """One provider and one GPU is a single point of failure. The spot SKU we
    benchmarked on stopped existing mid-run — not "got expensive", ceased to
    be offered. Anything that fits the model will do."""

    @staticmethod
    def _priced(monkeypatch, prices, out_of_stock=()):
        """Stub live pricing: {(infra, accel): $/hr}; anything in
        `out_of_stock` is reported by the provider as unavailable; the rest
        is 'could not ask'."""
        def fake(self, i, a):
            if (i, a) in prices:
                return SkyPilotCompute._IN_STOCK, prices[(i, a)]
            if (i, a) in out_of_stock:
                return SkyPilotCompute._OUT_OF_STOCK, None
            return SkyPilotCompute._UNKNOWN, None
        monkeypatch.setattr(SkyPilotCompute, "_live_price", fake)

    def test_a_single_ask_still_produces_one_candidate(self, monkeypatch):
        c = _compute(monkeypatch, _FakeSky(), infra="aws", accelerators="H100:1")
        assert c._ordered_candidates() == [("aws", "H100:1")]

    def test_candidates_are_the_cross_product(self, monkeypatch):
        self._priced(monkeypatch, {})
        c = _compute(monkeypatch, _FakeSky(), infra=["a", "b"],
                     accelerators=["H100:1", "H200:1"])
        assert set(c._ordered_candidates()) == {
            ("a", "H100:1"), ("a", "H200:1"), ("b", "H100:1"), ("b", "H200:1")}

    def test_cheapest_live_price_wins_not_catalog_order(self, monkeypatch):
        """Declared order is H100 then H200; live prices invert it."""
        self._priced(monkeypatch, {("p", "H100:1"): 3.25, ("p", "H200:1"): 0.94})
        c = _compute(monkeypatch, _FakeSky(), infra="p",
                     accelerators=["H100:1", "H200:1"])
        assert c._ordered_candidates() == [("p", "H200:1"), ("p", "H100:1")]

    def test_unpriced_candidates_go_after_priced_in_declared_order(self, monkeypatch):
        """Unknown is not unavailable — we would rather try than refuse."""
        self._priced(monkeypatch, {("p", "H200:1"): 4.0})
        c = _compute(monkeypatch, _FakeSky(), infra="p",
                     accelerators=["A:1", "H200:1", "B:1"])
        assert c._ordered_candidates() == [("p", "H200:1"), ("p", "A:1"), ("p", "B:1")]

    def test_known_out_of_stock_ranks_below_merely_unknown(self, monkeypatch):
        """"The provider says there is none" is worse news than "we could not
        ask" — today it was the difference between a wasted launch and a real
        one, and SkyPilot reports both as ResourcesUnavailableError."""
        self._priced(monkeypatch, {("p", "C:1"): 1.0},
                     out_of_stock={("p", "A:1")})
        c = _compute(monkeypatch, _FakeSky(), infra="p",
                     accelerators=["A:1", "B:1", "C:1"])
        assert c._ordered_candidates() == [("p", "C:1"), ("p", "B:1"), ("p", "A:1")]

    def test_resources_is_a_list_so_skypilot_treats_it_as_ordered(self, monkeypatch):
        """A list is `ordered`; a set is `any_of` and lets the optimizer
        re-sort by catalog price — the price we know to be wrong."""
        self._priced(monkeypatch, {})
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky, infra="p", accelerators=["H100:1", "H200:1"])
        c.up()
        res = sky.launched[0]["task"].resources
        assert isinstance(res, list)
        assert [kw["accelerators"] for _, kw in res] == ["H100:1", "H200:1"]

    def test_price_ceiling_reaches_every_candidate(self, monkeypatch):
        """Without it a fallback chain can quietly escalate to a costly box."""
        self._priced(monkeypatch, {})
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky, infra="p", accelerators=["H100:1", "H200:1"],
                     max_hourly_cost=2.5)
        c.up()
        assert all(kw["max_hourly_cost"] == 2.5
                   for _, kw in sky.launched[0]["task"].resources)


class TestManagedJobs:
    def test_managed_uses_the_jobs_api(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky, managed=True, price_check=False)
        c.up()
        assert sky.jobs.launched and not sky.launched

    def test_unmanaged_uses_plain_launch(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky, managed=False, price_check=False)
        c.up()
        assert sky.launched and not sky.jobs.launched

    def test_recovery_strategy_reaches_resources(self, monkeypatch):
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky, managed=True, price_check=False,
                     job_recovery="EAGER_NEXT_REGION")
        c.up()
        (_, kw), = sky.jobs.launched[0]["task"].resources
        assert kw["job_recovery"] == "EAGER_NEXT_REGION"

    def test_recovery_strategy_is_not_sent_to_unmanaged_clusters(self, monkeypatch):
        """job_recovery is a managed-job concept; an unmanaged cluster has no
        controller to act on it."""
        sky = _FakeSky(endpoint_after=1)
        c = _compute(monkeypatch, sky, managed=False, price_check=False,
                     job_recovery="FAILOVER")
        c.up()
        (_, kw), = sky.launched[0]["task"].resources
        assert "job_recovery" not in kw

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
        self.Task = _FakeTask

    def launch(self, task, **kw):
        self.launched.append({"task": task, **kw})
        return "req-launch"

    def stream_and_get(self, rid):
        return rid

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
        _, kw = sky.launched[0]["task"].resources
        assert kw["ports"] == [SERVER_PORT]

    def test_gpu_request_selects_the_gpu_extra(self, monkeypatch):
        sky = _FakeSky()
        c = _compute(monkeypatch, sky, accelerators="A100:1")
        c.up()
        task = sky.launched[0]["task"]
        assert "--extra gpu" in task.setup and "--extra gpu" in task.run

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
        assert passed <= {"cluster_name", "idle_minutes_to_autostop", "down", "fast"}


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

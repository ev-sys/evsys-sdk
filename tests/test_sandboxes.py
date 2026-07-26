"""The sandbox extension point — base-class orchestration, the registry +
factory, and the built-in ``local`` provider running for real.

The ``e2b`` provider is covered structurally only (registration, Config
validation, lazy vendor import); an actual E2B spawn needs ``E2B_API_KEY`` and
lives in the gated smoke.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel, ValidationError

from evsys_sdk.registry import _sandboxes, get_sandbox, list_sandboxes, register_sandbox
from evsys_sdk.sandboxes import (
    BaseSandbox,
    SandboxSetupError,
    available_sandboxes,
    build_sandbox,
    resolve_envs,
)


class MemSandbox(BaseSandbox):
    """Minimal provider: the five methods, nothing else."""

    name = "mem"
    workdir = "/box"

    class Config(BaseModel):
        model_config = {"extra": "forbid"}
        flavor: str = "plain"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.fs: dict[str, str] = {}
        self.commands: list[str] = []
        self.started = self.killed = False
        self.exit_code = 0

    def start(self):
        self.started = True

    def write(self, path, content):
        self.fs[path] = content

    def read(self, path):
        return self.fs.get(path)

    def exec(self, cmd, *, timeout_s, cwd=None, on_line=None):
        self.commands.append(cmd)
        if on_line:
            on_line(f"out: {cmd}")
        return self.exit_code, f"out: {cmd}"

    def kill(self):
        self.killed = True


class TestBaseOrchestration:
    """stage / setup / collect are written once in the base — every provider,
    including ones the SDK has never seen, inherits them."""

    def test_stage_writes_under_the_providers_workdir(self):
        sbx = MemSandbox()
        sbx.stage({"prompt.txt": "seed", "skills/a/SKILL.md": "# a"})
        assert sbx.fs == {"/box/prompt.txt": "seed", "/box/skills/a/SKILL.md": "# a"}

    def test_collect_returns_only_changed_files(self, tmp_path):
        """An untouched file must NOT round-trip: host mtimes drive UI signals."""
        sbx = MemSandbox()
        baseline = {"prompt.txt": "seed", "notes.md": "unchanged"}
        sbx.stage(baseline)
        sbx.fs["/box/prompt.txt"] = "IMPROVED"          # the agent edited one file
        landed = sbx.collect(
            [("prompt.txt", tmp_path / "prompt.txt"), ("notes.md", tmp_path / "notes.md")],
            baseline,
        )
        assert landed == ["prompt.txt"]
        assert (tmp_path / "prompt.txt").read_text() == "IMPROVED"
        assert not (tmp_path / "notes.md").exists()

    def test_collect_skips_missing_files(self, tmp_path):
        sbx = MemSandbox()
        assert sbx.collect([("gone.txt", tmp_path / "gone.txt")], {}) == []

    def test_setup_required_raises_optional_reports(self):
        sbx = MemSandbox()
        sbx.exit_code = 1
        with pytest.raises(SandboxSetupError, match="setup_cmd failed"):
            sbx.setup("bad-install", required=True)

        seen: list[str] = []
        sbx.setup("bad-install", required=False, on_line=seen.append, label="sdk_install")
        assert seen and "sdk_install failed" in seen[0]

    def test_setup_noop_without_a_command(self):
        sbx = MemSandbox()
        sbx.setup(None)
        assert sbx.commands == []

    def test_context_manager_starts_and_tears_down(self):
        sbx = MemSandbox()
        with sbx as box:
            assert box.started and not box.killed
        assert sbx.killed

    def test_teardown_never_raises(self):
        class Exploding(MemSandbox):
            def kill(self):
                raise RuntimeError("provider hiccup")

        with Exploding():  # __exit__ swallows it — teardown is best-effort
            pass

    def test_unimplemented_methods_are_explicit(self):
        bare = BaseSandbox()
        for call in (lambda: bare.write("p", "c"), lambda: bare.read("p"),
                     lambda: bare.exec("cmd", timeout_s=1)):
            with pytest.raises(NotImplementedError):
                call()


class TestRegistryAndFactory:
    def test_builtins_are_registered(self):
        assert {"e2b", "local"} <= set(list_sandboxes())
        assert available_sandboxes() == list_sandboxes()
        assert get_sandbox("e2b").name == "e2b"

    def test_build_from_spec_dict_and_string(self):
        try:
            register_sandbox("t_mem")(type("TMem", (MemSandbox,), {}))
            a = build_sandbox({"kind": "t_mem", "params": {"flavor": "spicy"}})
            b = build_sandbox("t_mem")
            assert a.cfg.flavor == "spicy" and b.cfg.flavor == "plain"
            assert a.started and b.started
        finally:
            _sandboxes.unregister("t_mem")

    def test_build_can_defer_start(self):
        try:
            register_sandbox("t_mem2")(type("TMem2", (MemSandbox,), {}))
            sbx = build_sandbox("t_mem2", start=False)
            assert not sbx.started
        finally:
            _sandboxes.unregister("t_mem2")

    def test_params_are_validated_against_the_provider_config(self):
        try:
            register_sandbox("t_mem3")(type("TMem3", (MemSandbox,), {}))
            with pytest.raises(ValidationError):
                build_sandbox({"kind": "t_mem3", "params": {"flavour": "typo"}})
        finally:
            _sandboxes.unregister("t_mem3")

    def test_unknown_kind_lists_what_is_available(self):
        with pytest.raises(KeyError, match=r"Available:.*e2b"):
            build_sandbox("does_not_exist")

    def test_resolve_envs_passes_through_only_what_is_set(self, monkeypatch):
        monkeypatch.setenv("EVSYS_TEST_KEY", "abc")
        monkeypatch.delenv("EVSYS_TEST_MISSING", raising=False)
        assert resolve_envs(["EVSYS_TEST_KEY", "EVSYS_TEST_MISSING"]) == {
            "EVSYS_TEST_KEY": "abc"}
        assert resolve_envs(None) == {}


class TestLocalProvider:
    """The zero-dependency provider — a real staging dir and real subprocesses."""

    def test_full_copy_in_run_copy_out_cycle(self, tmp_path):
        host_artifact = tmp_path / "prompt.txt"
        host_artifact.write_text("seed")
        baseline = {"prompt.txt": "seed"}

        with build_sandbox("local") as sbx:
            work = Path(sbx.workdir)
            assert work.is_dir()
            sbx.stage(baseline)
            assert (work / "prompt.txt").read_text() == "seed"

            # an "agent" rewrites the artifact through a shell command
            lines: list[str] = []
            code, out = sbx.exec("printf 'IMPROVED' > prompt.txt && echo done",
                                 timeout_s=30, on_line=lines.append)
            assert code == 0 and "done" in out and lines == ["done"]

            assert sbx.collect([("prompt.txt", host_artifact)], baseline) == ["prompt.txt"]

        assert host_artifact.read_text() == "IMPROVED"
        assert not work.exists()  # scratch dir removed on teardown

    def test_nonzero_exit_is_reported_not_raised(self):
        with build_sandbox("local") as sbx:
            code, out = sbx.exec("echo boom >&2; exit 3", timeout_s=30)
        assert code == 3 and "boom" in out

    def test_env_passthrough_reaches_the_command(self):
        with build_sandbox("local", envs={"EVSYS_SECRET": "s3cr3t"}) as sbx:
            _, out = sbx.exec("printf '%s' \"$EVSYS_SECRET\"", timeout_s=30)
        assert out.strip() == "s3cr3t"

    def test_inherit_env_false_hides_the_host_environment(self, monkeypatch):
        monkeypatch.setenv("EVSYS_HOST_ONLY", "leaked")
        with build_sandbox({"kind": "local", "params": {"inherit_env": False}}) as sbx:
            _, out = sbx.exec("printf '%s' \"${EVSYS_HOST_ONLY:-}\"", timeout_s=30)
        assert out.strip() == ""

    def test_timeout_is_reported_as_124(self):
        with build_sandbox("local") as sbx:
            code, out = sbx.exec("sleep 5", timeout_s=0.3)
        assert code == 124 and "timed out" in out

    def test_keep_and_explicit_workdir(self, tmp_path):
        target = tmp_path / "scratch"
        with build_sandbox({"kind": "local",
                            "params": {"workdir": str(target), "keep": True}}) as sbx:
            sbx.stage({"a.txt": "hi"})
        assert (target / "a.txt").read_text() == "hi"   # survives teardown

    def test_read_missing_file_is_none(self):
        with build_sandbox("local") as sbx:
            assert sbx.read(sbx.path("nope.txt")) is None


class _FakeE2BSandbox:
    """Stands in for ``e2b.Sandbox`` — records how the adapter drives the SDK."""

    created: ClassVar[list[tuple[tuple, dict]]] = []

    def __init__(self, *args, **kwargs):
        _FakeE2BSandbox.created.append((args, kwargs))
        self.files = self
        self.commands = self
        self._files: dict[str, str] = {}
        self.killed = False
        self.run_kwargs: dict = {}

    # files.write / files.read
    def write(self, path, content):
        self._files[path] = content

    def read(self, path):
        if path not in self._files:
            raise FileNotFoundError(path)
        return self._files[path]

    # commands.run
    def run(self, cmd, **kw):
        self.run_kwargs = dict(kw, cmd=cmd)
        for cb_name in ("on_stdout", "on_stderr"):
            if kw.get(cb_name):
                kw[cb_name](f"{cb_name}: {cmd}")
        return type("R", (), {"stdout": "out", "stderr": "err", "exit_code": 7})()

    def kill(self):
        self.killed = True


@pytest.fixture
def fake_e2b(monkeypatch):
    """Install a stub ``e2b`` module so the adapter can be exercised without
    the optional extra or an API key."""
    import sys
    import types

    _FakeE2BSandbox.created = []
    mod = types.ModuleType("e2b")
    mod.Sandbox = _FakeE2BSandbox
    monkeypatch.setitem(sys.modules, "e2b", mod)
    return _FakeE2BSandbox


class TestE2BProvider:
    def test_config_surface(self):
        cls = get_sandbox("e2b")
        cfg = cls.Config(template="evsys-agent", metadata={"run": "1"})
        assert cfg.template == "evsys-agent" and cfg.metadata == {"run": "1"}
        with pytest.raises(ValidationError):
            cls.Config(templat="typo")

    def test_construction_does_not_touch_the_vendor_sdk(self):
        """Building the object must not import `e2b` — it is an optional extra,
        so `evsys list sandboxes` works without it installed."""
        sbx = get_sandbox("e2b")(envs={}, timeout_s=60, template="x")
        assert sbx.cfg.template == "x"
        sbx.kill()  # a no-op before start(), not an AttributeError

    def test_start_passes_template_envs_and_timeout_grace(self, fake_e2b):
        sbx = build_sandbox({"kind": "e2b", "params": {"template": "tpl",
                                                       "metadata": {"run": "1"}}},
                            envs={"ANTHROPIC_API_KEY": "k"}, timeout_s=300)
        args, kwargs = fake_e2b.created[-1]
        assert args == ("tpl",)                       # template is positional
        assert kwargs["envs"] == {"ANTHROPIC_API_KEY": "k"}
        assert kwargs["metadata"] == {"run": "1"}
        assert kwargs["timeout"] == 300 + 120         # TTL head-room over the run
        sbx.kill()

    def test_start_without_a_template_uses_the_default_image(self, fake_e2b):
        sbx = build_sandbox("e2b", timeout_s=60)
        args, kwargs = fake_e2b.created[-1]
        assert args == () and "metadata" not in kwargs
        sbx.kill()

    def test_file_and_command_round_trip(self, fake_e2b):
        lines: list[str] = []
        with build_sandbox("e2b", envs={"K": "v"}, timeout_s=60) as sbx:
            inner = sbx._sbx
            sbx.stage({"prompt.txt": "seed"})
            assert inner._files == {f"{sbx.workdir}/prompt.txt": "seed"}
            assert sbx.read(sbx.path("prompt.txt")) == "seed"
            assert sbx.read(sbx.path("missing.txt")) is None   # not an exception
            code, out = sbx.exec("run me", timeout_s=42, cwd="/box", on_line=lines.append)

        assert (code, out) == (7, "out\nerr")          # exit code + merged streams
        assert lines == ["on_stdout: run me", "on_stderr: run me"]
        assert inner.run_kwargs["timeout"] == 42 and inner.run_kwargs["cwd"] == "/box"
        assert inner.run_kwargs["envs"] == {"K": "v"}
        assert inner.killed and sbx._sbx is None       # torn down, handle dropped

    def test_kill_is_best_effort(self, fake_e2b):
        sbx = build_sandbox("e2b", timeout_s=60)
        sbx._sbx.kill = lambda: (_ for _ in ()).throw(RuntimeError("api down"))
        sbx.kill()  # must not raise — teardown is best-effort
        sbx.kill()  # idempotent

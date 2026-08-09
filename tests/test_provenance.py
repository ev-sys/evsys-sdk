"""Trigger provenance — the escalation → experiment link, and the sandbox
results sync that brings an agent's experiments home before the box dies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from evsys_sdk.provenance import (
    AGENT_AUTORESEARCH,
    AGENT_TRIGGER,
    ENV_AGENT,
    ENV_ESCALATION,
    current_trigger,
    trigger_env,
    trigger_tags,
)


class TestContext:
    def test_env_round_trip(self):
        env = trigger_env("/p/.evsys/triggers/escalations/escalation-000015.json",
                          agent=AGENT_AUTORESEARCH, agent_run="run7", sandbox="e2b")
        assert env[ENV_ESCALATION] == "escalation-000015"   # stem, not the path
        assert env[ENV_AGENT] == AGENT_AUTORESEARCH
        ctx = current_trigger(env)
        assert ctx == {"escalation": "escalation-000015", "agent": "autoresearch",
                       "agent_run": "run7", "sandbox": "e2b"}

    def test_absent_when_not_agent_spawned(self):
        """A human running `evsys run` by hand gets no stamp — not a fake one."""
        assert current_trigger({}) is None
        assert current_trigger({"PATH": "/usr/bin"}) is None

    def test_partial_context_still_reads(self):
        assert current_trigger({ENV_AGENT: AGENT_TRIGGER}) == {"agent": "trigger"}

    def test_tags_make_it_filterable(self):
        tags = trigger_tags({"agent": "autoresearch", "escalation": "escalation-7"})
        assert tags == ["autoresearch", "escalation:escalation-7"]
        assert trigger_tags(None) == []

    def test_env_omits_unset_fields(self):
        env = trigger_env(None, agent=AGENT_TRIGGER)
        assert env == {ENV_AGENT: AGENT_TRIGGER}


class TestExperimentStamp:
    def test_experiment_records_the_escalation(self, monkeypatch):
        """The whole point: an experiment on disk says which escalation
        caused it, so the UI can scope to one autoresearch run."""
        from evsys_sdk.training.callbacks import EvsysLoggerCallback, LogContext

        monkeypatch.setenv(ENV_ESCALATION, "escalation-000015")
        monkeypatch.setenv(ENV_AGENT, AGENT_AUTORESEARCH)

        created: dict = {}

        class _Store:
            def create_experiment(self, **kw):
                created.update(kw)
                return {"id": "exp1"}

        cb = EvsysLoggerCallback()
        cb._store = _Store()
        monkeypatch.setattr(cb, "_ensure_store", lambda ctx: cb._store)

        class _Cfg:
            name = "distill-0727"
            metadata: ClassVar[dict] = {"hypothesis": "raise depth"}

        ctx = LogContext(output_dir=Path("."), config=_Cfg())
        cb.on_experiment_start(ctx)

        assert created["config"]["trigger"]["escalation"] == "escalation-000015"
        assert created["config"]["trigger"]["agent"] == "autoresearch"
        assert "escalation:escalation-000015" in created["tags"]
        assert ctx.ids["experiment_id"] == "exp1"

    def test_unstamped_without_an_agent(self, monkeypatch):
        from evsys_sdk.training.callbacks import EvsysLoggerCallback, LogContext

        monkeypatch.delenv(ENV_ESCALATION, raising=False)
        monkeypatch.delenv(ENV_AGENT, raising=False)
        created: dict = {}

        class _Store:
            def create_experiment(self, **kw):
                created.update(kw)
                return {"id": "exp1"}

        cb = EvsysLoggerCallback()
        cb._store = _Store()
        monkeypatch.setattr(cb, "_ensure_store", lambda ctx: cb._store)

        class _Cfg:
            name = "manual"
            metadata: ClassVar[dict] = {}

        cb.on_experiment_start(LogContext(output_dir=Path("."), config=_Cfg()))
        assert created.get("config") is None
        assert created.get("tags") is None


class TestSpawnStampsTheEnv:
    def test_local_spawn_passes_provenance(self, tmp_path, monkeypatch):
        import evsys_sdk.triggers.agent as agentmod
        from evsys_sdk.config import TriggerAgentConfig

        root = tmp_path / ".evsys" / "triggers"
        (root / "escalations").mkdir(parents=True)
        esc = root / "escalations" / "escalation-000015.json"
        esc.write_text("{}")

        seen: dict = {}

        def fake(cmd, *, cwd, log_file, detach, env=None):
            seen.update(env or {})
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text("")
            return None

        monkeypatch.setattr(agentmod, "_LAUNCH", fake)
        agentmod.spawn(esc, agent_cfg=TriggerAgentConfig(enabled=True),
                       root=root, cwd=tmp_path)
        assert seen[ENV_ESCALATION] == "escalation-000015"
        assert seen[ENV_AGENT] == AGENT_TRIGGER


class TestSandboxResultsSync:
    """An agent that runs experiments INSIDE a sandbox must not lose them when
    the box dies — the fixed artifact list cannot name generated experiment ids."""

    def _sandbox(self, files: dict[str, str]):
        from evsys_sdk.sandboxes import BaseSandbox

        class _Box(BaseSandbox):
            name = "t_sync"
            workdir = "/box"

            def __init__(self, **kw):
                super().__init__(**kw)
                self.fs = {f"/box/{k}": v for k, v in files.items()}

            def write(self, path, content):
                self.fs[path] = content

            def read(self, path):
                return self.fs.get(path)

            def exec(self, cmd, *, timeout_s, cwd=None, on_line=None):
                # emulate `find <dir> -type f`
                target = cmd.split("find ")[1].split(" -type")[0].strip("'\"").rstrip("/")
                prefix = target + "/"
                hits = [p for p in sorted(self.fs) if p.startswith(prefix)]
                return 0, "\n".join(hits) + "\n"

        return _Box()

    def test_whole_experiment_tree_comes_back(self, tmp_path):
        box = self._sandbox({
            "evsys_sdk/experiments/e1/experiment.json": '{"id": "e1"}',
            "evsys_sdk/generations/r1/generation.json": '{"id": "r1"}',
            "evsys_sdk/generations/r1/metrics.jsonl": '{"step": 1}\n',
            "evsys_sdk/generations/r1/predictions.jsonl": '{"kind": "train"}\n',
            "prompt.txt": "unrelated",
        })
        landed = box.collect_tree("evsys_sdk", tmp_path)
        assert len(landed) == 4
        assert (tmp_path / "evsys_sdk/experiments/e1/experiment.json").read_text() == '{"id": "e1"}'
        assert json.loads((tmp_path / "evsys_sdk/generations/r1/generation.json").read_text())["id"] == "r1"
        assert not (tmp_path / "prompt.txt").exists()   # outside the tree

    def test_unchanged_files_do_not_round_trip(self, tmp_path):
        box = self._sandbox({"evsys_sdk/a.json": "same", "evsys_sdk/b.json": "new"})
        landed = box.collect_tree("evsys_sdk", tmp_path, baseline={"evsys_sdk/a.json": "same"})
        assert landed == ["evsys_sdk/b.json"]

    def test_byte_budget_is_enforced(self, tmp_path):
        box = self._sandbox({f"evsys_sdk/f{i}.json": "x" * 100 for i in range(10)})
        landed = box.collect_tree("evsys_sdk", tmp_path, max_bytes=250)
        assert len(landed) == 2          # a sandbox must not be able to fill our disk

    def test_missing_tree_is_empty_not_an_error(self, tmp_path):
        box = self._sandbox({"prompt.txt": "x"})
        assert box.collect_tree("evsys_sdk", tmp_path) == []

    def test_live_sync_lands_results_while_the_agent_still_runs(self, tmp_path):
        """The sync used to run once, at teardown — so a 40-minute agent showed
        nothing at all until it finished. It runs on a timer now."""
        import threading

        from evsys_sdk.triggers import remote as rm

        box = self._sandbox({"evsys_sdk/experiments/e1/experiment.json": '{"id": "e1"}'})
        stop = threading.Event()
        monkey = rm.SYNC_EVERY_S
        rm.SYNC_EVERY_S = 0.01
        try:
            t = threading.Thread(target=rm._live_sync, args=(box, tmp_path, {}, stop))
            t.start()
            landed = tmp_path / "evsys_sdk/experiments/e1/experiment.json"
            for _ in range(200):                      # ≤2s, no sleep-and-hope
                if landed.exists():
                    break
                stop.wait(0.01)
            stop.set()
            t.join(2)
        finally:
            rm.SYNC_EVERY_S = monkey
        assert landed.read_text() == '{"id": "e1"}'   # arrived mid-run, not at the end

    def test_live_sync_survives_a_broken_sandbox(self, tmp_path):
        """A sync failure must never take down the agent it is watching."""
        import threading

        from evsys_sdk.triggers import remote as rm

        class _Dead:
            envs: ClassVar[dict] = {}

            def collect_tree(self, *a, **k):
                raise RuntimeError("sandbox went away")

            def list_tree(self, *a, **k):
                raise RuntimeError("sandbox went away")

        stop = threading.Event()
        monkey = rm.SYNC_EVERY_S
        rm.SYNC_EVERY_S = 0.01
        try:
            t = threading.Thread(target=rm._live_sync, args=(_Dead(), tmp_path, {}, stop))
            t.start()
            stop.wait(0.05)
            stop.set()
            t.join(2)
        finally:
            rm.SYNC_EVERY_S = monkey
        assert not t.is_alive()

    def test_mirror_is_pinned_to_the_live_workdir(self, tmp_path):
        """``EVSYS_LOG_DIR`` must follow the sandbox that is actually running.

        It used to be built from the module's default workdir, so on any
        provider that stages elsewhere (``local``'s scratch dir, ``modal`` with
        ``user:``) the mirror pointed at a directory the agent never wrote to —
        and every experiment it ran died with the box.
        """
        from evsys_sdk.triggers.remote import _pin_mirror

        box = self._sandbox({})
        box.workdir = "/home/agent/evsys"
        _pin_mirror(box)
        assert box.envs["EVSYS_LOG_DIR"] == "/home/agent/evsys/evsys_sdk"

    def test_files_the_agent_created_come_home(self, tmp_path):
        """`collect` only round-trips paths that were STAGED. A script the agent
        wrote, the dataset it curated and its run log were invisible — the
        actual evidence of what it did died with the sandbox."""
        from evsys_sdk.triggers.remote import _collect_new_files

        box = self._sandbox({
            "gate.py": "staged",                       # staged in, unchanged
            "my_experiment.py": "the agent wrote this",
            "rl_tasks.jsonl": '{"task_id": "t1"}\n',
            "experiment.log": "EXPERIMENT DONE",
            "evsys_sdk/experiments/e1/experiment.json": "{}",   # nested: tree sync
        })
        landed = _collect_new_files(box, tmp_path, {"gate.py": "staged"})
        assert sorted(landed) == ["experiment.log", "my_experiment.py", "rl_tasks.jsonl"]
        assert (tmp_path / "my_experiment.py").read_text() == "the agent wrote this"
        assert not (tmp_path / "gate.py").exists()          # unchanged, no round-trip
        assert not (tmp_path / "evsys_sdk").exists()        # left to collect_tree

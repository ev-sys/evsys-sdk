"""Tests for the local observability UI (`evsys ui`) — the state collector and
the HTTP surface, over a fixture project shaped like a real daemon run."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from evsys_sdk.config import SystemConfig
from evsys_sdk.ui.server import _make_handler, _parse_duration_s, _tail_jsonl, collect_state

# ---------------------------------------------------------------------------
# fixture project: system.yaml + gate.py + prompt.txt + a daemon's .evsys/
# ---------------------------------------------------------------------------

SYSTEM_YAML = {
    "traces": {
        "trace_sources": [
            {"kind": "langgraph", "params": {"project_name": "support-bot"}, "pull_every": "4s"}
        ]
    },
    "trigger": {
        "kind": "failing_answers",
        "import_path": "gate.py",
        "params": {"threshold": 0.4},
        "every_n": 5,
        "window": 12,
    },
}

GATE_SRC = "# my gate fn\nclass FailingAnswers: ...\n"


def _trace(i: int, score: float | None) -> dict:
    feedback = [] if score is None else [
        {"key": "correct", "score": score, "comment": "graded", "source": "langsmith", "turn": None}
    ]
    return {
        "trace_id": f"trace-{i:04d}",
        "messages": [
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": f"answer {i}"},
        ],
        "feedback": feedback,
        "metadata": {"source": "langgraph", "timestamp": f"2026-07-19 12:00:{i:02d}+00:00"},
    }


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "system.yaml").write_text(yaml.safe_dump(SYSTEM_YAML))
    (tmp_path / "gate.py").write_text(GATE_SRC)
    (tmp_path / "prompt.txt").write_text("You are a helpful assistant.")

    tr = tmp_path / ".evsys" / "traces" / "langgraph"
    tr.mkdir(parents=True)
    rows = [_trace(0, 1.0), _trace(1, 0.0), _trace(2, None)]
    (tr / "traces.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    trg = tmp_path / ".evsys" / "triggers"
    for sub in ("escalations", "verdicts", "agent-runs"):
        (trg / sub).mkdir(parents=True)
    (trg / "policy.json").write_text(json.dumps(
        {"kind": "failing_answers", "import_path": "gate.py",
         "params": {"threshold": 0.4, "cooldown": 30}, "every_n": 5, "window": 12}))
    (trg / "state.json").write_text(json.dumps(
        {"window": [_trace(9, 0.0)], "counters": {"n_seen": 15, "since_last_eval": 0}}))
    (trg / "log.jsonl").write_text(
        json.dumps({"event": "evaluate", "kind": "failing_answers", "escalate": False,
                    "reason": "only 3 traces", "signal": {}, "n_seen": 5}) + "\n"
        + json.dumps({"event": "evaluate", "kind": "failing_answers", "escalate": True,
                      "reason": "failure 100% >= 40%", "signal": {"failure_rate": 1.0}, "n_seen": 15}) + "\n"
        + json.dumps({"event": "spawn", "escalation": "escalation-00000015.json", "pid": 123}) + "\n"
        + "{torn line"  # daemon was mid-append — must be tolerated
    )
    (trg / "escalations" / "escalation-00000015.json").write_text(json.dumps(
        {"reason": "failure 100% >= 40%", "signal": {"failure_rate": 1.0},
         "trace_ids": ["trace-0001"], "kind": "failing_answers", "n_seen": 15}))
    (trg / "verdicts" / "escalation-00000015.json").write_text(json.dumps(
        {"worth_autoresearch": True, "reasoning": "real recurring failure", "hypothesis": "format gap"}))
    (trg / "agent-runs" / "escalation-00000015.log").write_text("Done. Rewrote prompt.txt.")
    return tmp_path


def _cfg() -> SystemConfig:
    return SystemConfig(**SYSTEM_YAML)


# ---------------------------------------------------------------------------
# collect_state
# ---------------------------------------------------------------------------

def test_collect_state_traces(project: Path) -> None:
    s = collect_state(project, _cfg())
    assert s["project"]["name"] == project.name
    assert s["traces"]["total"] == 3
    assert s["traces"]["sources"] == ["langgraph"]
    assert [t["trace_id"] for t in s["traces"]["items"]] == ["trace-0000", "trace-0001", "trace-0002"]


def test_collect_state_trigger_and_gate(project: Path) -> None:
    s = collect_state(project, _cfg())
    events = [line["event"] for line in s["trigger"]["log"]]
    assert events == ["evaluate", "evaluate", "spawn"]  # torn tail line dropped
    assert s["trigger"]["state"]["counters"]["n_seen"] == 15
    assert s["trigger"]["state"]["window_size"] == 1
    # policy comes from the live policy.json (agent-retuned cooldown), not the YAML seed
    assert s["gate"]["policy"]["params"]["cooldown"] == 30
    assert s["gate"]["source"] == GATE_SRC
    assert s["gate"]["source_path"] == "gate.py"


def test_collect_state_escalation_join(project: Path) -> None:
    s = collect_state(project, _cfg())
    (e,) = s["escalations"]
    assert e["id"] == "escalation-00000015"
    assert e["event"]["trace_ids"] == ["trace-0001"]
    assert e["verdict"]["worth_autoresearch"] is True
    assert e["agent_log"] == "Done. Rewrote prompt.txt."


def test_prompt_diff_against_snapshot(project: Path) -> None:
    trg = project / ".evsys" / "triggers"
    # no snapshot → no diff
    s = collect_state(project, _cfg())
    assert s["prompt"]["diff"] is None and s["prompt"]["diff_base"] is None
    # snapshot identical to the live prompt → still no diff
    snaps = trg / "prompt-snapshots"
    snaps.mkdir()
    (snaps / "escalation-00000015.txt").write_text("You are a helpful assistant.")
    s = collect_state(project, _cfg())
    assert s["prompt"]["diff"] is None
    assert s["escalations"][0]["prompt_before"] == "You are a helpful assistant."
    # live prompt rewritten → unified diff against the escalation-time snapshot
    (project / "prompt.txt").write_text("You are a helpful assistant.\nEnd with ANSWER: <integer>.")
    s = collect_state(project, _cfg())
    assert s["prompt"]["diff_base"] == "escalation-00000015"
    assert "+End with ANSWER: <integer>." in s["prompt"]["diff"]
    assert "prompt.txt @ escalation-00000015" in s["prompt"]["diff"]


def _write_oa_phase(run_dir: Path, stem: str, scores: list[float]) -> None:
    d = run_dir / stem / "evals"
    d.mkdir(parents=True, exist_ok=True)
    for i, sc in enumerate(scores, 1):
        (d / f"{i}.json").write_text(json.dumps({"eval": i, "score": sc, "candidate": "p"}))


def test_collect_optimizations_phases_and_best(project: Path) -> None:
    run = project / "oa_omni"
    _write_oa_phase(run, "oa-explore-gepa", [0.5, 0.7])
    _write_oa_phase(run, "oa-explore-best_of_n", [0.4])
    _write_oa_phase(run, "oa-gepa", [0.6, 0.8])
    (run / "oa-gepa" / "evals" / "torn.json").write_text("{not json")  # mid-write: skipped
    (run / "oa_summary.json").write_text(json.dumps({"engine": "gepa", "phases": []}))

    s = collect_state(project, _cfg())
    (r,) = s["optimizations"]
    assert r["run"] == "oa_omni" and r["done"] is True
    by = {(p["engine"], p["phase"]): p for p in r["phases"]}
    assert by[("gepa", "explore")]["points"][-1]["best"] == 0.7  # best-so-far, not raw
    assert by[("best_of_n", "explore")]["points"] == [{"eval": 1, "score": 0.4, "best": 0.4}]
    assert [q["best"] for q in by[("gepa", "main")]["points"]] == [0.6, 0.8]


def test_collect_optimizations_running_run_not_done(project: Path) -> None:
    _write_oa_phase(project / "oa_live", "oa-gepa", [0.5])  # no oa_summary.json yet
    s = collect_state(project, _cfg())
    (r,) = s["optimizations"]
    assert r["done"] is False and r["phases"][0]["phase"] == "main"


def test_no_optimization_dirs_is_empty(project: Path) -> None:
    s = collect_state(project, _cfg())
    assert s["optimizations"] == []
    assert s["context"] == {"sources": {}, "total": 0}


def test_optimization_best_prompt_and_examples(project: Path) -> None:
    run = project / "oa_x"
    _write_oa_phase(run, "oa-gepa", [0.6])
    (run / "prompts.json").write_text(json.dumps({"system_prompt": "WRITE LIKE ME"}))
    (run / "examples.json").write_text(json.dumps(
        [{"brief": "b", "real": "r", "seed_gen": "s", "seed_score": 0.4,
          "omni_gen": "o", "omni_score": 0.8}]))
    (r,) = collect_state(project, _cfg())["optimizations"]
    assert r["best_prompt"] == "WRITE LIKE ME"
    assert r["examples"][0]["omni_score"] == 0.8


def test_collect_context_items(project: Path) -> None:
    d = project / ".evsys" / "context" / "directory"
    d.mkdir(parents=True)
    rows = [{"item_id": f"i{n}", "source": "directory", "entity": "shrey",
             "content": f"Subject: hello {n}\n\nbody", "metadata": {"path": f"/x/{n}.txt"}}
            for n in range(3)]
    (d / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + "{torn")
    ctx = collect_state(project, _cfg())["context"]
    assert ctx["total"] == 3
    items = ctx["sources"]["directory"]
    assert items[0]["item_id"] == "i2"  # newest first
    assert items[0]["entity"] == "shrey" and "hello 2" in items[0]["content"]


def test_prompt_rewritten_flag(project: Path) -> None:
    esc = project / ".evsys" / "triggers" / "escalations" / "escalation-00000015.json"
    prompt = project / "prompt.txt"
    # prompt older than the escalation → not rewritten
    os.utime(prompt, (esc.stat().st_mtime - 100, esc.stat().st_mtime - 100))
    assert collect_state(project, _cfg())["prompt"]["rewritten"] is False
    # prompt newer than the escalation → the agent's autoresearch rewrite
    os.utime(prompt, (esc.stat().st_mtime + 100, esc.stat().st_mtime + 100))
    s = collect_state(project, _cfg())
    assert s["prompt"]["rewritten"] is True
    assert s["prompt"]["text"] == "You are a helpful assistant."


def test_daemon_liveness_window(project: Path) -> None:
    log = project / ".evsys" / "triggers" / "log.jsonl"
    now = log.stat().st_mtime
    # pull_every=4s → horizon max(12s, 10s)=12s
    assert collect_state(project, _cfg(), now=now + 1)["project"]["daemon_live"] is True
    assert collect_state(project, _cfg(), now=now + 60)["project"]["daemon_live"] is False


def test_empty_project(tmp_path: Path) -> None:
    """A project the daemon has never run in — everything renders as empty."""
    s = collect_state(tmp_path, _cfg())
    assert s["traces"] == {"total": 0, "items": [], "sources": []}
    assert s["trigger"]["log"] == []
    assert s["escalations"] == []
    assert s["prompt"]["text"] is None
    assert s["project"]["daemon_live"] is False
    # no policy.json yet → the YAML seed is shown
    assert s["gate"]["policy"]["kind"] == "failing_answers"
    assert s["gate"]["policy"]["params"] == {"threshold": 0.4}


def test_no_trigger_config(tmp_path: Path) -> None:
    cfg = SystemConfig(traces=SYSTEM_YAML["traces"])
    s = collect_state(tmp_path, cfg)
    assert s["gate"]["policy"] is None
    assert s["escalations"] == []


def test_helpers(tmp_path: Path) -> None:
    assert _parse_duration_s("4s") == 4.0
    assert _parse_duration_s("5m") == 300.0
    assert _parse_duration_s("1h") == 3600.0
    assert _parse_duration_s("garbage", default=60.0) == 60.0
    rows, total = _tail_jsonl(tmp_path / "missing.jsonl", 10)
    assert (rows, total) == ([], 0)
    p = tmp_path / "x.jsonl"
    p.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(20)))
    rows, total = _tail_jsonl(p, 5)
    assert total == 20 and [r["i"] for r in rows] == [15, 16, 17, 18, 19]


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------

@pytest.fixture()
def server(project: Path):
    handler = _make_handler(project, _cfg(), None)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    import threading

    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_http_index(server: str) -> None:
    with urllib.request.urlopen(f"{server}/") as r:
        body = r.read().decode()
    assert r.status == 200
    # the shell + the views it can render; the data arrives over /api/*
    assert "<title>evsys/obs</title>" in body
    for view in ("System", "Experiments", "Traces"):
        assert view in body


def test_http_api_state(server: str) -> None:
    with urllib.request.urlopen(f"{server}/api/state") as r:
        state = json.loads(r.read())
    assert r.status == 200
    assert state["traces"]["total"] == 3
    assert state["escalations"][0]["verdict"]["worth_autoresearch"] is True


def test_http_index_with_query(server: str) -> None:
    """`/?theme=dark` (the demo deep-link) must still serve the page."""
    with urllib.request.urlopen(f"{server}/?theme=dark") as r:
        assert r.status == 200


def test_http_404(server: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(f"{server}/nope")
    assert ei.value.code == 404


def test_serve_boots_and_stops(project: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """serve() loads the yaml, binds localhost, opens the browser, and exits
    cleanly on Ctrl-C (simulated by serve_forever raising KeyboardInterrupt)."""
    import evsys_sdk.ui.server as srv_mod

    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    monkeypatch.setattr(srv_mod.ThreadingHTTPServer, "serve_forever",
                        lambda self, poll_interval=0.5: (_ for _ in ()).throw(KeyboardInterrupt))
    srv_mod.serve(project / "system.yaml", port=0, open_browser=True,
                  prompt_file=project / "prompt.txt")
    out = capsys.readouterr().out
    assert "serving http://127.0.0.1:" in out
    assert opened and opened[0].startswith("http://127.0.0.1:")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def test_cli_ui_subcommand_parses() -> None:
    import argparse

    from evsys_sdk.cli import _cmd_ui, main  # noqa: F401 — import proves wiring exists

    # Re-build the parser via main()'s argparse by parsing --help would exit; instead
    # verify the dispatch target directly: a namespace routed to _cmd_ui calls serve().
    called = {}

    import evsys_sdk.cli as cli_mod

    def fake_serve(config, port, open_browser, prompt_file):
        called.update(config=config, port=port, open_browser=open_browser, prompt_file=prompt_file)

    import evsys_sdk.ui as ui_mod

    orig = ui_mod.serve
    ui_mod.serve = fake_serve
    try:
        ns = argparse.Namespace(config="system.yaml", port=0, no_open=True, prompt_file=None)
        assert cli_mod._cmd_ui(ns) == 0
    finally:
        ui_mod.serve = orig
    assert called == {"config": "system.yaml", "port": 0, "open_browser": False, "prompt_file": None}

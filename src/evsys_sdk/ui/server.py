"""Local observability UI over the continual-learning loop (Layers 1-3).

``evsys ui system.yaml`` serves a single-page dashboard on localhost and opens
it in the browser. It is a pure *reader* over the same on-disk state the daemon
(``evsys traces pull --watch``) writes:

  * ``.evsys/traces/<source>/traces.jsonl``  — Layer-1 ingested traces
  * ``.evsys/triggers/{policy,state}.json``  — the live gate policy + rolling state
  * ``.evsys/triggers/log.jsonl``            — every gate evaluation / spawn event
  * ``.evsys/triggers/escalations/*.json``   — trigger-fired events
  * ``.evsys/triggers/verdicts/*.json``      — what the trigger agent decided
  * ``.evsys/triggers/agent-runs/*.log``     — the headless claude transcript
  * the project's live prompt file (e.g. ``prompt.txt``), if present

Nothing is mutated and nothing leaves localhost — stdlib ``http.server`` only,
no new dependencies. The frontend (``static/index.html``) polls ``/api/state``.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

from ..config import SystemConfig

__all__ = ["collect_state", "serve"]

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*$")
_UNIT_S = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0}

#: Traces shipped per source per poll — the feed shows the recent tail, not history.
TRACE_TAIL = 100
#: Gate-log lines shipped per poll.
LOG_TAIL = 500


def _parse_duration_s(text: str, default: float = 60.0) -> float:
    """'4s' / '5m' / '1h' → seconds (mirrors the daemon's pull_every strings)."""
    m = _DURATION.match(text or "")
    return float(m.group(1)) * _UNIT_S[m.group(2)] if m else default


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def _tail_jsonl(path: Path, limit: int) -> tuple[list[dict], int]:
    """Last ``limit`` parsed rows of a JSONL file + the total row count."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return [], 0
    rows = []
    for line in lines[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # torn tail write mid-append
            continue
    return rows, len(lines)


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _collect_traces(project_dir: Path, cfg: SystemConfig) -> dict:
    roots = {spec.state_dir for spec in cfg.traces.trace_sources} or {".evsys/traces"}
    sources: list[str] = []
    items: list[dict] = []
    total = 0
    freshest: float | None = None
    for root in sorted(roots):
        base = project_dir / root
        if not base.is_dir():
            continue
        for src_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            jl = src_dir / "traces.jsonl"
            rows, n = _tail_jsonl(jl, TRACE_TAIL)
            if n:
                sources.append(src_dir.name)
                total += n
                items.extend(rows)
                mt = _mtime(jl)
                if mt is not None:
                    freshest = mt if freshest is None else max(freshest, mt)
    # Oldest-first within the shipped tail (the UI reverses for newest-first).
    items.sort(key=lambda t: str((t.get("metadata") or {}).get("timestamp") or ""))
    return {"total": total, "items": items[-TRACE_TAIL:], "sources": sources, "freshest_mtime": freshest}


def _collect_escalations(root: Path) -> list[dict]:
    esc_dir = root / "escalations"
    if not esc_dir.is_dir():
        return []
    out = []
    for path in sorted(esc_dir.glob("*.json")):
        stem = path.stem
        out.append(
            {
                "id": stem,
                "event": _read_json(path),
                "verdict": _read_json(root / "verdicts" / f"{stem}.json"),
                "agent_log": _read_text(root / "agent-runs" / f"{stem}.log"),
                "prompt_before": _read_text(root / "prompt-snapshots" / f"{stem}.txt"),
                "mtime": _mtime(path),
            }
        )
    return out


#: Eval points shipped per optimization phase per poll.
OPT_TAIL = 400


def _collect_optimizations(project_dir: Path) -> list[dict]:
    """optimize_anything run dirs → per-phase best-so-far traces for the panel.

    Layout (written by the ``optimize_anything`` algorithm): ``<run>/oa[-explore]-
    <engine>/evals/<n>.json`` + ``summary.json``, ``<run>/oa_summary.json`` after
    completion. Pure reads; a mid-write eval file is skipped, not fatal.
    """
    runs: list[dict] = []
    for run_dir in sorted(project_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        phase_dirs = sorted(d for d in run_dir.glob("oa-*") if (d / "evals").is_dir())
        if not phase_dirs:
            continue
        phases = []
        for pd in phase_dirs:
            explore = pd.name.startswith("oa-explore-")
            engine = pd.name.removeprefix("oa-explore-" if explore else "oa-")
            points, best = [], 0.0
            evals = sorted(
                (f for f in pd.glob("evals/*.json") if f.stem.isdigit()),
                key=lambda p: int(p.stem),
            )[-OPT_TAIL:]
            for f in evals:
                d = _read_json(f)
                if not isinstance(d, dict) or "score" not in d:
                    continue
                best = max(best, float(d["score"]))
                points.append({"eval": d.get("eval"), "score": d["score"], "best": round(best, 4)})
            phases.append({
                "engine": engine,
                "phase": "explore" if explore else "main",
                "points": points,
                "summary": _read_json(pd / "summary.json"),
            })
        summary = _read_json(run_dir / "oa_summary.json")
        runs.append({
            "run": run_dir.name,
            "phases": phases,
            "summary": summary,
            "done": summary is not None,
            "mtime": _mtime(run_dir),
        })
    return runs


def _prompt_diff(escalations: list[dict], prompt_text: str | None) -> tuple[str | None, str | None]:
    """Unified diff of the live prompt against the newest escalation-time snapshot
    (what the trigger agent started from). ``(None, None)`` when there is no
    snapshot or nothing changed; ``(diff, escalation_id)`` otherwise."""
    latest = next((e for e in reversed(escalations) if e.get("prompt_before") is not None), None)
    if latest is None or prompt_text is None:
        return None, None
    before = latest["prompt_before"]
    if before == prompt_text:
        return None, None
    diff = "\n".join(
        difflib.unified_diff(
            before.splitlines(), prompt_text.splitlines(),
            fromfile=f"prompt.txt @ {latest['id']}", tofile="prompt.txt (live)", lineterm="",
        )
    )
    return diff, latest["id"]


def _collect_gate(project_dir: Path, cfg: SystemConfig, root: Path) -> dict:
    policy = _read_json(root / "policy.json")
    if policy is None and cfg.trigger is not None:
        # Daemon hasn't seeded policy.json yet — show the YAML seed it will write.
        policy = {
            "kind": cfg.trigger.kind,
            "import_path": cfg.trigger.import_path,
            "params": cfg.trigger.params,
            "every_n": cfg.trigger.every_n,
            "window": cfg.trigger.window,
        }
    source = source_path = None
    import_path = (policy or {}).get("import_path")
    if import_path and str(import_path).endswith(".py"):
        candidate = Path(import_path)
        if not candidate.is_absolute():
            candidate = project_dir / candidate
        source = _read_text(candidate)
        source_path = str(import_path)
    return {"policy": policy, "source": source, "source_path": source_path}


def collect_state(
    project_dir: Path,
    cfg: SystemConfig,
    prompt_file: Path | None = None,
    now: float | None = None,
) -> dict:
    """One poll of the whole loop — everything the dashboard renders, read fresh
    from disk. Pure reads; safe to call while the daemon is writing."""
    now = time.time() if now is None else now
    project_dir = project_dir.resolve()
    trigger_root = project_dir / (cfg.trigger.state_dir if cfg.trigger else ".evsys/triggers")

    traces = _collect_traces(project_dir, cfg)
    log, _ = _tail_jsonl(trigger_root / "log.jsonl", LOG_TAIL)
    raw_state = _read_json(trigger_root / "state.json") or {}
    escalations = _collect_escalations(trigger_root)

    # Live-ness: the daemon is "live" if it wrote traces or gate-log recently
    # (within a few poll intervals of the fastest-polling source).
    pulls = [_parse_duration_s(s.pull_every) for s in cfg.traces.trace_sources] or [60.0]
    horizon = max(3 * min(pulls), 10.0)
    stamps = [m for m in (traces.pop("freshest_mtime"), _mtime(trigger_root / "log.jsonl")) if m]
    daemon_live = bool(stamps) and now - max(stamps) < horizon

    if prompt_file is None:
        default = project_dir / "prompt.txt"
        prompt_file = default if default.exists() else None
    prompt_text = _read_text(prompt_file) if prompt_file else None
    prompt_mtime = _mtime(prompt_file) if prompt_file else None
    esc_mtimes = [e["mtime"] for e in escalations if e["mtime"] is not None]
    rewritten = bool(prompt_mtime and esc_mtimes and prompt_mtime > min(esc_mtimes))
    prompt_diff, diff_base = _prompt_diff(escalations, prompt_text)

    return {
        "project": {"name": project_dir.name, "dir": str(project_dir), "daemon_live": daemon_live, "now": now},
        "traces": traces,
        "trigger": {
            "log": log,
            "state": {
                "counters": raw_state.get("counters", {}),
                "window_size": len(raw_state.get("window", [])),
            },
        },
        "gate": _collect_gate(project_dir, cfg, trigger_root),
        "escalations": escalations,
        "prompt": {
            "text": prompt_text,
            "path": str(prompt_file) if prompt_file else None,
            "mtime": prompt_mtime,
            "rewritten": rewritten,
            "diff": prompt_diff,
            "diff_base": diff_base,
        },
        "optimizations": _collect_optimizations(project_dir),
    }


def _index_html() -> bytes:
    return (resources.files("evsys_sdk.ui") / "static" / "index.html").read_bytes()


def _make_handler(project_dir: Path, cfg: SystemConfig, prompt_file: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = _index_html()
                self._reply(200, "text/html; charset=utf-8", body)
            elif path == "/api/state":
                state = collect_state(project_dir, cfg, prompt_file=prompt_file)
                self._reply(200, "application/json", json.dumps(state).encode())
            else:
                self._reply(404, "text/plain", b"not found")

        def _reply(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass  # silence per-request stderr noise

    return Handler


def serve(
    config_path: str | Path,
    port: int = 7749,
    open_browser: bool = True,
    prompt_file: str | Path | None = None,
) -> None:
    """Serve the observability UI for one system.yaml until interrupted."""
    import yaml

    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = SystemConfig(**raw)
    project_dir = config_path.parent
    pf = Path(prompt_file).resolve() if prompt_file else None

    server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(project_dir, cfg, pf))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"evsys ui: watching {project_dir}")
    print(f"evsys ui: serving {url}  (Ctrl-C to stop)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

"""Remote (sandboxed) execution for the trigger + autoresearch agents.

Instead of shelling out to ``claude -p`` on the host, each agent runs in its
own fresh sandbox — the trigger agent first; on a YES verdict, a SECOND fresh
sandbox runs the autoresearch stage. Both get the same staged skills.

*Which* sandbox is a config choice, not a fact about this module: the provider
is resolved by name through the sandbox registry
(:mod:`evsys_sdk.sandboxes`), so ``sandbox: {kind: e2b}``,
``{kind: local}``, or a provider the project registered itself all run the
identical two-stage flow. Everything below talks to
:class:`~evsys_sdk.sandboxes.base.BaseSandbox`.

**Why copy-in / copy-out** (the traces-access decision): the agent contract is
already file-based — an escalation JSON, a window of traces, the live policy,
the gate fn source, the prompt file, skills. All small. So the sandbox is
staged with a snapshot of exactly that corpus under its workdir, the agent
runs against local-looking relative paths, and afterwards only the known
artifact set is copied back to the host:

  * ``verdict.json``      → the driver's verdicts dir
  * ``policy.json``       → the gate retune (daemon hot-reloads it)
  * the gate ``.py``      → a rewritten trigger fn (hot-reloaded too)
  * the declared artifacts → the autoresearch rewrites

No tunnels, no host filesystem mounts, no inbound network: the sandbox holds
nothing but the staged snapshot and the model key. A malicious or confused
agent can at worst corrupt the files we explicitly copy back.

Vendor SDKs are imported lazily by their provider (``e2b`` ships in the
``remote`` extra). Tests either register a fake provider or replace the
:data:`_SANDBOX_FACTORY` seam; a real gated smoke needs ``E2B_API_KEY``.
"""

from __future__ import annotations

import json
import shlex
import threading
from pathlib import Path
from typing import Any

from ..logger import get_logger
from ..provenance import AGENT_AUTORESEARCH, AGENT_TRIGGER, trigger_env
from ..sandboxes import DEFAULT_WORKDIR, build_sandbox, resolve_envs
from .agent import build_command

log = get_logger(__name__)

WORKDIR = DEFAULT_WORKDIR
"""Default sandbox-side workdir. A provider may use its own (``LocalSandbox``
stages into a scratch dir) — read ``sandbox.workdir`` rather than this when the
path has to be real."""

REMOTE_AUTORESEARCH_PROMPT = """You are the evsys **autoresearch agent**, running remotely. The \
trigger agent already judged this escalation worth fixing — your job is to actually improve the \
system's artifacts.

- Escalation event:  {escalation_path}
- Trigger-agent verdict (read its hypothesis): {verdict_path}
- Ingested traces (JSONL per source under this dir): {traces_dir}
- The artifacts you may improve (project-relative; ONLY these leave this sandbox): {artifacts}
- Skills: ./skills/  — the project's own improvement playbooks; follow them.

Do this:
1. Read the verdict's hypothesis and the implicated traces.
2. Design the smallest change to the listed artifacts that addresses the failure mode. You MAY run
   experiments with the evsys SDK (installed here): evaluations, and training/weight updates through
   hosted backends (e.g. tinker) — compute happens on the backend, not in this sandbox.
3. Rewrite the artifact(s). Anything you write outside the listed artifacts is discarded.

Be decisive; validate before you overwrite."""


def _make_sandbox(remote_cfg: Any, envs: dict[str, str]) -> Any:
    """Resolve ``remote.sandbox: {kind, params}`` into a started sandbox."""
    return build_sandbox(
        getattr(remote_cfg, "sandbox", None) or "e2b",
        envs=envs, timeout_s=remote_cfg.timeout_s,
    )


_SANDBOX_FACTORY = _make_sandbox  # seam

_PENDING: list[threading.Thread] = []
"""In-flight detached remote agents, so a caller about to exit can drain them."""


# ---------------------------------------------------------------------------
# Staging (pure — unit-testable without a sandbox)
# ---------------------------------------------------------------------------


def _tail(text: str, lines: int) -> str:
    rows = text.splitlines()
    return "\n".join(rows[-lines:]) + ("\n" if rows else "")


def build_manifest(
    *,
    escalation_path: Path,
    root: Path,
    cwd: Path,
    artifacts: list[str],
    include_traces: str = "window",
    trace_tail_lines: int = 500,
) -> dict[str, str]:
    """The copy-in set: ``{sandbox-relative path: content}``.

    Mirrors the local layout under the sandbox workdir so the agent's mission
    prompt can reference the same relative paths it would see on the host.
    ``artifacts`` are the project-declared improvable files (globs allowed).
    """
    manifest: dict[str, str] = {}

    def _add(rel: str, path: Path) -> None:
        try:
            manifest[rel] = path.read_text()
        except OSError:
            pass

    _add(f".evsys/triggers/escalations/{escalation_path.name}", escalation_path)
    _add(".evsys/triggers/policy.json", root / "policy.json")
    for pattern in artifacts:
        for match in sorted(cwd.glob(pattern)):
            if match.is_file():
                _add(str(match.relative_to(cwd)), match)

    # the gate fn source, when the live policy points at a .py file
    try:
        policy = json.loads((root / "policy.json").read_text())
        import_path = str(policy.get("import_path") or "")
        if import_path.endswith(".py"):
            _add(import_path, (cwd / import_path))
    except (OSError, json.JSONDecodeError):
        pass

    traces_dir = root.parent / "traces"
    if traces_dir.is_dir():
        for tj in sorted(traces_dir.glob("*/traces.jsonl")):
            text = tj.read_text() if include_traces == "all" else _tail(
                tj.read_text(), trace_tail_lines)
            manifest[f".evsys/traces/{tj.parent.name}/traces.jsonl"] = text

    skills_dir = cwd / "skills"
    if skills_dir.is_dir():
        for f in sorted(skills_dir.rglob("*")):
            if f.is_file() and f.stat().st_size < 256_000:
                _add(str(f.relative_to(cwd)), f)

    return manifest


# ---------------------------------------------------------------------------
# The two-stage remote run
# ---------------------------------------------------------------------------


def _stage_and_run(sbx: Any, *, manifest: dict[str, str], prompt_argv: list[str],
                   remote_cfg: Any, log_file: Path, stage: str) -> tuple[int, str]:
    """Stage the snapshot, provision, then run one agent — provider-agnostic:
    ``stage`` / ``setup`` come from :class:`BaseSandbox`, and only ``exec`` is
    the provider's own."""
    sbx.stage(manifest)
    try:
        sbx.setup(remote_cfg.setup_cmd, required=True, label="setup_cmd")
    except Exception as e:
        raise RuntimeError(f"[{stage}] {e}") from e
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Observability: the sandbox agent's stdout streams into the SAME
    # agent-runs log the local path uses, line by line — so `evsys ui`'s
    # transcript panel (and `tail -f`) follows the sandboxed claude live.
    with log_file.open("a") as f:
        f.write(f"\n===== remote {stage}: started =====\n")
        f.flush()

        def _on_line(line: str) -> None:
            f.write(line if line.endswith("\n") else line + "\n")
            f.flush()

        # best-effort: a failed SDK install is logged, it does not kill the run
        sbx.setup(getattr(remote_cfg, "sdk_install", None), required=False,
                  on_line=_on_line, label="sdk_install")
        # `< /dev/null`: a headless claude waits ~3s for stdin it will never
        # get in a sandbox, then warns. Close it explicitly.
        try:
            code, out = sbx.exec(shlex.join(prompt_argv) + " < /dev/null",
                                 timeout_s=remote_cfg.timeout_s,
                                 cwd=sbx.workdir, on_line=_on_line)
        except Exception as e:
            # The sandbox died under us (expired, evicted, network). Say so IN
            # the transcript — otherwise the log simply stops after "started"
            # and there is nothing anywhere that says why.
            f.write(f"===== remote {stage}: FAILED: {type(e).__name__}: {e} =====\n")
            raise
        f.write(f"===== remote {stage}: exit {code} =====\n")
    return code, out


MIRROR_DIR = "evsys_sdk"
"""Where the SDK's local mirror lands inside the sandbox — ``EVSYS_LOG_DIR``
is set to this, relative to the workdir, so it comes back at a known root."""

RESULT_DIRS = (MIRROR_DIR, "outputs")
"""Directories synced back wholesale when the sandbox dies.

``evsys_sdk`` is the machine-readable record (experiments, runs, metrics,
evals, rollouts) the UI reads. ``outputs`` is the human-readable side — the
run's logs, the training rollouts, the agent's own experiment log. Without the
second one you can see that a run happened and not what it printed."""


def _pin_mirror(sbx: Any) -> None:
    """Point the sandbox-side mirror at ``<workdir>/evsys_sdk``, so the results
    sync knows where to look.

    Set from the LIVE sandbox's ``workdir``, not the module constant: a
    provider is free to stage elsewhere (``LocalSandbox`` uses a scratch dir;
    ``ModalSandbox`` with ``user: agent`` stages under that user's home). Using
    the constant pointed the mirror at a directory the agent never wrote to on
    any such provider, and every experiment it ran died with the box.

    Safe before ``start()``: adapters read ``envs`` when they boot, not when
    they are constructed.
    """
    sbx.envs["EVSYS_LOG_DIR"] = sbx.path(MIRROR_DIR)


def _collect_experiments(sbx: Any, cwd: Path, baseline: dict[str, str]) -> list[str]:
    """Bring the agent's EXPERIMENTS home before the sandbox dies.

    The fixed artifact list covers the four files the agent may rewrite. It
    cannot cover what its experiments wrote — experiment records, runs, step
    metrics, evals, captured rollouts — because those have generated ids
    nobody can enumerate in advance. Without this the sandbox is a black hole:
    the agent trains a model, evaluates it, and every trace of that work dies
    with the box, leaving only a prompt diff and no way to see what it tried.
    """
    landed: list[str] = []
    for d in RESULT_DIRS:
        try:
            landed += sbx.collect_tree(d, cwd, baseline=baseline)
        except Exception as e:  # a results-sync failure must not fail the run
            log.warning("[remote] could not collect %s: %s", d, e)
    if landed:
        log.info("[remote] brought back %d experiment file(s) from the sandbox", len(landed))
    return landed


def _collect_new_files(sbx: Any, cwd: Path, baseline: dict[str, str]) -> list[str]:
    """Bring home the files the agent CREATED at the workdir root.

    ``collect`` only round-trips paths that were staged IN, so a script the
    agent wrote, the dataset it curated and its run log — the actual evidence
    of what it did — were invisible. Top level only, and skipping the dirs the
    tree sync already covers.
    """
    keep: list[tuple[str, Path]] = []
    try:
        for rel in sbx.list_tree("", max_files=400):
            if "/" in rel or rel in baseline:
                continue                     # nested (covered elsewhere) or staged
            keep.append((rel, cwd / rel))
    except Exception as e:
        log.warning("[remote] could not list new files: %s", e)
        return []
    landed = sbx.collect(keep, baseline)
    if landed:
        log.info("[remote] brought back %d file(s) the agent wrote: %s",
                 len(landed), ", ".join(landed[:6]))
    return landed


def run_remote(escalation_path: Path, *, agent_cfg: Any, root: Path, cwd: Path,
               verdict_path: Path, log_file: Path) -> dict[str, Any]:
    """Run the trigger agent (and, on YES, the autoresearch agent) in whichever
    sandbox provider ``remote.sandbox.kind`` names."""
    remote_cfg = agent_cfg.remote
    envs = resolve_envs(remote_cfg.env_passthrough)
    prompt_file = getattr(agent_cfg, "prompt_file", None) or "prompt.txt"
    # the general improve-contract; the prompt file is only the DEFAULT artifact
    artifacts = list(remote_cfg.artifacts) or [prompt_file]
    manifest = build_manifest(
        escalation_path=escalation_path, root=root, cwd=cwd, artifacts=artifacts,
        include_traces=remote_cfg.include_traces,
        trace_tail_lines=remote_cfg.trace_tail_lines,
    )

    # sandbox-relative paths for the mission prompt (same shape as local)
    sbx_esc = f".evsys/triggers/escalations/{escalation_path.name}"

    gate_rel = None
    try:
        policy = json.loads((root / "policy.json").read_text())
        if str(policy.get("import_path") or "").endswith(".py"):
            gate_rel = str(policy["import_path"])
    except (OSError, json.JSONDecodeError):
        pass

    # Provenance travels INTO the sandbox: experiments the agent launches in
    # there get stamped with the escalation exactly as a host-side agent's do.
    sandbox_kind = getattr(getattr(remote_cfg, "sandbox", None), "kind", None) or "e2b"
    stage1_envs = {**envs,
                   **trigger_env(escalation_path, agent=AGENT_TRIGGER,
                                 agent_run=escalation_path.stem,
                                 sandbox=sandbox_kind)}

    result: dict[str, Any] = {"stage1_exit": None, "stage2_exit": None,
                              "artifacts": [], "experiments": [], "new_files": []}
    sbx = _SANDBOX_FACTORY(remote_cfg, stage1_envs)
    _pin_mirror(sbx)
    try:
        # absolute paths come from the live sandbox, not a constant: a provider
        # is free to stage somewhere else (LocalSandbox uses a scratch dir).
        sbx_root = sbx.path(".evsys/triggers")
        argv = build_command(
            sbx.path(sbx_esc), agent_cfg=agent_cfg, root=Path(sbx_root),
            verdict_path=f"{sbx_root}/verdicts/{escalation_path.stem}.json",
        )
        code, _ = _stage_and_run(sbx, manifest=manifest, prompt_argv=argv,
                                 remote_cfg=remote_cfg, log_file=log_file, stage="trigger-agent")
        result["stage1_exit"] = code
        pair_map: dict[str, Path] = {
            f".evsys/triggers/verdicts/{escalation_path.stem}.json": verdict_path,
            ".evsys/triggers/policy.json": root / "policy.json",
        }
        for rel in manifest:
            if not rel.startswith(".evsys/") and not rel.startswith("skills/"):
                pair_map[rel] = cwd / rel
        if gate_rel:
            pair_map[gate_rel] = cwd / gate_rel
        result["artifacts"] = sbx.collect(list(pair_map.items()), manifest)
        result["experiments"] = _collect_experiments(sbx, cwd, manifest)
        result["new_files"] = _collect_new_files(sbx, cwd, manifest)
    finally:
        sbx.kill()

    # stage 2: fresh sandbox for autoresearch on a YES verdict
    verdict = {}
    try:
        verdict = json.loads(verdict_path.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    if (verdict.get("worth_autoresearch") and remote_cfg.autoresearch_sandbox
            and getattr(agent_cfg, "autoresearch", True)):
        template = remote_cfg.autoresearch_prompt_template or REMOTE_AUTORESEARCH_PROMPT
        prompt = template.format(
            escalation_path=sbx_esc,
            verdict_path=f".evsys/triggers/verdicts/{escalation_path.stem}.json",
            traces_dir=".evsys/traces",
            artifacts=", ".join(artifacts),
        )
        argv2 = [getattr(agent_cfg, "claude_bin", "claude"), "-p", prompt,
                 "--permission-mode", getattr(agent_cfg, "permission_mode", "acceptEdits")]
        if getattr(agent_cfg, "model", None):
            argv2 += ["--model", agent_cfg.model]
        manifest2 = dict(manifest)
        manifest2[f".evsys/triggers/verdicts/{escalation_path.stem}.json"] = json.dumps(verdict)
        stage2_envs = {**envs,
                       **trigger_env(escalation_path, agent=AGENT_AUTORESEARCH,
                                             agent_run=escalation_path.stem,
                                             sandbox=sandbox_kind)}
        sbx2 = _SANDBOX_FACTORY(remote_cfg, stage2_envs)
        _pin_mirror(sbx2)
        try:
            code2, _ = _stage_and_run(sbx2, manifest=manifest2, prompt_argv=argv2,
                                      remote_cfg=remote_cfg, log_file=log_file,
                                      stage="autoresearch")
            result["stage2_exit"] = code2
            art_pairs = [(rel, cwd / rel) for rel in manifest2
                         if not rel.startswith(".evsys/") and not rel.startswith("skills/")]
            result["artifacts"] += sbx2.collect(art_pairs, manifest2)
            result["experiments"] += _collect_experiments(sbx2, cwd, manifest2)
            result["new_files"] += _collect_new_files(sbx2, cwd, manifest2)
        finally:
            sbx2.kill()

    log.info("[remote] agents done (stage1=%s stage2=%s artifacts=%s experiment files=%s)",
             result["stage1_exit"], result["stage2_exit"], result["artifacts"],
             len(result["experiments"]))
    return result


def spawn_remote(escalation_path: Path, *, agent_cfg: Any, root: Path, cwd: Path,
                 verdict_path: Path, log_file: Path, detach: bool = True) -> Any:
    """Detached → daemon thread (mirrors the local fire-and-forget Popen);
    foreground → run inline and return the result dict."""
    if not detach:
        return run_remote(escalation_path, agent_cfg=agent_cfg, root=root, cwd=cwd,
                          verdict_path=verdict_path, log_file=log_file)

    def _target() -> None:
        try:
            run_remote(escalation_path, agent_cfg=agent_cfg, root=root, cwd=cwd,
                       verdict_path=verdict_path, log_file=log_file)
        except Exception as e:  # a failed remote run must not kill the daemon
            log.warning("[remote] agent run failed: %s", e)

    t = threading.Thread(target=_target, name="evsys-remote-agent", daemon=True)
    t.start()
    _PENDING.append(t)
    return t


def join_pending(timeout_s: float | None = None) -> int:
    """Wait for in-flight remote agents, returning how many were still running.

    A detached LOCAL agent is a `Popen` in its own session, so it outlives the
    daemon that spawned it. A detached REMOTE agent is a daemon thread, which
    does NOT — a one-shot `evsys traces pull` would exit and kill the agent
    before it had done anything at all. Callers that are about to exit must
    drain them.
    """
    alive = [t for t in _PENDING if t.is_alive()]
    for t in alive:
        t.join(timeout_s)
    return len([t for t in alive if t.is_alive()])


__all__ = ["build_manifest", "join_pending", "run_remote", "spawn_remote", "WORKDIR",
           "REMOTE_AUTORESEARCH_PROMPT"]

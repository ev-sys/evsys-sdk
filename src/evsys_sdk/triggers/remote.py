"""Remote (E2B) execution for the trigger + autoresearch agents.

Instead of shelling out to ``claude -p`` on the host, each agent runs in its
own E2B sandbox — the trigger agent first; on a YES verdict, a SECOND fresh
sandbox runs the autoresearch stage. Both get the same staged skills.

**Why copy-in / copy-out** (the traces-access decision): the agent contract is
already file-based — an escalation JSON, a window of traces, the live policy,
the gate fn source, the prompt file, skills. All small. So the sandbox is
staged with a snapshot of exactly that corpus under :data:`WORKDIR`, the agent
runs against local-looking relative paths, and afterwards only the known
artifact set is copied back to the host:

  * ``verdict.json``      → the driver's verdicts dir
  * ``policy.json``       → the gate retune (daemon hot-reloads it)
  * the gate ``.py``      → a rewritten trigger fn (hot-reloaded too)
  * the prompt file       → the autoresearch rewrite

No tunnels, no host filesystem mounts, no inbound network: the sandbox holds
nothing but the staged snapshot and the model key. A malicious or confused
agent can at worst corrupt the four files we explicitly copy back.

The E2B SDK is imported lazily (``remote`` extra); tests replace
:data:`_SANDBOX_FACTORY` with a fake, and a real gated smoke needs
``E2B_API_KEY``.
"""

from __future__ import annotations

import json
import shlex
import threading
from pathlib import Path
from typing import Any

from ..logger import get_logger
from .agent import build_command

log = get_logger(__name__)

WORKDIR = "/home/user/evsys"

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


class _E2BSandbox:
    """Thin adapter over the E2B SDK (the seam tests fake)."""

    def __init__(self, template: str | None, envs: dict[str, str], timeout_s: float) -> None:
        from e2b import Sandbox  # lazy: the `remote` extra

        kwargs: dict[str, Any] = {"envs": envs, "timeout": int(timeout_s) + 120}
        self._sbx = Sandbox(template, **kwargs) if template else Sandbox(**kwargs)
        self._envs = envs

    def write(self, path: str, content: str) -> None:
        self._sbx.files.write(path, content)

    def read(self, path: str) -> str | None:
        try:
            return self._sbx.files.read(path)
        except Exception:
            return None

    def run(self, cmd: str, *, timeout_s: float, cwd: str | None = None,
            on_line: Any = None) -> tuple[int, str]:
        cb = (lambda data: on_line(str(data))) if on_line else None
        result = self._sbx.commands.run(
            cmd, envs=self._envs, timeout=int(timeout_s), cwd=cwd,
            on_stdout=cb, on_stderr=cb,
        )
        out = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
        return int(result.exit_code or 0), out

    def kill(self) -> None:
        try:
            self._sbx.kill()
        except Exception:  # pragma: no cover - best-effort teardown
            pass


def _make_sandbox(remote_cfg: Any, envs: dict[str, str]) -> Any:
    return _E2BSandbox(remote_cfg.template, envs, remote_cfg.timeout_s)


_SANDBOX_FACTORY = _make_sandbox  # seam


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
    for rel, content in manifest.items():
        sbx.write(f"{WORKDIR}/{rel}", content)
    if remote_cfg.setup_cmd:
        code, out = sbx.run(remote_cfg.setup_cmd, timeout_s=600, cwd=WORKDIR)
        if code != 0:
            raise RuntimeError(f"[{stage}] setup_cmd failed ({code}): {out[-500:]}")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Observability: the sandbox agent's stdout streams into the SAME
    # agent-runs log the local path uses, line by line — so `evsys ui`'s
    # transcript panel (and `tail -f`) follows the remote claude live.
    with log_file.open("a") as f:
        f.write(f"\n===== remote {stage}: started =====\n")
        f.flush()

        def _on_line(line: str) -> None:
            f.write(line if line.endswith("\n") else line + "\n")
            f.flush()

        if getattr(remote_cfg, "sdk_install", None):
            code, out = sbx.run(remote_cfg.sdk_install, timeout_s=600, cwd=WORKDIR)
            if code != 0:  # best-effort: log it, don't kill the run
                _on_line(f"[sdk_install failed ({code})] {out[-300:]}")
        code, out = sbx.run(shlex.join(prompt_argv), timeout_s=remote_cfg.timeout_s,
                            cwd=WORKDIR, on_line=_on_line)
        f.write(f"===== remote {stage}: exit {code} =====\n")
    return code, out


def _copy_back(sbx: Any, pairs: list[tuple[str, Path]], baseline: dict[str, str]) -> list[str]:
    """Copy the allowed artifacts back — but only the ones the agent actually
    CHANGED vs what was staged (an untouched file must not round-trip: host
    mtimes drive UI signals like the prompt's 'rewritten' flag)."""
    landed = []
    for rel, dest in pairs:
        content = sbx.read(f"{WORKDIR}/{rel}")
        if content is None or content == baseline.get(rel):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        landed.append(rel)
    return landed


def run_remote(escalation_path: Path, *, agent_cfg: Any, root: Path, cwd: Path,
               verdict_path: Path, log_file: Path) -> dict[str, Any]:
    """Run the trigger agent (and, on YES, the autoresearch agent) in E2B."""
    remote_cfg = agent_cfg.remote
    import os

    envs = {k: os.environ[k] for k in remote_cfg.env_passthrough if os.environ.get(k)}
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
    sbx_root = f"{WORKDIR}/.evsys/triggers"
    sbx_verdict = f"{sbx_root}/verdicts/{escalation_path.stem}.json"
    argv = build_command(
        f"{WORKDIR}/{sbx_esc}", agent_cfg=agent_cfg, root=Path(sbx_root),
        verdict_path=sbx_verdict,
    )

    gate_rel = None
    try:
        policy = json.loads((root / "policy.json").read_text())
        if str(policy.get("import_path") or "").endswith(".py"):
            gate_rel = str(policy["import_path"])
    except (OSError, json.JSONDecodeError):
        pass

    result: dict[str, Any] = {"stage1_exit": None, "stage2_exit": None, "artifacts": []}
    sbx = _SANDBOX_FACTORY(remote_cfg, envs)
    try:
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
        result["artifacts"] = _copy_back(sbx, list(pair_map.items()), manifest)
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
        sbx2 = _SANDBOX_FACTORY(remote_cfg, envs)
        try:
            code2, _ = _stage_and_run(sbx2, manifest=manifest2, prompt_argv=argv2,
                                      remote_cfg=remote_cfg, log_file=log_file,
                                      stage="autoresearch")
            result["stage2_exit"] = code2
            art_pairs = [(rel, cwd / rel) for rel in manifest2
                         if not rel.startswith(".evsys/") and not rel.startswith("skills/")]
            result["artifacts"] += _copy_back(sbx2, art_pairs, manifest2)
        finally:
            sbx2.kill()

    log.info("[remote] agents done (stage1=%s stage2=%s artifacts=%s)",
             result["stage1_exit"], result["stage2_exit"], result["artifacts"])
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
    return t


__all__ = ["build_manifest", "run_remote", "spawn_remote", "WORKDIR",
           "REMOTE_AUTORESEARCH_PROMPT"]

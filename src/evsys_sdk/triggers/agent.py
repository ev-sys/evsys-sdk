"""Headless trigger-agent invocation — spawn Claude Code (``claude -p``) on an
escalation event.

When the cheap deterministic gate escalates, this is the seam that hands the
batch to the (heavier) **trigger agent**: a one-shot, headless Claude Code run
pointed at the escalation event. The agent assesses the batch, writes a verdict,
may retune ``policy.json`` (the self-improving gate), and — on a YES — launches
the autoresearch agent (``training-decider``) to actually run an experiment.

The driver spawns this **detached** (fire-and-forget) so a per-trace ingestion
hook is never blocked on a full agent run; the demo / CLI can spawn it in the
foreground (``detach=False``) to watch the run. The actual process launch goes
through :data:`_LAUNCH`, a module-level seam tests monkeypatch so the suite never
shells out to a real ``claude``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ..logger import get_logger

log = get_logger(__name__)

DEFAULT_PROMPT = """You are the evsys **trigger agent**. The cheap deterministic gate just escalated \
a batch of production agent traces and it is your job to decide whether they are worth spending \
autoresearch budget on.

Escalation event: {escalation_path}
Ingested traces:  {traces_dir}
Live gate policy: {policy_path}
Write your verdict to: {verdict_path}

Do this:
1. Read the escalation event and the implicated traces. Use the `assess-traces` skill to judge whether \
this batch reflects a real, learnable failure mode (not noise).
2. Write a verdict JSON to the path above: \
{{"escalation": "<event file>", "worth_autoresearch": <bool>, "reasoning": "<why>", "hypothesis": "<what to try>"}}.
3. Consider retuning the gate: if it fired on noise (or missed obvious failures), edit {policy_path} \
using the `tune-trigger` skill — this is the self-improving gate.
4. {autoresearch_clause}

Keep it cheap and decisive: you are the gatekeeper, not the researcher."""

DISTILL_PROMPT = """You are the evsys **distiller agent**. The cheap deterministic gate just \
escalated a batch of coding-agent traces (Claude Code sessions). Your job is to convert them into \
evaluation + training data and launch the PRESET experiment — nothing more.

Escalation event: {escalation_path}
Ingested traces:  {traces_dir}
Live gate policy: {policy_path}
Write your verdict to: {verdict_path}
Preset experiment template: {experiment_template}
Holdout fraction: {holdout_fraction}
Benchmark dir: {benchmark_dir}   Train dir: {train_dir}

Hard rules:
- Do NOT invoke the `training-decider` agent. Do NOT design, choose, or tune a training \
algorithm — the algorithm is FIXED by the experiment template.
- Never let an eval session's data into the training rows (the holdout split is the \
contamination boundary).

Do this, following the `distill-traces` skill:
1. Assess the escalated traces (`assess-traces` skill) and write the verdict JSON \
({{"escalation", "worth_autoresearch", "reasoning", "hypothesis"}}) to the path above. If the \
batch is noise, stop here (you may retune {policy_path} via `tune-trigger`).
2. Split sessions chronologically: newest {holdout_fraction} of sessions -> eval, rest -> train. \
Write the eval set as a benchmark dir under {benchmark_dir} and training rows under {train_dir}.
3. Materialize the experiment FROM THE TEMPLATE (copy {experiment_template}, fill only names/paths), \
launch it, and monitor: poll its run outputs on a sensible cadence, abort on NaN/stalled loss, and \
write a short report next to the verdict when training ends.

Be decisive and cheap."""

AUTORESEARCH_ON = (
    "If (and only if) the batch is worth it, launch the autoresearch agent: invoke the "
    "`training-decider` agent with your hypothesis + the implicated trace ids so it designs and runs "
    "the next experiment."
)
AUTORESEARCH_OFF = (
    "Do NOT launch autoresearch — stop after writing the verdict (a later step consumes it)."
)


def build_command(escalation_path: str | Path, *, agent_cfg: Any, root: str | Path,
                  verdict_path: str | Path) -> list[str]:
    """Build the ``claude -p ...`` argv for one escalation. Pure — no side effects."""
    escalation_path = Path(escalation_path)
    root = Path(root)
    autoresearch = getattr(agent_cfg, "autoresearch", True)
    mode = getattr(agent_cfg, "mode", "verdict")
    default = DISTILL_PROMPT if mode == "distill" else DEFAULT_PROMPT
    template = getattr(agent_cfg, "prompt_template", None) or default
    distill = getattr(agent_cfg, "distill", None)
    prompt = template.format(
        escalation_path=escalation_path,
        traces_dir=(root.parent / "traces"),
        policy_path=(root / "policy.json"),
        verdict_path=verdict_path,
        autoresearch_clause=(AUTORESEARCH_ON if autoresearch else AUTORESEARCH_OFF),
        experiment_template=getattr(distill, "experiment_template", ""),
        holdout_fraction=getattr(distill, "holdout_fraction", 0.2),
        benchmark_dir=getattr(distill, "benchmark_dir", "data/benchmark"),
        train_dir=getattr(distill, "train_dir", "data/train"),
    )
    cmd = [getattr(agent_cfg, "claude_bin", "claude"), "-p", prompt,
           "--permission-mode", getattr(agent_cfg, "permission_mode", "acceptEdits")]
    if getattr(agent_cfg, "model", None):
        cmd += ["--model", agent_cfg.model]
    if getattr(agent_cfg, "plugin_dir", None):
        cmd += ["--plugin-dir", agent_cfg.plugin_dir]
    cmd += list(getattr(agent_cfg, "extra_args", None) or [])
    return cmd


def _launch(cmd: list[str], *, cwd: str | Path, log_file: Path, detach: bool) -> Any:
    """The one real side effect (monkeypatched in tests). Detached → fire-and-forget
    ``Popen`` in its own session; foreground → ``run`` and return the completed process."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if detach:
        f = log_file.open("w")
        return subprocess.Popen(
            cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
        )
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, check=False,
    )
    log_file.write_text((proc.stdout or "") + (proc.stderr or ""))
    return proc


_LAUNCH = _launch  # seam


def _snapshot_prompt(escalation_path: Path, *, agent_cfg: Any, root: Path, cwd: Path) -> None:
    """Copy the live prompt file to ``<root>/prompt-snapshots/<escalation>.txt`` so
    the UI can diff an autoresearch rewrite against what the agent started from.
    Best-effort: a missing prompt file just means no snapshot."""
    prompt = cwd / (getattr(agent_cfg, "prompt_file", None) or "prompt.txt")
    try:
        text = prompt.read_text()
    except OSError:
        return
    snap = root / "prompt-snapshots" / f"{escalation_path.stem}.txt"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text(text)


def spawn(escalation_path: str | Path, *, agent_cfg: Any, root: str | Path,
          cwd: str | Path | None = None, detach: bool = True) -> Any:
    """Spawn the trigger agent on one escalation event. Returns the Popen (detached)
    or the CompletedProcess (foreground). ``root`` is the trigger state dir."""
    escalation_path = Path(escalation_path)
    root = Path(root)
    verdict_path = root / "verdicts" / f"{escalation_path.stem}.json"
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = root / "agent-runs" / f"{escalation_path.stem}.log"
    _snapshot_prompt(escalation_path, agent_cfg=agent_cfg, root=root,
                     cwd=Path(cwd or root.parent.parent))
    remote_cfg = getattr(agent_cfg, "remote", None)
    if remote_cfg is not None and getattr(remote_cfg, "enabled", False):
        from .remote import spawn_remote

        log.info("[trigger] spawning REMOTE agent on %s (detach=%s)", escalation_path.name, detach)
        return spawn_remote(escalation_path, agent_cfg=agent_cfg, root=root,
                            cwd=Path(cwd or root.parent.parent),
                            verdict_path=verdict_path, log_file=log_file, detach=detach)
    cmd = build_command(escalation_path, agent_cfg=agent_cfg, root=root, verdict_path=verdict_path)
    log.info("[trigger] spawning agent on %s (detach=%s)", escalation_path.name, detach)
    return _LAUNCH(cmd, cwd=(cwd or root.parent.parent), log_file=log_file, detach=detach)


__all__ = ["build_command", "spawn"]

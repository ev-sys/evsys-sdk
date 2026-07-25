---
name: distill-traces
description: >
  How the distiller agent (trigger.agent.mode: distill) converts an escalated
  batch of coding-agent traces into an eval benchmark + training rows, then
  launches and monitors the PRESET experiment. The agent never designs an
  algorithm and never invokes training-decider — the experiment template is
  the contract. Use on a distill-mode escalation, after assess-traces says
  the batch is real.
---

# Distilling escalated coding traces into data + a preset run

You are the gatekeeper-turned-data-engineer. The gate escalated; `assess-traces`
said the failure batch is real. Your output is DATA plus a LAUNCHED PRESET RUN,
not research decisions.

## 1. The split comes first (contamination boundary)

Group traces by session (`trace_id`), order sessions by `metadata.timestamp`,
and cut the newest `holdout_fraction` of sessions as EVAL. Everything else is
TRAIN. Record the boundary timestamp. From this point:

- eval sessions feed ONLY the benchmark;
- train sessions feed ONLY training rows;
- if a session is ambiguous (spans the boundary), put it in eval.

## 2. Eval set → a Benchmark dir

Write `{benchmark_dir}/distill-<date>/`:

- `tasks.jsonl` — one HarborTask row per eval session: `prompt` = the session's
  first user message plus a short repo-context header (cwd, git branch);
  verifier: prefer an objective check when the trace shows one (a test command
  that must pass → the sandbox verifier; a known artifact string →
  `in_process` / `contains`), else `llm_judge`.
- `metadata.yaml` — `split: {train_until: <ts>, eval_from: <ts>}`, source repo,
  git branch, and how many sessions each side got.

## 3. Train sessions → training rows

Write `{train_dir}/distill-<date>.jsonl`, PROMPT_DATASET-shaped rows:
`{"inputs": {"question": <first user message + repo-context header>},
"expected": <the session's final assistant answer>}`. Skip sessions whose
feedback marks them as interrupted with no resolution.

## 4. Materialize + launch the preset experiment

1. `evsys new-experiment distill-<date>`
2. Copy the preset template over the scaffolded `config.yaml`. Fill ONLY:
   `name`, `data.path`, `metadata.benchmark.path`, and the snapshot block's
   `repo_dir`/`ref` (from trace metadata `cwd`/`git_branch`). Touch nothing
   else — especially not `algorithm`.
3. Launch `python experiments/<dir>/run.py` in the background.

## 5. Overlook training (bounded)

Poll the run's `outputs/<name>/` every few minutes: step metrics jsonl
(loss curve), `checkpoints.jsonl`. Abort the run if loss goes NaN or has not
improved for a third of max_steps. When it ends, write
`reports/distill-<date>.md`: final loss, checkpoint path, benchmark score
pre/post if available, and one paragraph on whether the failure mode from the
escalation looks addressed. Then stop — do not start another run.

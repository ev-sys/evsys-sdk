---
name: diagnose-experiment
description: >
  Diagnose a training experiment end-to-end: rendered training inputs (after
  transforms and templates), ground-truth labels, train/eval prompt parity,
  and final-model success/failure examples. Writes DIAGNOSIS.md with checks,
  bugs, and improvements. Use when debugging bad eval scores, train/eval
  mismatch, template bugs, coverage gaps, or when the user asks to diagnose
  or audit an experiment.
---

# Diagnose experiment

Systematically audit one experiment **before** trusting its metrics. Output a
single report: `experiments/<exp>/DIAGNOSIS.md`.

Read `getting-experiment-context` first if the run is on the dashboard; this
skill covers **local artifacts + config** regardless of where training ran.

## Quick start

```text
Task progress:
- [ ] 0. Locate experiment dir, config, and final checkpoint
- [ ] 1. Render training inputs (post-transform, post-template)
- [ ] 2. Verify ground truth used for training
- [ ] 3. Compare eval prompts vs training prompts
- [ ] 4. Collect final-model successes and failures
- [ ] 5. Write DIAGNOSIS.md
```

Follow the steps below and write `DIAGNOSIS.md` by hand. Do not skip the report.

---

## Step 0 — Gather context

| What | Where |
|------|--------|
| Config | `experiments/<exp>/config.yaml` (+ `config_*.yaml`, `run.py` overrides) |
| Run output | `experiments/<exp>/.evsys/` or `output_dir` in config |
| Aggregated metrics | `results.json`, `REPORT.md`, dashboard API |
| Eval rollouts | `.evsys/<run>/harbor_eval/**`, `partition_eval_rollouts.jsonl`, dashboard `predictions?kind=eval` |
| Training logs | `.evsys/<run>/logs/metrics.jsonl`, `hyperparams.json` |

Record: algorithm kind (`sft`, `sdft`, …), model, renderer, continual stages,
benchmark names, and run status (`completed` / `failed`).

---

## Step 1 — Exact training input (after transforms + templates)

Training input is **not** the raw JSONL row. Reconstruct the full pipeline:

```text
raw JSONL row
  → data.transforms[]     (config.run.data.transforms)
  → algorithm field extract (e.g. question from inputs.question / messages)
  → user_template         (algorithm.params.user_template)
  → demo_template         (SDFT teacher only — algorithm.params.demo_template)
  → renderer / chat template (model.renderer_name, enable_thinking)
  → tokenized ModelInput
```

### 1a. Raw row

Load 3–5 rows from the training path in config (`run.data.path` or continual
stage dataset). Note fields: `inputs`, `expected`, `messages`, `metadata`.

```python
import json
from pathlib import Path

def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

rows = load_jsonl("data/datasets/.../stage0.sdft.train.jsonl")
for r in rows[:3]:
    print(r.keys(), r.get("inputs"), r.get("expected"), r.get("metadata"))
```

### 1b. After transforms

Apply each transform in order (same as `runner._apply_transforms`):

```python
from evsys_sdk.registry import get_transform
from evsys_sdk.config import TransformSpec

specs = cfg.run.data.transforms  # list of {kind, params}
rows = raw_rows
for spec in specs:
    t = get_transform(spec.kind)(**(spec.params or {}))
    rows = list(t(rows))
```

Document what each transform changed (e.g. `jsonl_to_chat` adds `messages`).

### 1c. After algorithm templates

**SFT / anchor:** user turn = `user_template.format(question=..., prompt=...)`.

**SDFT student rollout:** zero-shot user turn (same `user_template`, no demo).

**SDFT teacher prompt:** user turn = `demo_template.format(question=..., golden_answer=...)`.

Render with the same renderer the run used:

```python
# SDFT path (feature branch / renderer_name set):
# algorithm._student_user_content(q) → renderer.build_generation_prompt(messages)

# Generic chat-template path:
from evsys_sdk.training.templates import messages_to_model_input
# messages = [system, user(demo_template...)] for teacher
# messages = [system, user(user_template...)] for student / SFT anchor
```

**Print decoded text** (not just token ids) for at least 3 rows. Flag:

- Double prefixes (e.g. `Query: Query: ...` when JSONL already has `Query:`)
- Thinking tokens when `enable_thinking: false`
- Demo template leaking into student zero-shot prompt (should not)
- `system_prompt` mismatch across teacher / student / eval

### 1d. Completion token stats

From `.evsys/<run>/logs/metrics.jsonl`:

- `sdft/total_completion_tokens` — should be low (~150–300) for no-thinking
- `train/mean_loss` — sanity check curves

---

## Step 2 — Ground truth used for training

| Algorithm | Label source | Where it appears |
|-----------|--------------|------------------|
| SFT | `expected` or assistant message | CE on golden completion tokens |
| SDFT distillation | Teacher top-K on **student completion** | Not the raw `expected` slug directly |
| SDFT hybrid anchor | `expected` → `<answer>{slug}</answer>` | `sdft/n_sft_anchors`, `sft_anchor_alpha` |
| RL / harbor | Verifier reward on rollout | `expected` in task verifier |

For each sample row, record:

```text
row_id / tool_slug: <metadata.tool_slug or task_id>
raw expected: <expected field>
training target: <what loss actually optimizes>
  SDFT: student samples then teacher top-K at sampled tokens
  anchor: <answer>SLUG</answer> (+ stop token)
```

**Checks:**

- [ ] `expected` non-empty and matches metadata (`tool_slug`, verifier slug)
- [ ] Golden slug appears in training data for that toolkit (coverage)
- [ ] Hybrid anchor uses zero-shot user prompt when `sft_anchor_zero_shot: true`
- [ ] For classification: acceptable alternates listed in benchmark verifier?

Compare benchmark tools vs train tools (unique slugs):

```bash
.venv/bin/python data/process/toolkit_tool_coverage_table.py   # composio-bench
```

High `bench \ train` → eval tasks the model never saw as labels.

---

## Step 3 — Eval input parity (train vs eval)

Eval prompt chain (from `metadata.benchmark[]`):

```text
task.instruction (benchmark tasks.jsonl)
  → chat_template.user_template   (often "Query: {prompt}")
  → chat_template.system_prompt
  → enable_thinking
  → renderer / ChatTemplatedInference
  → model.generate
```

Training student/anchor chain:

```text
row question field
  → algorithm.params.user_template   (often "{question}")
  → algorithm.params.system_prompt
  → enable_thinking / renderer_name
```

### Side-by-side table (required in report)

For one in-distribution task, show:

| Surface | system_prompt | user content | enable_thinking | renderer |
|---------|---------------|--------------|-----------------|----------|
| Train (student) | … | … | … | … |
| Train (teacher demo) | … | … | … | … |
| Train (SFT anchor) | … | … | … | … |
| Eval | … | … | … | … |

**Common bugs:**

| Symptom | Likely cause |
|---------|--------------|
| Train great, eval ~0 | `user_template` differs (`{question}` vs `Query: {prompt}`) or double prefix |
| Long CoT at eval | `enable_thinking: true` on eval, false on train |
| Random slugs | Eval missing `chat_template` wrapper |
| Good diagonal, bad off-diagonal | Expected for continual; check replay / frozen teachers |
| pass@1 high on val, low on full eval | Val split ⊆ train; full eval has uncovered tools |

Extract eval template from config:

```yaml
metadata:
  benchmark:
    - name: val_tk_GMAIL
      path: data/benchmark/composio_val_tk_GMAIL
      chat_template:
        system_prompt: "..."
        user_template: "Query: {prompt}"
        enable_thinking: false
```

Extract train template from:

```yaml
run:
  algorithm:
    params:
      user_template: "{question}"
      system_prompt: "..."
      enable_thinking: false
      renderer_name: qwen3_5_disable_thinking
```

**Rule:** The string passed to the model's user turn at eval must match the
student/anchor training distribution. Teacher demo format does not need to
match eval (teacher-only).

---

## Step 4 — Final model successes and failures

Use the **last completed checkpoint** (or user-specified stage).

### Local rollouts

| Source | Path |
|--------|------|
| In-run harbor eval | `.evsys/<run>/harbor_eval/<bench>/jobs/**/agent/completion.txt` |
| Post-hoc matrix eval | `partition_eval_rollouts.jsonl` |
| Aggregated only | `results.json` metrics (no examples — re-run eval to get rollouts) |

If only aggregates exist, run a small re-eval (≤20 tasks) and save JSONL:

```python
# Pattern: partition_eval_sdft.py — score checkpoint, append per-task rows:
# {task_id, instruction, expected, outputs, rewards, pass, toolkit, ...}
```

### Dashboard

```python
preds = store.get_predictions(run_id, kind="eval", limit=200)
failures = [p for p in preds if (p.get("reward") or 0) < 1.0]
successes = [p for p in preds if (p.get("reward") or 0) >= 1.0]
```

### Classify failures (minimum 10)

| Pattern | Meaning |
|---------|---------|
| `wrong_slug` | Parsed `<answer>` but incorrect tool |
| `no_answer_tag` | Missing `<answer>...</answer>` |
| `near_miss` | Synonym slug (e.g. `GMAIL_SEND_EMAIL` vs `GMAIL_CREATE_EMAIL_DRAFT`) |
| `uncovered_tool` | Expected slug never in training data |

Include 3 successes + 3 failures per eval partition / toolkit in the report.

---

## Step 5 — Write DIAGNOSIS.md

Save to `experiments/<exp>/DIAGNOSIS.md`. Use [report-template.md](report-template.md).

Severity tags:

- **CRITICAL** — train/eval mismatch, wrong labels, missing checkpoint
- **WARNING** — coverage gaps, short training, high forgetting
- **INFO** — optimization suggestions

Always end with **Recommended next steps** (ordered, actionable).

---

## SDK reference (where logic lives)

| Concern | Module |
|---------|--------|
| Transform pipeline | `evsys_sdk/runner.py` → `_apply_transforms` |
| SDFT templates | `evsys_sdk/algorithms/sdft.py`, `training/sdft_data.py` |
| Eval chat wrap | `evsys_sdk/inference/chat_templated.py` |
| Benchmark scoring | `evsys_sdk/experiment.py` → `_eval_arm`, `_eval_arm_harbor` |
| Harbor rollouts | `evsys_sdk/training/harbor_eval.py` |

---

## Related skills

- `getting-experiment-context` — dashboard history + predictions API
- `using-the-sdk` — EvsysStore, Workspace, experiment launch

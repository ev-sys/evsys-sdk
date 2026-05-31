---
name: Using the trajectory-labs SDK
description: How to push training runs (SFT/RL/distillation) to the Trajectory dashboard. Use when writing experiment scripts that should appear in the dashboard, or when wiring metrics/predictions/conclusions into an existing training loop.
---

# Using the trajectory-labs SDK

This SDK is the **write side** of the Trajectory dashboard. It pushes
experiments, per-step training metrics, eval runs, predictions, and a final
conclusion to the backend at `/api/dashboard/api/sdk/...`. The backend persists
to Supabase; the dashboard at `dev.trajectoryevals.com` reads from there.

If you only need to **read** previous runs (history, prior hypotheses,
checkpoints), use the `getting-experiment-context` skill instead.

## Quick install

```bash
pip install -e /path/to/trajectory-labs-sdk    # local dev
# or, once published:
pip install trajectory-labs
```

Required env:
```bash
export TRAJECTORY_API_URL="https://backend-dev-p0tj.onrender.com"
export TRAJECTORY_API_KEY="sk_..."   # from dashboard → Settings → API keys
```

## The 80% case: `Experiment.from_yaml(...).run()`

In a project scaffolded with `trajex init-project`, every experiment lives at
`experiments/<yyyymmdd>_<slug>/{config.yaml,run.py}`. The script is three
lines — all knobs live in YAML:

```python
# experiments/20260531_lora_rank_sweep_4b/run.py
from trajectory_labs import Experiment
import scripts  # registers project verifiers / metrics / transforms

Experiment.from_yaml("config.yaml").run()
```

Config carries the hypothesis, success metric, benchmark, and sweep matrix
under `metadata` + the usual `RunConfig` blocks:

```yaml
name: lora_rank_sweep_4b_sft
output_dir: ./.trajectory/outputs/lora_rank_sweep_4b_sft

metadata:
  hypothesis: "Higher LoRA rank → higher composio pass@1"
  tags: [sft, qwen3_4b, lora-rank-sweep]
  success_metric: pass_rate
  benchmark:
    path: data/benchmark/composio_eval_v2
    id: <paste from `trajex benchmark upload data/benchmark/composio_eval_v2`>
    breakdown_keys: [toolkit]

matrix:
  base_run:
    name: sft_4b_think
    seed: 42
    data:
      source_kind: jsonl
      path: data/datasets/sft_overdose/v17/train.jsonl
    model:
      name: Qwen/Qwen3-4B
      renderer_name: qwen3_5
    algorithm:
      kind: tinker_sft
      params:
        learning_rate: 1.0e-4
        num_epochs: 10
        batch_size: 32
        lora_rank: 0          # placeholder — swept below
    backend:
      kind: tinker
  axes:
    algorithm.params.lora_rank: [1, 4, 16]
  name_template: "{base}__r{algorithm.params.lora_rank}"
```

What `Experiment.run()` does for you, so you don't hand-roll it:
  * creates the experiment record on the dashboard with hypothesis + tags;
  * expands the matrix and creates one run record per arm;
  * **isolates per-arm failures** — one raising arm doesn't kill the sweep;
  * auto-forwards the local `metrics.jsonl` to `store.log_metrics`
    (no `backfill_step_metrics` call);
  * scores each completed arm against `metadata.benchmark` and records the
    eval row;
  * picks the best arm by `success_metric` and writes a conclusion +
    `best_score` on the experiment.

Subclass `Experiment` to override `_eval_arm`, `_pick_best`,
`_build_conclusion`, etc. when a project needs project-specific scoring or
ranking — the default behavior is the common case.

## Scaffolding new things

```bash
trajex init-project <name>                       # whole project skeleton
trajex new-experiment <slug> [--project-root .]  # experiments/<today>_<slug>/
trajex benchmark upload data/benchmark/<name>    # register a harbor benchmark
```

## Low-level path: `ExperimentRun` context manager

Use this only when you're doing something `Experiment` doesn't model yet
(custom rollout loops, manual prediction streaming, post-hoc patching):

```python
from trajectory_labs import DashboardClient, ExperimentRun

client = DashboardClient()  # picks up env vars

with ExperimentRun(
    client,
    experiment_name="sft_lora_r32_v1",
    hypothesis="LoRA r=8 → r=32 lifts pass@1 on the eval set",
    hypothesis_reasoning=(
        "Prior r=8 runs plateaued at 0.71 while train-loss was still "
        "dropping → adapter capacity is the bottleneck, not data."
    ),
    plan="SFT 2 epochs, lr=1e-5, bs=16, eval every 250 steps on v3 split.",
    tags=["axis:lora", "recipe:sft"],
    recipe_kind="sft",
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    run_config={"lr": 1e-5, "batch_size": 16, "epochs": 2, "lora_r": 32},
) as run:
    for step, batch in enumerate(loader, start=1):
        loss = train_step(batch)
        if step % 50 == 0:
            run.log_step(step, loss=loss, learning_rate=lr_schedule(step))

        if step % 250 == 0:
            metrics, preds = evaluate(model)            # your eval fn
            run.log_eval(step=step, benchmark_id=benchmark_id,
                         metrics=metrics)               # {"pass_at_1": ..., ...}
            run.log_predictions(preds)                  # list[dict]; see schema below

    run.set_best_score(best_pass_at_1)
    run.set_conclusion(
        "r=32 raised pass@1 from 0.71 → 0.83 with no train-loss instability. "
        "Hypothesis confirmed; promote checkpoint at step 1500."
    )
# clean exit → run + experiment auto-marked completed,
# best_score + conclusion patched on the experiment.
# raised exception → both marked failed with the exception message.
```

## What each field means

| Field | Where it lives | Why it matters |
|---|---|---|
| `experiment_name` | `training_experiments.experiment_name` | shown on the experiments grid |
| `hypothesis` | `training_experiments.hypothesis` | the eyebrow line on the experiment page |
| `hypothesis_reasoning` | `training_experiments.hypothesis_reasoning` | the WHY behind the hypothesis — what prior evidence motivated it |
| `plan` | `training_experiments.plan` | the concrete recipe in words |
| `conclusion` | `training_experiments.conclusion` | the takeaway, set at end of run |
| `tags` | `training_experiments.tags` | filter chips on the dashboard |

(The model is **not** an experiment field — it's a per-run hyperparameter in
`runs.run_config`. `client` was removed; the project is the top-level scope.)

`hypothesis_reasoning` and `conclusion` are what future agents (and future
you) read first when revisiting this experiment. Treat them as the
experiment's commit message, not as optional metadata.

## Prediction row schema (`run.log_predictions`)

Each item in the list:
```python
{
    "kind": "eval" | "rollout",
    "task_id":     "...",        # stable id of the eval item
    "instruction": "...",        # what the model was asked
    "model_output":"...",        # what the model produced
    "expected":    "...",        # ground truth (eval) or None (rollout)
    "reward":      1.0,          # float — used to compute pass-rate
    "step":        500,          # optional, for eval-over-time
    "eval_id":     "...",        # optional, links to the evals row
    "sample_idx":  0,            # optional, for RL rollouts
    "advantage":   0.42,         # optional, for RL
    "metadata":    {...},        # optional free-form
}
```

Push these in bulk per eval step — the dashboard renders them as the Eval
results table on the run page.

## Step-metric schema (`run.log_step`)

```python
run.log_step(step, loss=..., accuracy=..., learning_rate=...,
             grad_norm=..., tokens_per_sec=...)
```

All fields optional except `step`. Renders as the training-loss chart and
the per-step metrics table on the run page.

## When to use the low-level `DashboardClient` directly

- Multi-run campaigns in one experiment (e.g. groups × seeds) → pass
  `experiment_id=...` to successive `ExperimentRun(...)` blocks, or call
  `client.create_run(experiment_id=..., group_id=..., seed=...)` yourself.
- Recording an eval of a run on a benchmark:
  `client.create_eval(run_id, benchmark_id=..., metrics={...})`.
- Patching a finished experiment with a revised conclusion:
  `client.update_experiment(exp_id, conclusion="...")`.

## Common mistakes to avoid

1. **Don't fabricate `wandb_run_url`**. If you didn't actually start a
   wandb run, leave it unset — a bad URL renders as a broken iframe.
2. **Don't push a `model_output` that is a Python `repr()` of a list of
   messages**. Render the assistant turn as plain text first; the
   dashboard displays it verbatim.
3. **Don't call `set_best_score` with the latest step's score** — pass the
   best score across the whole run (typically `max(pass_at_1)` across eval
   steps).
4. **One `ExperimentRun` = one run.** If you want multiple runs (groups ×
   seeds) under the same experiment, reuse `experiment_id`:
   ```python
   exp = client.create_experiment(...)
   for sweep_config in sweep:
       with ExperimentRun(client, experiment_id=exp["id"], ...) as run:
           ...
   ```
5. **Errors are not silent.** Any non-2xx raises `DashboardClientError`
   with the HTTP body in the message — read it; the backend's whitelist
   tells you which fields it rejected.

## End-to-end sanity check

A small smoke run confirms the full SDK → backend → Supabase → dashboard loop.

---
name: Using the trajectory-experiments SDK
description: How to push training runs (SFT/RL/distillation) to the Trajectory dashboard. Use when writing experiment scripts that should appear in the dashboard, or when wiring metrics/predictions/conclusions into an existing training loop.
---

# Using the trajectory-experiments SDK

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
pip install trajectory-experiments
```

Required env:
```bash
export TRAJECTORY_API_URL="https://backend-dev-p0tj.onrender.com"
export TRAJECTORY_API_KEY="sk_..."   # from dashboard → Settings → API keys
```

## The 80% case: `ExperimentRun` context manager

```python
from trajectory_experiments import DashboardClient, ExperimentRun

client = DashboardClient()  # picks up env vars

with ExperimentRun(
    client,
    experiment_name="composio_sft_lora_r32_v1",
    client_name="composio",
    hypothesis="LoRA r=8 → r=32 lifts pass@1 on composio_eval_v3",
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
            run.log_eval(step=step, eval_name="composio_eval_v3",
                         metrics=metrics)               # {"pass_at_1": ..., ...}
            run.log_predictions(preds)                  # list[dict]; see schema below

    run.set_best_score(best_pass_at_1)
    run.set_conclusion(
        "r=32 raised pass@1 from 0.71 → 0.83 with no train-loss instability. "
        "Hypothesis confirmed; promote checkpoint at step 1500 to leaderboard."
    )
# clean exit → generation + experiment auto-marked completed,
# best_score + conclusion patched on the experiment.
# raised exception → both marked failed with the exception message.
```

That single block is **all you need for a typical run**. The lower-level
`DashboardClient` methods exist for non-standard flows.

## What each field means

| Field | Where it lives | Why it matters |
|---|---|---|
| `experiment_name` | `training_experiments.experiment_name` | shown on the experiments grid |
| `hypothesis` | `training_experiments.hypothesis` | the eyebrow line on the experiment page |
| `hypothesis_reasoning` | `training_experiments.hypothesis_reasoning` | the WHY behind the hypothesis — what prior evidence motivated it |
| `plan` | `training_experiments.plan` | the concrete recipe in words |
| `conclusion` | `training_experiments.conclusion` | the takeaway, set at end of run |
| `tags` | `training_experiments.tags` | filter chips on the dashboard |
| `client_name` | `training_experiments.client` | scopes the workspace (e.g. "composio") |

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
    "eval_name":   "composio_eval_v3",   # optional
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

- Multi-generation sweeps in one experiment → pass `experiment_id=...` to
  successive `ExperimentRun(...)` blocks, or call `client.create_generation`
  yourself.
- Posting a benchmark result tied to a `test_dataset_id`:
  `client.record_benchmark(test_dataset_id=..., model_ref=..., score=...)`.
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
4. **One `ExperimentRun` = one generation.** If you want multiple
   generations under the same experiment, reuse `experiment_id`:
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

A working smoke run lives at `/tmp/e2e_composio_mini.py` in the parent
repo. Confirms the full SDK → backend → Supabase → dashboard loop.

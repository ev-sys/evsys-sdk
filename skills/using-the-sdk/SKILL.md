---
name: using-the-sdk
description: >
  How to use the trajectory-labs SDK to read project goals + experiment history
  and to create/launch experiments: TrajectoryStore (backend-routed data access)
  and Workspace (local dataset cache). Use when writing code that reads or writes
  experiments/runs/datasets/benchmarks/metrics, or materializes data for training.
---

# Using the trajectory-labs SDK

Two objects. **`TrajectoryStore`** routes every call through the backend gateway
with your Bearer API key — **no Supabase key in the SDK**; the backend checks
project membership and does the DB I/O. **`Workspace`** caches remote datasets to
local JSONL for fast training.

```bash
export TRAJECTORY_API_URL="https://<backend>"   # backend base URL
export TRAJECTORY_API_KEY="sk_..."               # dashboard → Settings → API keys
export TRAJECTORY_PROJECT_ID="<uuid>"            # default project for project-scoped calls
```

```python
from trajectory_labs import TrajectoryStore, Workspace
store = TrajectoryStore()           # reads the env above
```

## Hierarchy

`project → goals[] (versioned) + datasets[]/benchmarks[] → experiments → groups → runs → checkpoints/evals/metrics`

## Reading (agent context)

```python
store.current_goal()                       # project's active goal (read-only!)
store.list_goals()                         # all goal versions, oldest→newest

store.experiment_summaries()               # [{id,name,hypothesis,conclusion,is_valid}] — broad scan
store.experiment_detail(exp_id)            # {experiment, groups:[{...,runs:[{run_config,evals,checkpoints}]}]}
store.experiment_detail(exp_id, include_metrics=True)   # + per-run train-log series
store.get_metrics(run_id, name="val_loss", split="val") # a single metric series
store.list_datasets(); store.list_benchmarks()
```

`run_config` is the run's **hyperparameters** (incl. the model). `evals` carry the
benchmark result metrics (`pass_at_1`, …). Skip `is_valid=False` experiments as evidence.

## Writing (launching an experiment)

```python
exp = store.create_experiment(experiment_name="small_vs_large_model",
                              hypothesis="the smaller model matches the larger on this benchmark",
                              project_goal_id=store.current_goal()["id"])
g4 = store.create_group(exp["id"], "Qwen 4B", description="v2 dataset, train Qwen 4B")
for seed in (1, 2):
    run = store.create_run(experiment_id=exp["id"], group_id=g4["id"], seed=seed,
                           recipe_kind="sft",
                           run_config={"model": "Qwen/Qwen3-4B", "lr": 1e-5})  # model = hyperparam
    # ... train ... then:
    store.log_metrics(run_id=run["id"], step=10, metrics={"loss": 1.2})
    store.log_metrics(run_id=run["id"], step=10, split="val", metrics={"val_loss": 1.4})
    ckpt = store.add_checkpoint(run_id=run["id"], uri="tinker://final", label="final",
                               step=100, is_final=True)
    store.create_eval(run_id=run["id"], benchmark_id=bm["id"], checkpoint_id=ckpt["id"],
                      metrics={"pass_at_1": 0.83})
    store.update_run(run["id"], status="completed")

store.set_conclusion(exp["id"], "4B matched 9B at half the cost — promote.")
# store.invalidate_experiment(exp_id, reason="...")   # if a bug is found later
```

## Materializing data for training (Workspace)

Remote-first: pull a dataset to local JSONL **once**, then train from the file.

```python
ws = Workspace(store)                       # root: $TRAJECTORY_WORKSPACE or ./.trajectory (gitignored)
mat = ws.pull_dataset(dataset_id)           # cache-hit if already local; else fetch+write
# mat.path  → JSONL of RAW rows (one per line)
# mat.format, mat.transform → how to render raw → typed (source_kind=jsonl + transform)
bench = ws.pull_benchmark(benchmark_id)
script = ws.script_path(exp["id"]); out = ws.outputs_dir(run["id"])
```

Rows are stored **raw** (+ the dataset's recorded `transform`); render typed rows
on read. `pull_dataset(..., force=True)` re-pulls; otherwise a valid local copy is
reused (manifest-guarded).

## Rules of thumb

- **Never** call `store.set_goal(...)` unless the user explicitly asks to change the goal.
- The model is a **per-run hyperparameter** in `run_config`, not an experiment field.
- `hypothesis` and `conclusion` are the experiment's commit message — write them well.
- Verifiers/metrics are SDK-registered **by name** (`@register_verifier` / `@register_metric`);
  benchmark rows reference a verifier by `verifier_name`, not inline code.

---
name: Getting context from previous Trajectory experiments
description: How to pull history of past experiments, generations, eval results, predictions, and rendered training data from the Trajectory dashboard backend — so you can build on prior runs instead of starting cold. Use before proposing a new hypothesis or recipe.
---

# Getting context from previous Trajectory experiments

Before proposing a new experiment, **read what already ran**. Skipping
this is the #1 cause of duplicate work and rediscovered dead-ends.

This skill covers the **read side** of the Trajectory backend. To push
runs to the dashboard use the `using-trajectory-sdk` skill.

## Endpoints

All endpoints are auth'd via the same Bearer token as the SDK
(`TRAJECTORY_API_KEY`). Base URL: `TRAJECTORY_API_URL` (dev:
`https://backend-dev-p0tj.onrender.com`).

| Endpoint | Returns |
|---|---|
| `GET /api/dashboard/api/experiments/?client=<name>&limit=N` | list of experiments (newest first) |
| `GET /api/dashboard/api/experiments/<exp_id>/` | one experiment, full row |
| `GET /api/dashboard/api/experiments/<exp_id>/generations/` | every generation under an experiment |
| `GET /api/dashboard/api/generations/<gen_id>/` | one generation, full row |
| `GET /api/dashboard/api/generations/<gen_id>/evals/` | eval timeline (one row per `(step, eval_name)`) |
| `GET /api/dashboard/api/generations/<gen_id>/predictions/?kind=eval&limit=N&offset=N` | per-task model outputs |
| `GET /api/dashboard/api/generations/<gen_id>/rendered/?limit=N&offset=N` | exact rendered training rows |
| `GET /api/dashboard/api/generations/<gen_id>/step-metrics/` | training-loss timeline |
| `GET /api/dashboard/api/benchmark-runs/?test_dataset_id=...` | leaderboard rows |
| `GET /api/dashboard/api/problem-statements/?client=...` | declared problems for a client |

## The recommended recall flow

```python
import os, requests

API = os.environ["TRAJECTORY_API_URL"]
H   = {"Authorization": f"Bearer {os.environ['TRAJECTORY_API_KEY']}"}

# 1) What's been tried for this client?
exps = requests.get(f"{API}/api/dashboard/api/experiments/",
                    params={"client": "composio", "limit": 50},
                    headers=H).json()["experiments"]

# 2) For each, read the high-signal fields. These are the four you want:
for e in exps:
    print(e["experiment_name"], "→", e.get("best_score"))
    print("  hypothesis:        ", e.get("hypothesis"))
    print("  why we believed it:", e.get("hypothesis_reasoning"))
    print("  plan:              ", e.get("plan"))
    print("  conclusion:        ", e.get("conclusion"))
    print("  tags:              ", e.get("tags"))

# 3) Drill into the best generation of the most promising experiment:
exp_id = exps[0]["id"]
gens = requests.get(f"{API}/api/dashboard/api/experiments/{exp_id}/generations/",
                    headers=H).json()["generations"]
best = max((g for g in gens if g.get("combined_score") is not None),
           key=lambda g: g["combined_score"], default=None)

# 4) Look at where it failed (the predictions with reward=0):
if best:
    preds = requests.get(
        f"{API}/api/dashboard/api/generations/{best['id']}/predictions/",
        params={"kind": "eval", "limit": 200},
        headers=H,
    ).json()["predictions"]
    failures = [p for p in preds if (p.get("reward") or 0) < 1.0]
    # → inspect failures; they tell you what the next experiment should attack.
```

## Reading conclusions FIRST

`conclusion` and `hypothesis_reasoning` are the highest-signal fields in
the database. They are written by humans (or by an agent at end of run)
specifically so future readers don't have to reverse-engineer the result
from raw metrics.

When asked "what's been tried on X?", load these two fields for every
experiment matching X *before* loading any metrics or predictions. They
will collapse the search space dramatically.

## When metrics disagree with conclusions

Trust the conclusion — but verify by spot-checking 5–10 failure
predictions. Frequent causes of "good score, bad conclusion":
- the run was scored on a custom split (`eval_dataset.official == false`)
- evaluator rewards were too generous (look at `metadata`)
- the model memorised the eval set (check `tags` for "leakage")

## Building a context bundle for a new hypothesis

Before proposing a hypothesis, assemble:
1. The 5 most recent experiments with this `client`.
2. Their `(hypothesis, hypothesis_reasoning, conclusion, best_score, tags)`.
3. The best generation's `run_config` and its top-10 eval failures.
4. Any matching `problem_statements/<id>/` so the problem framing is fixed.

Then write your new `hypothesis` and `hypothesis_reasoning` such that they
explicitly reference what changed relative to the most-similar prior run.
The dashboard will thread these via `parent_experiment_id` if you pass it.

## CLI shortcut

In the parent repo, `scripts/exp_query.py` wraps these endpoints with
shell-friendly subcommands (`list`, `show`, `preds`, `failures`). Prefer
it for ad-hoc exploration; use the Python flow above when feeding context
into an agent loop.

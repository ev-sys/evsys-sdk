# Local logging design — human track + agent track

Status: proposal. Branch `feat/local-logging`, based on `feat/harbor-integration` (#37).

> Foundation note: the offline `LocalStore` + `resolve_store` toggle live in #33
> (`feat/local-store`) and are **not** on this base yet. This branch currently has the
> original `LocalExperimentStore` (flat-by-id mirror). The two-track layout below assumes
> that `LocalStore`/`resolve_store` foundation is merged/cherry-picked first; until then
> the same renderers can hang off `LocalExperimentStore` directly.

## Goal

Every experiment writes to **one local root (`.evsys/`)** in **two parallel surfaces**:

- **human/** — clean, minimal, decoded-to-text. Only the critical milestones and a few
  representative samples per phase. A researcher opens one file and sees how the run is
  going. No token-id dumps, no full metric firehose.
- **agent/** — dense, complete, machine-shaped (JSON/JSONL, token ids, every metric,
  every trajectory). What an agent or the local UI reads back; lossless.

One write event feeds both. The human renderer *curates*; the agent renderer *keeps
everything*.

## Where we are today (on this base)

Local mirroring is `LocalExperimentStore`, writing **flat-by-id** under the local root
(now `.evsys/` after the root unification on this branch):

```
.evsys/
  experiments/{experiment_id}/experiment.json
  generations/{run_id}/generation.json, metrics.jsonl, evals.jsonl, predictions.jsonl
```

This flat-by-id JSONL is effectively **already the agent track** — it just isn't nested
under its experiment, isn't human-readable, and the human-relevant data (decoded
rollouts, supervised/target tokens, a data preview) never reaches the store at all
(it lives only under `output_dir/harbor_rollouts/` and inside the algorithm at tokenize
time). So today there is no human track and the agent track is scattered.

## Target directory layout

One readable folder per experiment; runs/arms named by arm name, not `run_id`.

```
.evsys/
  cache/                              # project-wide dataset/benchmark pull cache (machine)
  experiments/
    {exp-slug}/                       # e.g. "tool-search-lr-sweep"
      summary.md                      # ← HUMAN ENTRY POINT (hypothesis, status, arms, best, conclusion)
      human/
        01_data/        datasets.md, sample.jsonl          # what's going in (+5 rendered rows)
        02_rollouts/    rollouts.md                        # a few decoded trajectories (best/median/worst)
        03_target_tokens/ supervised_examples.md          # supervised spans marked; SDFT top-K
        04_training/    metrics.csv, checkpoints.md        # whitelisted scalars; checkpoint list
        05_benchmark/   results.md, predictions.md         # metrics table; failure-biased samples
      agent/
        index.json                                         # slug ↔ experiment_id ↔ run_ids ↔ dashboard ids
        experiment.json, groups.jsonl
        runs/{arm-name}/
          run.json metrics.jsonl evals.jsonl predictions.jsonl checkpoints.jsonl
          rollouts.jsonl target_tokens.jsonl hyperparams.json run_result.json
          harbor/                                           # harbor's own jobs dir, relocated here
```

Both `LocalExperimentStore` and `Workspace` write a self-ignoring `.gitignore` (`*`) at
`.evsys/`, so the whole tree stays untracked.

## How the logging is designed (architecture)

```
producers ──emit──> LocalLogger ──fan-out──> [ HumanRenderer, AgentRenderer ]
(store calls,                                  (curate)        (verbatim)
 loop, algos,
 harbor, workspace)
```

1. **`LocalLogger`** — one object with an explicit, phase-aware event vocabulary. It owns
   the two renderers and just fans each event out. It is the local sink that the store
   writes through (the local arm of `resolve_store` once #33 lands).

2. **Two renderers** implement the same event interface:
   - `AgentRenderer` = today's flat-by-id behavior, re-rooted under `agent/runs/{arm}/`
     (nested, lossless). No sampling.
   - `HumanRenderer` = curated markdown/CSV projection per the curation rules below.

3. **Event vocabulary** = the current store surface **plus** the phase events the human
   track needs (these are the new ones):
   - have today: `experiment_created/updated`, `run_created/updated`, `step_metrics`,
     `eval`, `predictions` (and `group`/`checkpoint` once #33's surface lands)
   - **new**: `data_materialized(samples)`, `target_tokens(samples)`, `rollouts(group)`

4. **Producers of the new events** emit into the same logger, handed down via
   `RunContext` (e.g. `ctx.extras["local_logger"]`):
   - `Workspace._materialize` → `data_materialized` (n_rows, transforms, ~5 rendered rows)
   - algorithm tokenize step (SFT/RL/SDFT) → `target_tokens` (supervised spans / top-K)
   - `run_harbor_rollouts` → `rollouts` (decoded turns + reward + usage)
   - loop checkpoint manager → `checkpoint`
   This is what finally lands rollouts/target-tokens/data in a human-visible place.

5. **Remote parity.** Renderers are independent of transport, so the same human/agent
   tree can be produced whether the run is local or remote — the human track is useful
   even when records also go to the dashboard. Registered as `log_store` kinds
   `human` / `agent` and composed via the existing `multiplex` store, so a researcher
   can toggle either from config.

## Curation rules (what makes the human track "no overload")

| Phase | Human keeps | Agent keeps |
|---|---|---|
| Data | first ~5 rendered rows + dataset name/version/format/n_rows/transforms | all rows ref + full meta |
| Rollouts | ~3 per sampled step (best/median/worst by reward), decoded to text; cadence = eval points + first/last step | every trajectory, token ids, usage |
| Target tokens | a few examples with supervised span marked + counts; SDFT top-K for a few positions | full masks/weights/advantages/logprobs, SDFT (N,K) targets |
| Training metrics | whitelist (`loss/nll`, `reward_mean`, `lr`, `val_*`) → `metrics.csv`; milestones → `summary.md` | every metric, every step |
| Benchmark | metrics table + breakdowns; failure-biased sample of predictions | all per-task predictions + token ids |

## Harbor's own logging (relevant to the agent track)

Harbor's `Job` engine writes a dense native tree at `jobs_dir = workspace_dir / "jobs"`,
and the host-side `EvsysVerifier` scores by reading files in the task dir
(`completion.txt` + `evsys_verifier.json`). Today `workspace_dir` is hardcoded to
`output_dir/harbor_rollouts` and harbor's `quiet`/`debug` are never set.

For this design:
- point harbor's `workspace_dir`/`jobs_dir` into `agent/runs/{arm}/harbor/` (so the dense
  dump lives in the agent track, not loose under `output_dir`);
- thread harbor's `quiet=True` through config so the human console isn't flooded.

## Implemented (this branch)

`RunLog` (`src/evsys_sdk/run_log.py`) realizes the two tracks **per run**, rooted at
the run dir (`.evsys/outputs/{slug}/{run}/`):

```
{run_dir}/
  human/   summary.md, 01_data/, 02_rollouts/, 03_target_tokens/, 04_training/, 05_benchmark/
  agent/   harbor/{train,val,sdft,eval/<bench>}/   ← harbor's own jobs_dir, REFERENCED not copied
  logs/    run_result.json                          ← existing agent-grade artifacts
```

**No copies of harbor data.** Harbor already writes a complete `result.json` per trial
(`<jobs_dir>/<job>/<trial>/result.json`: rollout token ids, reward, usage). So callers
pass `run_log.harbor_dir(phase)` as harbor's `workspace_dir` → harbor's `jobs/` lands in
the agent track natively. The human `02_rollouts/rollouts.md` is a *curated* view
(reward stats + best/median/worst decoded completions) rendered from the already-harvested
in-memory `TrajectoryGroup`s, with a pointer to `agent/harbor/{phase}/` for the full dump.

Wiring:
- `runner._execute_run` creates the `RunLog`, calls `log_data` (01), puts it in
  `ctx.extras["run_log"]`, and at the end renders `metrics.csv` (04) + `summary.md`.
- `rl` / `sdft` set their harbor workspace to `run_log.harbor_dir("train"|"sdft")` and
  `note_rollouts(...)` on a bounded cadence (step 0, every 10).
- `base` routes in-loop eval to `run_log.harbor_dir("val")`.
- `experiment._eval_arm_harbor` routes to `run_log.harbor_dir("eval/<bench>")` and renders
  `05_benchmark` (`results.md` + failure-biased `predictions.md`).

All `RunLog` writes are best-effort (never raise into the training loop) and the wiring is
guarded (`ctx.extras.get("run_log")`), so absence is a no-op.

## Remaining

- **`03_target_tokens`**: the `log_target_tokens` hook exists; wire SFT/RL/SDFT tokenize
  steps to emit a few supervised-span / top-K examples.
- **Per-eval in-loop rollout curation**: `base`/evaluators route val rollouts to the agent
  track but don't yet `note_rollouts` the curated human view (only post-training eval does).
- **Experiment-level rollup**: a top-level `experiments/{slug}/summary.md` across arms
  (today summary is per-run).
- **Harbor verbosity**: thread `quiet=True` into `JobConfig` so the human console stays clean
  (harbor writes all files regardless of `quiet`).

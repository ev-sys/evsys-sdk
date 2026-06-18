# Local logging design

Status: implemented. Branch `feat/local-logging`, based on `feat/harbor-integration` (#37).

## Goal

**One** clean, human-readable log per run — no separate "agent" dump — organized into
folders. It contains only the essentials, decoded to text (never token ids), and lives
under the run dir using the same local `.evsys` file logging. User code (custom
`transforms`, custom `build_batch`/algorithms, callbacks) can grab the *same* logger and
write into its own named folder.

## Layout (per run)

```
{run_dir}/                         # .evsys/outputs/{exp-slug}/{run}/  (one folder per run/stage)
  summary.md                       # hypothesis, status, validation headline
  01_data/                         # data AFTER transforms + the exact chat template going in
    data.md, after_transform.jsonl, chat_template.md
  02_training_rollouts/            # training rollout predictions (text) + per-token logprobs
    step_{n}.md                    #   + reward & advantage per sample per trajectory group
  03_validation_rollouts/          # validation rollout predictions (per eval)
    {bench}_step_{n}.md
  04_training_metrics/metrics.csv  # training metrics (whitelisted scalars)
  05_validation_metrics/           # validation metrics, one block per eval (every run-time)
    metrics.csv, metrics.md
  harbor/{train,val,sdft,eval}/    # harbor's OWN full rollout store — referenced, not copied
  <your-folder>/                   # anything user code logs via run_log.note()/record()/dir()
```

`harbor/` holds harbor's native per-trial `result.json` (token ids, reward, usage) — the
SDK never copies it; the numbered folders hold the clean decoded view.

## How it's wired (one logger, the same local logging)

`RunLog` (`src/evsys_sdk/run_log.py`) is created by the runner per run and set as the
current run log (a contextvar). It's reached three ways:

- **runner / SDK**: creates it, logs `01_data` (after transforms), renders `04`/`05`
  metrics + `summary.md` at the end.
- **algorithms / custom `build_batch`**: `ctx.extras["run_log"]`.
- **callbacks**: `state.run_log` (added to `LoopState`).
- **anywhere during a run (e.g. inside a `transforms` `__call__`)**:
  `from evsys_sdk import get_run_log`.

User-facing API for custom folders:

```python
log = get_run_log()
if log:
    log.note("my_transform", "dropped 3 rows missing tool_slug", title="cleanup")
    log.record("my_transform", {"dropped": 3})          # appends jsonl
    path = log.dir("scratch")                            # a folder to write whatever
```

SDK producers:
- `runner` → `01_data` (post-transform rows + meta).
- `sft` → `01_data/chat_template.md` (exact `apply_chat_template` text, not ids).
- `rl` / `sdft` → `02_training_rollouts/step_{n}.md` (predictions, per-token logprobs,
  reward+advantage per sample/group), on a bounded cadence (step 0, every 10). Harbor jobs
  routed to `harbor/train|sdft`.
- in-loop evaluators (`base`→`evaluators`) → `03_validation_rollouts` + `05` per eval; harbor
  jobs to `harbor/val`.
- `experiment._eval_arm_harbor` → `03`/`05` for post-training eval; harbor jobs to `harbor/eval`.

All `RunLog` writes are best-effort (never raise into training); every wiring point is
guarded on `ctx.extras.get("run_log")` / `state.run_log`, so absence is a no-op.

## Content kept (clean, no overload)

| Folder | What |
|---|---|
| 01_data | rows count + meta + transforms (in order), first ~5 rows, exact chat template for ~3 examples |
| 02_training_rollouts | per logged step: a few groups; per sample reward + advantage + decoded prediction + first ~80 tokens' logprobs |
| 03_validation_rollouts | per eval: a few groups' decoded predictions + rewards |
| 04_training_metrics | whitelisted scalars → CSV (`loss`, `nll`, `reward/mean`, `lr`, `kl`, `advantage`, …) |
| 05_validation_metrics | `val/*` scalars → CSV (accumulates across evals) + a markdown block per eval |

## Verification

- `tests/test_run_log.py` (11) — folders, chat-template-as-text, reward/advantage/per-token,
  metrics split (train vs val), custom-folder API, `get_run_log` contextvar. Harbor-free.
- `tests/test_runner_mock.py` — a real `run_experiment` (mock backend) produces the per-run
  folders + `summary.md` + `metrics.csv`.
- Smoke: a 2-stage **continual** run (mock_sft) produces both stages' folders, and a custom
  transform logs into its own `my_transform/` folder via `get_run_log()`.

The `rl`/`sdft`/`base`/`evaluators`/`sft`/`experiment` wiring runs only under `tinker`+`harbor`
(not installed in this worktree's venv); it is guarded and unit-tested at the `RunLog` level
but not executed end-to-end here. To run it: `uv sync --extra tinker` + `TINKER_API_KEY`.
```

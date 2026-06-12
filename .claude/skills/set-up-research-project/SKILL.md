---
name: Set up research project
description: Scaffold or migrate a repo to the evsys-sdk research-project layout (data/, src/, experiments/, .evsys/). Use when starting a new project that will use the SDK, or when the user has an existing ad-hoc evsys-sdk project (loose `training/` scripts, scattered data files) they want to bring into the standard shape.
---

# Set up research project

This skill stands up a new repo against the evsys-sdk research-project
layout, or migrates an existing one into it. Use it when:

  * The user is starting a new project that will use `evsys-sdk`.
  * The user has an existing ad-hoc project (e.g. `composio-bench`-style:
    loose `training/`, scattered `data/`, no `experiments/` or `src/`)
    and wants to bring it into the standard shape.

The target layout is the one documented in
`evsys-sdk/docs/DESIGN.md` → "Researcher-project layout":

```
<project>/
├── pyproject.toml                  # declares src/ as importable
├── README.md
├── data/
│   ├── raw/                        # untouched source dumps (gitignored)
│   ├── fetch/                      # python scripts that populate raw/
│   ├── process/                    # raw → datasets/<name>/v<N>/
│   ├── datasets/<name>/v1/{train,test}.jsonl + metadata.yaml
│   └── benchmark/<name>/tasks.jsonl + metadata.yaml [+ images/ + raw/]
├── src/                        # project-specific SDK extensions
│   ├── __init__.py                 # imports verifiers/metrics/transforms
│   ├── verifiers.py
│   ├── metrics.py
│   └── transforms.py
├── experiments/<yyyymmdd>_<slug>/
│   ├── config.yaml                 # ExperimentConfig
│   └── run.py                      # `Experiment.from_yaml("config.yaml").run()`
└── .evsys/                    # gitignored runtime mirror + outputs
```

## Decide: new or existing?

First, classify the working dir.

```bash
ls -A | head -20
```

  * **New / empty** repo (no `data/`, no `training/`, no `experiments/`,
    only `.git/`, `README.md`, `pyproject.toml`, etc.) → **Bootstrap path**.
  * **Existing project with training scripts or data** (you see anything
    like `training/run_*.py`, `training/trajectory_ext/`, loose
    `data/*.json`, `evals/`, `results/`) → **Migration path**.

If unsure, ask the user. Do not make destructive changes until you've
confirmed which path.

## Bootstrap path (new / empty repo)

1. Confirm the project name with the user (default to the directory basename).
2. Run:
   ```bash
   evsys init-project . --name <project_name>
   # or, if scaffolding into a separate dir:
   evsys init-project <path> --name <project_name>
   ```
   The CLI refuses non-empty dirs unless you pass `--force`. `--force` only
   fills in missing files — it never overwrites a file the user already wrote.
3. Create a `uv`-managed `.venv` and install the project into it:
   ```bash
   uv venv                    # creates .venv/ using requires-python from pyproject.toml
   uv pip install -e .        # installs the project (pulls in evsys-sdk) editable
   ```
   `uv venv` is idempotent — if `.venv/` already exists it leaves it alone, so
   it's safe to run on a repo that's already set up. `evsys init-project`
   already gitignores `.venv/`, so nothing to add there.
   Tell the user to activate it with `source .venv/bin/activate` (or just prefix
   commands with `uv run`, e.g. `uv run evsys new-experiment ...`).
4. Walk the user through what landed:
   * `data/` lineage convention (raw → fetch → process → datasets/<name>/v<N>/)
   * `src/{verifiers,metrics,transforms}.py` with commented examples
   * `pyproject.toml` declaring `src` as the importable package
   * `.gitignore` (`.evsys/`, `data/raw/`, `.venv/`)
5. Show how to add the first experiment:
   ```bash
   evsys new-experiment first_check
   ```
   Tell the user to edit the generated `config.yaml`, then
   `python experiments/<dir>/run.py` (or `uv run python experiments/<dir>/run.py`).
6. Point them at:
   * `evsys-sdk/docs/DESIGN.md` — layout rationale.
   * `using-evsys-sdk` skill — how `Experiment.from_yaml(...).run()` works
     end-to-end (dashboard records, sweep expansion, eval, conclusion).

## Migration path (existing project)

Aim for a small, reviewable migration. **Never delete user files without
confirmation.** Use `git mv` for everything you can so history is preserved.

### Step 1: map current → target

Survey the repo and propose mappings — present the full list to the user
before touching anything.

| Current shape | Target |
|---|---|
| `training/run_*.py` (one-off sweep scripts) | `experiments/<yyyymmdd>_<slug>/{config.yaml,run.py}` per script — extract the inline hypothesis + hyperparameters into `config.yaml`, replace the per-arm Python loop with a `matrix:` block, leave the OOP entrypoint in `run.py`. |
| `training/trajectory_ext/verifiers.py` | `src/verifiers.py` |
| `training/trajectory_ext/transforms.py` | `src/transforms.py` |
| `training/trajectory_ext/__init__.py` | merge into `src/__init__.py` |
| `training/backfill_step_metrics.py` | **delete** — `Experiment` auto-forwards step metrics |
| `training/_chunked_generate` / `_score` helpers | folded into `Benchmark.score()` — drop |
| `training/eval_*.py` (standalone eval scripts) | replace with `metadata.benchmark` block in `config.yaml` + the SDK's `Benchmark` |
| `data/eval_queries_v2.json` (eval set) | `data/benchmark/composio_eval_v2/tasks.jsonl` (harbor-format JSONL — one task per line) + `metadata.yaml`; convert via a small `data/process/<name>_to_harbor.py` script |
| `data/sft_overdose_v17_think.jsonl` (training data) | `data/datasets/sft_overdose/v17/train.jsonl` + `metadata.yaml` (source, parent version, row count) |
| `output/`, `checkpoints/`, scattered log dirs | `.evsys/` (gitignored) |
| `analysis/`, `notebooks/` | leave in place; not part of the layout |

### Step 2: scaffold the target dirs

Run `evsys init-project . --force` to fill in any missing standard files
without clobbering existing ones. Then create the data subdirs that didn't
exist yet (`data/datasets/<name>/v1/`, `data/benchmark/<name>/`).

If the project has no `uv`-managed environment yet, create one and install it
editable — `uv venv && uv pip install -e .`. `uv venv` won't disturb an
existing `.venv/`, and `uv pip install -e .` reconciles the project's existing
`pyproject.toml` dependencies (which the migration preserves). Make sure
`.venv/` is gitignored.

### Step 3: move files

For each mapping, propose the `git mv` (or write a small conversion script
when the target shape differs from the source). Pause for user confirmation
on anything that:
  * deletes a file (e.g. `backfill_step_metrics.py`),
  * converts a JSON eval set into harbor JSONL (verify a couple of rows
    round-trip correctly first),
  * touches `pyproject.toml` (existing dependencies must be preserved).

### Step 4: rewrite imports

After moves:
  * Files that imported `trajectory_ext.verifiers` → import from `src`.
  * Scripts that called `backfill_run` → delete the call (Experiment does it).
  * `from evsys_sdk import …` imports for the OOP path
    (`Experiment`, `Sweep`, `Benchmark`) are now top-level.

### Step 5: rebuild one experiment as a smoke test

Pick one prior training script and port it end-to-end to the new layout:
  1. `evsys new-experiment <slug_matching_old_script>`.
  2. Translate the script's hypothesis / hyperparameters / sweep axis into
     the new `config.yaml`'s `metadata` + `matrix:` blocks.
  3. Reduce `run.py` to `Experiment.from_yaml("config.yaml").run()`.
  4. Upload any benchmark it referenced:
     `evsys benchmark upload data/benchmark/<name>` — paste the printed id
     into `config.yaml`'s `metadata.benchmark.id`.
  5. Run it with the mock backend first (set `backend.kind: mock`) to
     confirm the wiring works without spending compute.

Only after that smoke succeeds do you propose porting the remaining
scripts — one at a time, each as its own PR if possible.

## Hard rules

  * **Never `git rm`** or `rm` a file without explicit user confirmation.
  * **Never overwrite** a user-authored file (`pyproject.toml`, `README.md`,
    config files) — scaffold around them.
  * **Preserve git history** — prefer `git mv` over plain `mv`.
  * **One migration per PR** if possible — don't bundle "port script A" and
    "convert benchmark B" and "delete utility C" into one mass move.
  * **Mock first, real backend second** — port one experiment with
    `backend.kind: mock`, get a passing smoke run, then enable the real
    backend.

## What to point the user at after you're done

  * `evsys new-experiment <slug>` for every subsequent experiment.
  * `evsys benchmark upload data/benchmark/<name>` whenever a benchmark
    (the TEST set, scored after training) changes content (idempotent
    re-upload returns "unchanged").
  * `evsys validation upload data/validation/<name>` for an in-loop
    VALIDATION set — scored every N steps during training to drive model
    selection. Paste the printed id into the run's `validation.dataset_id`
    and set `validation.eval_for_every` + `validation.metrics` (metrics.py
    kinds). Benchmark = final test; validation = model selection. Keep them
    separate so selection never keys off the test set.
  * `using-evsys-sdk` skill for the day-to-day patterns
    (`Experiment.from_yaml(...).run()`, sweep / matrix syntax, scoring).
  * `getting-experiment-context` skill if they want to recall prior results
    before designing a new experiment.

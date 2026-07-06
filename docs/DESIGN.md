# evsys-sdk — design notes

## Goals

1. **Single declarative YAML** drives a full experiment (data → train → eval),
   so an evolutionary algorithm can mutate it without writing Python.
2. **Modular** — adding a new algorithm/verifier/metric is a decorator + a
   Pydantic Config class. No library fork.
3. **Backend-pluggable** — same YAML can run locally on TRL or remotely on
   Tinker; backends are interchangeable.
4. **Decoupled storage** — the library imports zero Supabase code by default;
   Supabase is an optional adapter.

## Why protocols, not ABCs

PEP 544 protocols mean any class with the right methods satisfies the contract
— no inheritance from us. This is critical for third-party extensions: if you
have to subclass `evsys_sdk.algorithms.BaseAlgorithm`, you've
imported the world. With protocols, your `MyDPO` class is just plain Python.

## Why a registry per kind

Each extension point has its own registry (`_algorithms`, `_verifiers`, …).
Reasons:

* The YAML loader knows which registry to look up `kind:` in based on the
  surrounding context. No string-prefix tricks.
* `schema_for(kind, name)` exposes the per-extension JSON schema, which is
  exactly what an evolution algorithm needs to mutate the YAML safely.
* Per-extension entry-point groups mean external packages declare their
  contributions cleanly.

## YAML schema

Strict Pydantic v2: every field has a type, `extra='forbid'` everywhere, no
defaults that hide misspellings. The `kind:` discriminator selects which
registered class's `Config` validates the corresponding `params:` block.

The `matrix:` shorthand is a convenience that expands at load-time into
`runs:` — the result is the same `runs[]` shape, so the runner doesn't care.

## Lifecycle of a run

1. `run_experiment(cfg)` → for each `RunConfig`:
   1. Build `data_store`, `log_store` from top-level specs.
   2. Read raw rows via `data_store`.
   3. Apply `data.transforms[]` in order.
   4. Build `backend`, call `backend.prepare(model=..., run_dir=...)` → handles dict.
   5. Build `algorithm` with `params`.
   6. Construct `RunContext` carrying `data_store`, `log_store`, `backend`, `extras`.
   7. `algorithm.train(ctx)` → `RunResult`.
   8. `backend.teardown(handles)`.
   9. Best-effort eval (skips on failure — eval errors don't fail the run).
   10. Persist `run_result.json`.

## Researcher-project layout

Every project the deciding-experiments skill bootstraps follows the same shape so
scripts, benchmarks, and extensions land in predictable places. Scaffold a new
project with ``evsys init-project <name>``; the tree is:

```
<project>/
├── pyproject.toml                  # declares src/ as an importable pkg
├── README.md
├── data/
│   ├── raw/                        # untouched source dumps (gitignored)
│   ├── fetch/                      # Python scripts that populate raw/
│   ├── process/                    # raw → datasets/<name>/v<N>/
│   ├── datasets/                   # versioned train/test JSONL
│   │   └── <name>/v1/{train,test}.jsonl + metadata.yaml
│   ├── benchmark/                  # harbor-format TEST suites (final goal)
│   │   └── <name>/tasks.jsonl + metadata.yaml [+ images/ + raw/]
│   └── validation/                 # harbor-format VALIDATION sets (in-loop)
│       └── <name>/tasks.jsonl + metadata.yaml
├── src/                        # project-specific SDK extensions
│   ├── __init__.py                 # imports verifiers/metrics/transforms
│   ├── verifiers.py                # @register_verifier(_fn) classes/fns
│   ├── metrics.py                  # @register_metric
│   └── transforms.py               # @register_transform
├── experiments/
│   └── <yyyymmdd>_<slug>/          # `evsys new-experiment <slug>`
│       ├── config.yaml             # ExperimentConfig — model, data, sweep, metadata
│       └── run.py                  # Experiment.from_yaml("config.yaml").run()
└── .evsys/                    # local mirror + checkpoints + log_store output
```

Each ``config.yaml`` is **self-contained** — there is no project-root yaml that
experiments inherit from. The experiment-level fields (hypothesis, tags,
success_metric, benchmark) live under ``metadata:`` and are read by
``Experiment.run()``:

```yaml
metadata:
  hypothesis: "Higher LoRA rank improves pass@1"
  tags: [sft, qwen3_4b]
  success_metric: pass_rate
  benchmark:                        # TEST set — scored once, after training
    id: <dashboard benchmark id from `evsys benchmark upload`>   # preferred
    # name: composio_eval_v2     # alt: resolves to the latest version's id
    # path: data/benchmark/composio_eval_v2   # offline / dev fallback
    breakdown_keys: [toolkit]
```

### Referencing data by id / name (preferred over local paths)

Stored experiment scripts should reference data by **dashboard id** (or
**name**, which resolves to the latest version's id) rather than a local path.
The SDK pulls the rows once into the local `.evsys/` workspace cache and
trains/scores from there, so a committed script is portable and doesn't depend
on anyone's local file layout. `path` stays as an offline / dev fallback.

```yaml
run:
  data:
    dataset_id: <dashboard dataset id>     # or: dataset_name: sft_overdose
    transforms: [...]                       # applied to the pulled rows
    # source_kind/path are ignored when dataset_id/name is set
```

Same for the benchmark block above (`id` / `name` / `path`). Under the hood
both go through `Workspace.pull_dataset` / `pull_benchmark`, which cache to
`.evsys/<datasets|benchmarks>/<id>.jsonl` (version-immutable, so a given
id never changes).

### Benchmark (test) vs. validation (in-loop)

Both test and validation are entries in the same ``metadata.benchmark`` list;
``tags`` declare the role and ``run_every`` declares the cadence.

A **test benchmark** is the *final goal*: scored once after training (no
``run_every``), tagged ``[test]``. Model selection must never key off it.

A **validation benchmark** is scored *during* training to drive model
selection — same harbor format, tagged ``[val]`` with a positive
``run_every``:

```yaml
metadata:
  benchmark:
    - name: val_set                 # in-loop — scored every N steps
      id: <benchmark id>            # or path: data/benchmark/<name>
      tags: [val]
      run_every: 50                 # score every 50 training steps off the live model
      engine: harbor
    - name: full_test               # post-training only
      id: <benchmark id>
      tags: [test]
      engine: harbor
```

Every ``run_every`` steps the training loop scores the live model on the
``[val]`` entry and records ``val/<name>/<metric>`` curves under
``split="val"`` — separate from the post-training ``[test]`` eval.

## OOP entry points

The high-level path is one class with declarative inputs:

```python
from evsys_sdk import Experiment
import src   # registers project verifiers / metrics / transforms

Experiment.from_yaml("config.yaml").run()
```

``Experiment`` owns:
  * creating the dashboard experiment + per-arm run records;
  * expanding ``matrix`` / ``Sweep`` into one ``RunConfig`` per arm;
  * per-arm failure isolation (one arm raising doesn't kill the sweep);
  * post-train benchmark scoring via ``Benchmark.score(client)``;
  * auto-forwarding the local ``metrics.jsonl`` to the store
    (no manual ``backfill_step_metrics`` call);
  * aggregating ``best_score`` + ``conclusion`` and finalizing the experiment.

The legacy ``run_experiment(cfg)`` is still the inner runner that
``Experiment`` calls per arm — bypass ``Experiment`` only when you need to do
training without dashboard bookkeeping.

## What's NOT in v0.1

* Supabase adapters (planned: `evsys_sdk.adapters.supabase`).
* Evolutionary loop (kept in `backend/api/experiments/loop.py` for now).
* Distributed launchers (Modal, Slurm).
* Streaming / checkpoint resumption beyond what tinker_cookbook provides.

These all extend the same protocol surface and should arrive incrementally
without breaking changes to the public API.

## Backwards compatibility

* `version: 1` in the YAML root is currently advisory; bumped on schema breaks.
* Public symbols re-exported from `evsys_sdk/__init__.py` are the
  stable surface. Anything else may move.

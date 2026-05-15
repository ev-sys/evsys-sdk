# trajectory-experiments — design notes

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
have to subclass `trajectory_experiments.algorithms.BaseAlgorithm`, you've
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

## What's NOT in v0.1

* Supabase adapters (planned: `trajectory_experiments.adapters.supabase`).
* Evolutionary loop (kept in `backend/api/experiments/loop.py` for now).
* Distributed launchers (Modal, Slurm).
* Streaming / checkpoint resumption beyond what tinker_cookbook provides.

These all extend the same protocol surface and should arrive incrementally
without breaking changes to the public API.

## Backwards compatibility

* `version: 1` in the YAML root is currently advisory; bumped on schema breaks.
* Public symbols re-exported from `trajectory_experiments/__init__.py` are the
  stable surface. Anything else may move.

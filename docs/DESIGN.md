# trajectory-labs — design notes

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
have to subclass `trajectory_labs.algorithms.BaseAlgorithm`, you've
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

Every project the training-decider agent bootstraps follows the same shape so
scripts, benchmarks, and extensions land in predictable places. Scaffold a new
project with ``trajex init-project <name>``; the tree is:

```
<project>/
├── pyproject.toml                  # declares scripts/ as an importable pkg
├── README.md
├── data/
│   ├── raw/                        # untouched source dumps (gitignored)
│   ├── fetch/                      # Python scripts that populate raw/
│   ├── process/                    # raw → datasets/<name>/v<N>/
│   ├── datasets/                   # versioned train/test JSONL
│   │   └── <name>/v1/{train,test}.jsonl + metadata.yaml
│   └── benchmark/                  # harbor-format eval suites
│       └── <name>/tasks.jsonl + metadata.yaml [+ images/ + raw/]
├── scripts/                        # project-specific SDK extensions
│   ├── __init__.py                 # imports verifiers/metrics/transforms
│   ├── verifiers.py                # @register_verifier(_fn) classes/fns
│   ├── metrics.py                  # @register_metric
│   └── transforms.py               # @register_transform
├── experiments/
│   └── <yyyymmdd>_<slug>/          # `trajex new-experiment <slug>`
│       ├── config.yaml             # ExperimentConfig — model, data, sweep, metadata
│       └── run.py                  # Experiment.from_yaml("config.yaml").run()
└── .trajectory/                    # local mirror + checkpoints + log_store output
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
  benchmark:
    path: data/benchmark/composio_eval_v2
    id: <dashboard benchmark id from `trajex benchmark upload`>
    breakdown_keys: [toolkit]
```

## OOP entry points

The high-level path is one class with declarative inputs:

```python
from trajectory_labs import Experiment
import scripts   # registers project verifiers / metrics / transforms

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

## Image / multimodal SFT

`tinker_sft` builds every training example through the model's **renderer**
(`tinker_cookbook.renderers`) — the same code path tinker uses at inference, so
train and serve stay in distribution. There is no hand-rolled tokenization or
loss masking; the renderer returns the `ModelInput` (text **and** image chunks)
plus per-token weights, and we wrap it as a `Datum`.

Because of that, **image SFT needs no new config** — just:

* `model.renderer_name`: a vision renderer (e.g. `qwen3_vl`, `qwen3_vl_instruct`).
  `renderer_name` is required for `tinker_sft` (text renderers like `qwen3` too).
* image-bearing training rows: message `content` as a list of blocks
  (OpenAI `image_url` / Anthropic `image`, via `image_url_block` /
  `image_base64_block`).

The runner **auto-detects** image rows (`messages_have_images`) and, only then,
loads an `AutoImageProcessor` for the model and flattens the image blocks
(`normalize_message_images`) into the renderer's part shape. Pure-text runs are
unaffected. RL rollouts and image-based *eval/benchmark* scoring are not wired
yet (text-only `generate`) — follow-ups that can reuse the same helpers.

## What's NOT in v0.1

* Supabase adapters (planned: `trajectory_labs.adapters.supabase`).
* Evolutionary loop (kept in `backend/api/experiments/loop.py` for now).
* Distributed launchers (Modal, Slurm).
* Streaming / checkpoint resumption beyond what tinker_cookbook provides.

These all extend the same protocol surface and should arrive incrementally
without breaking changes to the public API.

## Backwards compatibility

* `version: 1` in the YAML root is currently advisory; bumped on schema breaks.
* Public symbols re-exported from `trajectory_labs/__init__.py` are the
  stable surface. Anything else may move.

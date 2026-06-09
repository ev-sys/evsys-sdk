# evsys-sdk — SDK Reference

> Package: `evsys_sdk` · Distribution: `evsys-sdk` · Version: `0.1.0` · Python ≥ 3.12 · License: MIT
> CLI entry point: `evsys`

A declarative, modular framework for LLM training experiments (SFT, RL,
distillation, prompt tuning). A single YAML file describes a full experiment —
**data → train → eval** — and every moving part (algorithm, backend, metric,
store, …) is a pluggable extension registered by a decorator. The same YAML
runs on a mock backend (tests), locally on TRL/PEFT, or remotely on Tinker.

The design target is an **evolutionary optimization loop** that mutates the YAML
without writing Python: each extension exposes a Pydantic `Config` whose JSON
schema defines the legal mutation space.

---

## Table of contents

1. [Install](#1-install)
2. [Quickstart](#2-quickstart)
3. [Core concepts](#3-core-concepts)
4. [The YAML schema (config models)](#4-the-yaml-schema-config-models)
5. [Matrix campaigns](#5-matrix-campaigns)
6. [The run lifecycle](#6-the-run-lifecycle)
7. [Extension points & protocols](#7-extension-points--protocols)
8. [Built-in extensions](#8-built-in-extensions)
9. [Data shapes (Harbor interchange types)](#9-data-shapes-harbor-interchange-types)
10. [The registry & writing extensions](#10-the-registry--writing-extensions)
11. [CLI reference (`evsys`)](#11-cli-reference-evsys)
12. [Eval harness](#12-eval-harness)
13. [Dashboard client](#13-dashboard-client)
14. [Public API surface](#14-public-api-surface)
15. [Relationship to the backend repo](#15-relationship-to-the-backend-repo)

---

## 1. Install

```bash
cd evsys-sdk
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[tinker,local,tensorboard]"
```

Optional dependency extras (`pyproject.toml`):

| Extra | Pulls in | Use |
|---|---|---|
| `tinker` | `tinker`, `tinker-cookbook`, `chz` | Hosted Tinker training/inference |
| `local` | `torch`, `transformers`, `trl`, `peft`, `datasets`, `accelerate` | Local GPU training |
| `tensorboard` | `tensorboard` | `TensorBoardLogStore` |
| `supabase` | `requests` | Supabase adapters (planned) |
| `plot` | `matplotlib` | Plotting |
| `claude` / `gemini` / `openai` | the respective SDK | Hosted inference clients (judges/baselines) |
| `frontier` | all three frontier SDKs | All hosted inference clients |
| `gepa` | `gepa` | GEPA prompt-tuning algorithm |
| `dev` | `pytest`, `pytest-asyncio`, `ruff` | Development |

Core dependencies are minimal: `pydantic>=2.10`, `pyyaml`, `typing-extensions`,
`requests`. The core imports **zero** training/Supabase code — heavy deps load
lazily only when the relevant extension actually runs.

For real Tinker runs: `export TINKER_API_KEY=...`.

---

## 2. Quickstart

**From YAML (canonical interface):**

```bash
evsys validate config.yaml --deep   # structure + each kind/params block
evsys run config.yaml                # execute, writes outputs/<run>/run_result.json
evsys list                           # everything in the registries
evsys schema algorithm tinker_sft    # JSON schema for one extension's params
```

**From Python:**

```python
from evsys_sdk import run_experiment, load_yaml

cfg = load_yaml("config.yaml")        # -> ExperimentConfig (matrix expanded)
results = run_experiment(cfg)          # -> list[RunResult]
for r in results:
    print(r.run_id, r.status, r.metrics)
```

A minimal config:

```yaml
version: 1
name: hello_sft
output_dir: ./outputs
data_store: { kind: local }
log_store:  { kind: jsonl }
run:
  name: r0
  data:
    source_kind: jsonl
    path: data/train.jsonl
    transforms:
      - { kind: jsonl_to_chat, params: {} }
  model: { name: Qwen/Qwen3-4B, renderer_name: qwen3 }
  algorithm: { kind: tinker_sft, params: { learning_rate: 1e-5, num_epochs: 1 } }
  backend: { kind: tinker }
  eval:
    enabled: true
    metrics: [ { kind: exact_match } ]
    inference: { kind: tinker }
```

---

## 3. Core concepts

```
ExperimentConfig (the YAML root)
├─ data_store : DataStoreSpec  ──► DataStore   (local / in_memory)
├─ log_store  : LogStoreSpec   ──► LogStore    (jsonl / tensorboard / multiplex)
└─ run | runs | matrix
   └─ RunConfig
      ├─ data      : DataConfig    (source_kind + path/rows/hf + transforms[])
      │                 └─ TransformSpec[]  ──► Transform
      ├─ model     : ModelConfig   (name, load_checkpoint_path, renderer_name)
      ├─ backend   : BackendConfig ──► Backend  (mock / local / tinker)
      ├─ algorithm : AlgorithmConfig ──► Algorithm (tinker_sft / local_rl / ...)
      └─ eval      : EvalConfig
         ├─ inference : InferenceSpec ──► InferenceClient
         └─ metrics   : MetricSpec[]   ──► Metric
```

Three ideas hold it together:

- **`kind` + `params` envelopes.** Every pluggable block in the YAML is just a
  string `kind` (a registry key) plus a free-form `params` dict. The runner
  looks `kind` up in the matching registry and validates `params` against the
  registered class's `.Config` Pydantic model. This indirection is what lets an
  evolution loop mutate YAML safely without importing typed classes.
- **Protocols, not base classes.** Extensions satisfy a `typing.Protocol`
  (PEP 544) — any class with the right methods works; no subclassing the
  library, so a third-party extension never has to "import the world."
- **One registry per extension point**, each with its own decorator and Python
  entry-point group, so external packages auto-extend the registries on import.

---

## 4. The YAML schema (config models)

Defined in `config.py`. All models inherit `_Strict` → `extra="forbid"`, so a
typo'd key is rejected loudly (critical when an algorithm is mutating the YAML).

### `ExperimentConfig` — the root

| Field | Type | Default | Notes |
|---|---|---|---|
| `version` | `int` | `1` | Schema version; advisory, bumped on breaking changes |
| `name` | `str` | — | Required |
| `description` | `str` | `""` | |
| `output_dir` | `str` | `"./outputs"` | Where local artifacts/logs are written |
| `data_store` | `DataStoreSpec` | `{kind: local}` | |
| `log_store` | `LogStoreSpec` | `{kind: jsonl}` | |
| `run` | `RunConfig?` | `None` | Exactly **one** of run/runs/matrix |
| `runs` | `list[RunConfig]?` | `None` | A campaign |
| `matrix` | `MatrixSpec?` | `None` | Expands into `runs` at load time |
| `parent_experiment_id` | `str?` | `None` | Evolutionary lineage |
| `metadata` | `dict` | `{}` | Free-form (budget, hypothesis, client tag) |

`model_post_init` enforces that **exactly one** of `run` / `runs` / `matrix` is set.

### `RunConfig` — one training run

| Field | Type | Default |
|---|---|---|
| `name` | `str` | — (unique within experiment) |
| `data` | `DataConfig` | — |
| `model` | `ModelConfig` | — |
| `algorithm` | `AlgorithmConfig` | — |
| `backend` | `BackendConfig` | `{kind: tinker}` |
| `eval` | `EvalConfig` | enabled, no metrics |
| `seed` | `int` | `42` |
| `tags` | `list[str]` | `[]` |

### `DataConfig`

| Field | Type | Default | Notes |
|---|---|---|---|
| `source_kind` | `"jsonl" \| "json" \| "in_memory" \| "hf_dataset"` | `"jsonl"` | Which built-in loader |
| `path` | `str?` | `None` | For jsonl/json (store-relative or absolute) |
| `rows` | `list[dict]?` | `None` | For in_memory |
| `hf_dataset` | `str?` | `None` | HuggingFace dataset id |
| `hf_split` | `str` | `"train"` | |
| `transforms` | `list[TransformSpec]` | `[]` | Applied in order to raw rows |

### `ModelConfig`

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | HF id, e.g. `Qwen/Qwen3-4B` |
| `load_checkpoint_path` | `str?` | Resume from Tinker/local checkpoint |
| `renderer_name` | `str?` | Tinker chat renderer hint, e.g. `qwen3` |

### `BackendConfig`

| Field | Type | Default |
|---|---|---|
| `kind` | `"mock" \| "local" \| "tinker"` | `"tinker"` |
| `params` | `dict` | `{}` |

### `EvalConfig`

| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | `bool` | `True` | |
| `metrics` | `list[MetricSpec]` | `[]` | Eval is skipped if empty |
| `inference` | `InferenceSpec?` | `None` | How to query the trained model |
| `eval_data` | `DataConfig?` | `None` | Falls back to training rows |
| `n_samples` | `int?` | `None` | Cap on eval examples |

### Generic `kind`+`params` specs

`AlgorithmConfig`, `VerifierSpec`, `MetricSpec`, `TransformSpec`,
`InferenceSpec`, `DataStoreSpec`, `LogStoreSpec` are all the same shape:
`{ kind: str, params: dict }`. `DataStoreSpec` defaults `kind="local"`,
`LogStoreSpec` defaults `kind="jsonl"`.

---

## 5. Matrix campaigns

`MatrixSpec` expands at **load time** into a flat `runs[]` (the runner never
sees `matrix`). Cartesian product over axes; each cell deep-copies `base_run`
and overrides the dotted-path fields.

```yaml
matrix:
  base_run:
    name: sweep            # full RunConfig template
    data: { ... }
    model: { name: Qwen/Qwen3-4B }
    algorithm: { kind: tinker_sft, params: { lora_rank: 8, learning_rate: 1e-4 } }
  axes:
    algorithm.params.lora_rank:     [1, 8, 32]
    algorithm.params.learning_rate: [1e-4, 5e-5]
  name_template: "{base}__rank{algorithm.params.lora_rank}__lr{algorithm.params.learning_rate}"
```

- **Axis keys are dotted paths** into the run dict (`_set_dotted`).
- **`name_template`** uses literal `{dotted.key}` placeholders (not
  `str.format`, so dots aren't attribute access) plus `{base}` = base run name.
  Floats are slugged (`1e-4` → `0p0001`-style via `%g`, `+`/`.` stripped).
- Without a template, names become `{base}__{lastkey}{value}__...`.
- The 3×2 example above yields **6 runs**.

---

## 6. The run lifecycle

`run_experiment(cfg_or_path)` (in `runner.py`) accepts an `ExperimentConfig` or
a YAML path, expands matrix, then for **each** `RunConfig` calls `_execute_run`:

1. **Build stores** from the top-level specs. For `jsonl`/`tensorboard` log
   stores (and `multiplex` children), `log_dir` is auto-filled to
   `<run_dir>/logs[/kind]` if not set.
2. **Build backend** from `run.backend`.
3. **Load rows** via `_load_rows` (dispatch on `source_kind`).
4. **Apply transforms** in order (`_apply_transforms`); each transform is called
   as `transform(rows) -> rows`.
5. **Build algorithm** from the registry with `run.algorithm.params`.
6. **`backend.prepare(model=..., run_dir=...)`** → a dict of handles. *If this
   raises, the run is recorded `status="failed"` and returned early.*
7. **Construct `RunContext`** carrying stores, backend, and an `extras` dict
   (`train_rows`, `n_train_rows`, `backend_handles`, `model_name`, `tags`).
8. **`log_store.log_hyperparams(...)`** (experiment/run name, model, backend, tags).
9. **`algorithm.train(ctx) -> RunResult`** inside try/finally; on exception →
   `status="failed"`. `backend.teardown(handles)` always runs in `finally`.
10. **Eval (best-effort)** — only if `status == "completed"`. Any eval exception
    is logged and swallowed; **eval never fails the run**. Eval metrics are
    merged into `result.metrics` under `eval/<metric_kind>` keys and logged.
11. **`log_store.close()`** then **persist** `run_dir/run_result.json`
    (`run_id`, `status`, `metrics`, `artifacts`, `error`, `hparams`, `ts`).

The eval loop (`_run_eval`) builds the inference client from
`eval.inference`, generates for each row (prompt = `row["prompt"]` or the last
message content), extracts `<answer>…</answer>` if present, and compares against
targets (`tool_slug`/`answer` + `toolkit`) using each configured `Metric`.

> Note: validation of `params` against each `.Config` is **lazy** by default
> (happens at instantiation in the runner). `evsys validate --deep` /
> `validate_yaml(path, deep=True)` forces it up front so unknown entry-point
> extensions don't fail prematurely.

---

## 7. Extension points & protocols

Defined in `protocols.py`. Most are `@runtime_checkable` so the registry can
give better errors. Every extension declares `name: ClassVar[str]` (registry
key); most also declare `Config: ClassVar[type]` (a Pydantic model for `params`).

### Run dataclasses (passed to every algorithm)

**`RunContext`** — everything an algorithm needs:
`run_id`, `output_dir`, `config` (the parsed `ExperimentConfig`), `data_store`,
`log_store`, `backend`, `extras: dict` (free bag — e.g. `train_rows`,
`backend_handles`, tinker `training_client`).

**`RunResult`** — what `train` returns:
`run_id`, `status` (`"completed"|"failed"|"cancelled"`),
`metrics: dict[str,float]`, `artifacts: dict[str,str]` (e.g.
`{"final_checkpoint": "s3://…"}`), `error: str?`, `extras: dict`.

**`VerificationResult`** — `reward: float`, `info: dict`.

### Protocol contracts

| Protocol | Required method(s) | Returns |
|---|---|---|
| `Algorithm` | `train(ctx: RunContext)` | `RunResult` |
| `Verifier` | `verify(*, prompt, completion, target)` | `VerificationResult` |
| `Metric` | `compute(*, predictions, targets)` | `float` |
| `DataStore` | `read_jsonl/write_jsonl/read_json/write_json/exists/list` | — |
| `LogStore` | `log_scalar/log_metrics/log_hyperparams/log_artifact/close` | — |
| `Backend` | `prepare(*, model, run_dir) -> dict`, `teardown(handles)` | handles dict |
| `InferenceClient` | `generate(*, prompt, max_tokens, temperature, stop)` | `str` |
| `Transform` | callable `__call__(rows) -> rows` | rows |

The `Algorithm` protocol is intentionally **not** parameterized by `Backend`:
the registry routes a `(recipe.kind, backend.kind)` pair to a concrete algorithm
implementation (e.g. `tinker_sft` vs `local_sft` vs `mock_sft`).

---

## 8. Built-in extensions

All self-register on `import evsys_sdk`.

| Registry | Built-in `kind`s |
|---|---|
| **algorithms** | `tinker_sft`, `tinker_rl`, `local_sft`, `local_rl`, `mock_sft`, `mock_rl`, `combo`, `gepa_prompt` |
| **backends** | `mock`, `local`, `tinker` |
| **metrics** | `exact_match`, `pass_at_k`, `mean_reward`, `toolkit_match` |
| **transforms** | `identity`, `jsonl_to_chat` (write your own via `@register_transform`) |
| **inference** | `mock`, `local`, `tinker`, `claude`, `openai`, `gemini` |
| **verifiers** | `format_only` (write your own via `@register_verifier`) |
| **data_stores** | `local`, `in_memory` |
| **log_stores** | `jsonl`, `tensorboard`, `multiplex` |

Notes:
- **Backends route by `(recipe, backend)`**: pick `mock` for tests, `local` for
  TRL+PEFT on your GPU, `tinker` for hosted training.
- **`combo`** chains phases (output of phase N feeds N+1); **`gepa_prompt`** does
  prompt search with no weight updates (needs the `gepa` extra).
- **`multiplex`** log store fans out to N child stores; the runner auto-fills
  each `jsonl`/`tensorboard` child's `log_dir`.
- Frontier inference clients (`claude`/`openai`/`gemini`) need the matching
  extra and serve as judges or baselines (not trainable).

---

## 9. Data shapes (Harbor interchange types)

`data_types.py` defines the row formats that runners consume and dashboards
render. These are **frozen dataclasses** that mirror the production internal
("Harbor") types, so JSONL round-trips between this SDK and the serving/dashboard
stack with no conversion.

### `TargetFormat` enum

`CHAT_MESSAGES` (SFT) · `HARBOR_TASK` (RL) · `PROMPT_DATASET` (GEPA prompt tuning).

### Row formats

| Dataclass | For | Fields |
|---|---|---|
| `ChatMessagesRow` | SFT | `messages: list[dict]`, `target_assistant: str`, `metadata` |
| `HarborTask` | RL | `task_id`, `instruction`, `verifier: VerifierPayload`, `metadata` |
| `PromptExample` | GEPA | `inputs: dict`, `expected: Any`, `metadata` |

### Verifier specs (data, not executors)

`VerifierPayload = InProcessVerifier | E2BVerifier | LLMJudgeVerifier`,
discriminated by `.kind`:

- **`InProcessVerifier`** (`in_process`) — `fn_name`, `expected`, `params`. A
  registered Python fn called as `fn(completion, expected, **params)`. Sub-ms;
  use for tool-call/exact match.
- **`E2BVerifier`** (`e2b`) — `dockerfile`, `test_sh`, `test_state_py`. Runs the
  model output + tests inside an E2B sandbox; pass/fail from exit code.
- **`LLMJudgeVerifier`** (`llm_judge`) — `judge_model`, `rubric`. Judge scores
  the completion against the rubric.

> These describe the *verification plan*. The runtime `Verifier` **Protocol**
> (in `protocols.py`) is what actually executes verification — deliberately
> separate concepts.

### Multimodal & helpers

- `text_block(text)`, `image_url_block(url, detail=)` (OpenAI shape),
  `image_base64_block(media_type, b64)` (Anthropic shape).
- `block_to_image_src(block)` → renderable URL/data-URL (handles both shapes).
- `has_images(row)` → bool.
- `detect_format(row)` → `"chat_messages"|"harbor_task"|"prompt_dataset"|"unknown"`.
- `from_dict(row)` / `to_dict(obj)` / `iter_jsonl(path)` round-trip mixed-format
  JSONL into typed rows.

---

## 10. The registry & writing extensions

`registry.py` exposes one `Registry` per kind plus `register_*` / `get_*` /
`list_*` helpers. Registering is a single decorator + a Pydantic `Config`.

```python
from typing import ClassVar
from pydantic import BaseModel
from evsys_sdk import register_algorithm, RunContext, RunResult

@register_algorithm("cosine_toy")
class CosineToy:
    name: ClassVar[str] = "cosine_toy"

    class Config(BaseModel):
        amplitude: float = 1.0
        steps: int = 100

    def __init__(self, **params):
        self.cfg = self.Config(**params)   # validates run.algorithm.params

    def train(self, ctx: RunContext) -> RunResult:
        for step in range(self.cfg.steps):
            ctx.log_store.log_scalar("loss", ..., step)
        return RunResult(run_id=ctx.run_id, status="completed",
                         metrics={"final_loss": ...},
                         artifacts={"final_checkpoint": "..."})
```

YAML can now use `algorithm: { kind: cosine_toy, params: { steps: 200 } }`.

**Registry behavior:** duplicate keys raise unless the same class;
`register(name)` also sets `cls.name = name`; `get` raises a `KeyError` listing
available keys. `schema_for(kind, name)` returns the `.Config`'s
`model_json_schema()` — this is exactly what an evolution loop reads to learn the
legal mutation space.

**Same pattern** for every point: `register_verifier`, `register_metric`,
`register_backend`, `register_inference`, `register_transform`,
`register_data_store`, `register_log_store`.

### External packages via entry points

No fork needed — declare in your `pyproject.toml`:

```toml
[project.entry-points."evsys_sdk.algorithms"]
my_dpo = "my_pkg.algorithms:MyDPO"
```

Groups: `evsys_sdk.{algorithms,verifiers,metrics,backends,inference,transforms,data_stores,log_stores}`.
`_entry_points.py` loads every entry point in those groups on import.

---

## 11. CLI reference (`evsys`)

```
evsys validate <path> [--deep]            # parse + (deep) validate kind/params blocks
evsys run <path> [-o/--output FILE]       # run; prints+writes JSON summary
                                           #   exit 0 if all completed, else 2
evsys list [--kind algorithms|backends|…] # enumerate registries
evsys schema <kind> <name>                # JSON schema for one extension's Config
                                           #   kind ∈ algorithm|backend|verifier|metric|
                                           #          transform|data_store|log_store|inference_client
evsys eval model ...                      # evaluate a checkpoint over the eval set
```

`eval model` flags: `--dataset --aliases [--secondary-aliases] --output-dir
--inference-kind {local,tinker,mock} --model-name [--adapter-path
--checkpoint-path --max-tokens --temperature --max-attempts --batch-size
--fail-on-retries]`.

---

## 12. Eval harness

`evsys_sdk.eval` — generic, domain-agnostic eval infra for scoring model
outputs (pass@k + alias matching). Every inference call is wrapped in
`call_with_retry` (exponential backoff); exhausted failures surface via a
`RetryReport` instead of aborting. Project-specific eval harnesses build on this
infra in their own repos.

Public surface:

- **`evaluate_model(...)`** — score an `InferenceClient` (local/tinker/mock checkpoint).
- **`AliasMatcher`** — alias matching (predicted vs verified aliases).
- **`score_rows(...)` / `ModelEvalConfig`** — generic scorer + eval config.
- **`EvalArtifacts` / `EvalSummary`** — result objects; `summary.retry_report`
  carries `total_failures`.
- **`RetryReport` / `RetryFailure` / `call_with_retry`** — the retry layer.
- **`format_summary_markdown(summary, title=)`** — pretty summary.
- Prompt helpers: `qwen_chat_prompt`, `qwen3_chat_template_prompt`,
  `extract_predicted_slug`, `DEFAULT_SYSTEM[_NO_THINK]`, `score_rows`,
  `load_eval_dataset`.

---

## 13. Dashboard client

`DashboardClient` (`dashboard_client.py`) pushes runs to the EvolvingSystems backend
(SDK → Django HTTP → Supabase) **and** always mirrors every write to a local
folder. It is wandb-offline-style robust: if the backend is unreachable
(timeout / connection error / 5xx) it logs a warning and keeps going on the
local mirror; a 4xx raises `DashboardClientError`.

**Auth:** requires an API key + project id by default; missing creds raise
`EvsysAuthError` unless offline mode is on.

**Env vars** (`constants.py`):

| Var | Default | Purpose |
|---|---|---|
| `EVSYS_API_URL` | `http://localhost:8000` | Backend base URL |
| `EVSYS_API_KEY` | — | Issued at dashboard *Settings → API keys* |
| `EVSYS_PROJECT_ID` | — | Shared per-project id |
| `EVSYS_LOG_DIR` | `./evsys_sdk` | Local mirror dir |
| `EVSYS_OFFLINE` | `false` | Local-mirror-only, no auth |
| `EVSYS_LOGGING_LEVEL` | — | SDK log level |

**`ExperimentRun`** is a context-manager wrapper for the common flow:

```python
from evsys_sdk import DashboardClient, ExperimentRun

client = DashboardClient()  # reads env vars
with ExperimentRun(client, experiment_name="sft_run_v9",
                   recipe_kind="sft", run_config={"lr": 1e-5}) as run:
    for step in range(1, 1001):
        run.log_step(step, loss=...)
    run.log_eval(metrics={"pass_at_1": 0.83}, benchmark_id="...")
    run.set_best_score(0.83)
    run.set_conclusion("rank-8 LoRA wins")
```

Lower-level client methods: `create_experiment`, `update_experiment`,
`create_generation`, `update_generation`, `log_step_metric`, `log_eval_run`,
`log_predictions`, `record_benchmark`. `ExperimentRun` also exposes
`log_predictions`, `update_generation/experiment`, and a `benchmark(...)`
context. SDK write routes live under `API_PREFIX = /api/dashboard/api`
(e.g. `/sdk/experiments/`, `/sdk/generations/{id}/step/`).

---

## 14. Public API surface

Everything importable from `evsys_sdk` (the stable surface;
anything not re-exported here may move):

- **Config models:** `ExperimentConfig`, `RunConfig`, `AlgorithmConfig`,
  `DataConfig`, `ModelConfig`, `BackendConfig`, `EvalConfig`, `DataStoreSpec`,
  `LogStoreSpec`, `MetricSpec`, `VerifierSpec`, `TransformSpec`, `InferenceSpec`.
- **Protocols & run types:** `Algorithm`, `Backend`, `DataStore`, `LogStore`,
  `Metric`, `Verifier`, `Transform`, `InferenceClient`, `RunContext`, `RunResult`.
- **YAML:** `load_yaml`, `dump_yaml`, `validate_yaml`.
- **Registry:** `register_*` / `get_*` / `list_*` for all 8 kinds.
- **Runner:** `run_experiment`.
- **Data shapes:** `TargetFormat`, `ChatMessagesRow`, `HarborTask`,
  `PromptExample`, the three verifier specs + `VerifierPayload`, block helpers,
  `detect_format`, `from_dict`, `to_dict`, `iter_jsonl`.
- **Dashboard:** `DashboardClient`, `ExperimentRun`, `DashboardClientError`,
  `EvsysAuthError`.
- **Logging:** `configure_logger`, `get_logger`, `set_level`.

---

## 15. Relationship to the backend repo

The SDK is the **external, declarative front end** to the production training
stack in `backend/api/experiments/`:

- `backend/api/experiments/types.py` is the **source of truth**: a richly typed
  tree of frozen dataclasses (`Experiment`, `GenerationConfig`, `DataPipeline`,
  concrete `SFT/RL/SDFT/Combo/PromptTuning` recipes, `Tinker/MLX/Modal` backends,
  `EvalPlan`, …) with compatibility validation baked into `__post_init__`.
- The SDK **flattens** that into generic `kind`+`params` Pydantic envelopes
  resolved against registries at load time, so an evolution loop can mutate YAML
  without importing the typed classes.
- The SDK's `data_types.py` mirrors **only** the row/verifier shapes
  (`ChatMessagesRow`, `HarborTask`, `PromptExample`, the three verifiers) so
  JSONL round-trips between the two stacks without conversion.
- The SDK's run-result types (`RunContext`, `RunResult`, `VerificationResult`)
  live in `protocols.py` rather than as an `Experiment` campaign object.

### Not in v0.1 (per `docs/DESIGN.md`)

Supabase data/log adapters (`evsys_sdk.adapters.supabase`), the
evolutionary loop port, distributed launchers (Modal/Slurm), and checkpoint
resumption beyond what `tinker_cookbook` provides. All extend the same protocol
surface and are expected to arrive without breaking the public API.
```

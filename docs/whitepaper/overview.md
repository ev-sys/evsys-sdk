# evsys-sdk - architecture overview

A whitepaper-style tour of the SDK's top-level components: the **Experiment**
(the organizing unit), the **Data** surface, the **Algorithm** surface, and the
**registries** that make every piece pluggable. Each section pairs prose with a
mermaid diagram.

---

## 0. The whole system at a glance

```mermaid
flowchart TB
    CFG["ExperimentConfig (YAML)<br/><i>the single canonical artifact</i>"]

    subgraph ORG["① Experiment layer - the organizing unit"]
        direction TB
        E["Experiment.run()"]
        EXP["expand: run / runs / matrix → arms<br/>n_repeats → seeded groups"]
        AR["ArmResult per run"]
        ER["ExperimentResult<br/>best_arm · best_score · conclusion · hypothesis"]
        E --> EXP --> AR --> ER
    end

    subgraph RUN["per-arm RunConfig (one training run)"]
        direction LR
        subgraph DATA["② Data surface"]
            direction TB
            SRC["raw source"] --> WSP["Workspace cache (.evsys/)"] --> TRN["transforms[]"] --> TYP["typed rows<br/>ChatMessagesRow · PromptExample · HarborTask"]
        end
        subgraph ALG["③ Algorithm surface"]
            direction TB
            BK["Backend<br/>mock · local · tinker"] --> AL["Algorithm.train(ctx)"]
            AL --> LOOP["(opt) TrainingLoop + StepBuilder"]
            AL --> RES["RunResult<br/>status · metrics · artifacts"]
        end
        subgraph EVALS["④ Evaluation"]
            direction TB
            BMK["Benchmark (test, once)"]
            VAL["Validation (in-loop)"]
            MET["Metric · Verifier"]
        end
        TYP --> AL
        AL --> EVALS
    end

    subgraph OBS["⑤ Observability & storage"]
        direction LR
        LS["LogStore"] --- DC["DashboardClient (offline-first)"] --- ST["EvsysStore (gateway)"]
    end

    REG["⑥ Registries (8) - kind → class<br/>algorithm · backend · transform · data_store · log_store · metric · verifier · inference"]

    CFG --> E
    EXP --> RUN
    AR --> RES
    AR --> EVALS
    E --> OBS
    REG -. "resolves every 'kind:' in the YAML" .-> RUN

    classDef user fill:#e8f0fe,stroke:#4285f4;
    classDef sdk fill:#f1f3f4,stroke:#9aa0a6;
    class CFG,SRC,TRN,AL,BMK,VAL,MET user;
    class E,EXP,AR,ER,WSP,TYP,BK,LOOP,RES,LS,DC,ST,REG sdk;
```

> Blue = things the **user** authors/implements. Grey = things the **SDK**
> owns. One `ExperimentConfig` ties it all together; every `kind:` in it
> resolves through a registry.

---

## 1. The organizing unit - the Experiment

The Experiment is the scientific container: a **hypothesis**, one or more
**training runs**, and an auto-synthesized **conclusion**.

```mermaid
flowchart TD
    EX["Experiment<br/>metadata: hypothesis · tags · success_metric · benchmark"]
    EX --> M{"run / runs / matrix"}
    M --> A1["arm = RunConfig (a 'cell')"]
    M --> A2["arm = RunConfig"]
    A1 -->|"n_repeats > 1"| G1["group: seeded replicates<br/>seeds [base, base+1, …] · shared group_id"]
    A1 --> R1["ArmResult<br/>metrics · eval_metrics · status"]
    A2 --> R2["ArmResult"]
    G1 --> R1
    R1 & R2 --> PICK["pick best by success_metric"]
    PICK --> ER["ExperimentResult<br/>best_arm · best_score · conclusion"]
```

**Vocabulary**

- **Experiment** - the top-level study. `Experiment.from_yaml(...).run()` owns
  dashboard experiment/run creation, sweep expansion, per-arm failure
  isolation, post-train scoring, metric forwarding, and conclusion building.
- **Training run (arm)** - one `RunConfig` = one concrete training job (one
  cell of a sweep). `runs` / `matrix` produce many arms.
- **Run group** - `n_repeats > 1` replicates an arm across seeds (shared
  `group_id`) so variance is a config field, not a bespoke script.
- **Hypothesis → success_metric → conclusion** - the loop: the hypothesis is
  the question; `success_metric` ranks arms into `best_arm`; the `conclusion`
  summarizes the outcome. All recorded on the dashboard.

**Config objects**

| Object | Role |
|---|---|
| `ExperimentConfig` | top level: `name`, `output_dir`, stores, one of `run`/`runs`/`matrix`, `n_repeats`/`base_seed`, `parent_experiment_id`, `metadata` |
| `RunConfig` | one run: `data`, `model`, `algorithm`, `backend`, `eval`, `validation`, `seed`, `tags` - the cell the two surfaces plug into |
| `MatrixSpec` / `Sweep` | cartesian expansion over dotted-path axes → many `RunConfig`s via one `expand_runs()` |
| `ExperimentResult` / `ArmResult` | outputs: per-arm metrics + the experiment-level `best_arm`, `conclusion`, `hypothesis` |

> Two runners: **`Experiment`** (OOP, with dashboard bookkeeping) wraps
> **`run_experiment(cfg)`** (the inner per-arm runner - use it directly to
> train without bookkeeping).

---

## 2. Data surface - raw → transforms → standardized formats

Standardize *anything* into a few typed shapes, then hand those to the
algorithm. Tokenization/supervision live below this boundary.

```mermaid
flowchart LR
    subgraph SRC["raw source (DataConfig)"]
        S1["dataset_id / dataset_name"]
        S2["jsonl · json · hf_dataset · in_memory"]
    end
    S1 --> WS["Workspace → .evsys/ cache<br/>version-immutable lineage"]
    WS --> RAW["raw rows (dicts)"]
    S2 --> RAW
    RAW --> TF["transforms[] (Transform)<br/>e.g. jsonl_to_chat"]
    TF --> PR["parse_rows(TargetFormat)<br/>strict typed boundary"]
    PR --> CM["ChatMessagesRow"]
    PR --> PE["PromptExample"]
    PR --> HT["HarborTask"]
    CM & PE & HT --> ALGO["→ algorithm (§3)"]

    classDef user fill:#e8f0fe,stroke:#4285f4;
    class S1,S2,TF user;
```

- **User defines:** a `DataConfig` (source + `transforms`), and a custom
  `Transform` when built-ins don't fit. Picks which typed format the algorithm
  consumes.
- **SDK handles:** loading, pull/cache-by-id with lineage (`Workspace`),
  running transforms in order, strict `parse_rows` conversion.

| Class | Implementable? | Contract |
|---|---|---|
| `DataConfig` | author in YAML | source + `transforms[]` |
| **`Transform`** | **yes** (`@register_transform`) | `__call__(rows) -> rows` + `Config` |
| `ChatMessagesRow` / `PromptExample` / `HarborTask` | choose shape | **data only - no supervision encoded** |
| `DataStore` | rarely | `read_jsonl/write_jsonl/read_json/write_json/exists/list` |
| `Workspace` / `MaterializedDataset` | no (SDK) | pull / cache / lineage |

---

## 3. Algorithm surface - Algorithm, Evaluation, Metrics

One required contract (`train(ctx) -> RunResult`) plus an optional gradient
toolkit and pluggable evaluation.

```mermaid
flowchart TD
    BK["Backend.prepare/teardown<br/>mock · local · tinker"] --> CTX["RunContext<br/>train_rows · backend_handles · log_store · output_dir"]
    CTX --> AL["Algorithm.train(ctx) -> RunResult<br/>@register_algorithm"]

    subgraph L2["optional gradient toolkit (Layer 2)"]
        SB["StepBuilder.build_batch(n) -> TrainingBatch"]
        LP["TrainingLoop<br/>fwd_bwd → optim → log → ckpt → eval"]
        LF["loss: named str OR LossCallable"]
        SB --> LP
        LF --> LP
    end

    AL -. "may use" .-> SB
    AL --> RR["RunResult<br/>status · metrics · artifacts"]

    subgraph EV["Evaluation"]
        BMK["Benchmark (test - scored once)"]
        VAL["Validation (in-loop - every N steps)"]
        MET["Metric.compute()"]
        VER["Verifier.verify() / reward"]
    end
    AL --> EV
    BMK --> MET
    VAL --> MET
    EV --> VER

    classDef user fill:#e8f0fe,stroke:#4285f4;
    class AL,SB,LF,MET,VER,BMK,VAL user;
```

### 3.1 Algorithm
- **`Algorithm`** (protocol): `name`, `Config`, `train(ctx) -> RunResult`. The
  only required contract - `train()` may do anything.
- **Optional toolkit:** `TrainingLoop` drives the gradient loop; a `StepBuilder`
  (`build_batch -> TrainingBatch`) is the unit of "a new gradient method";
  losses are a **named string** or a **`LossCallable`** (client-side, via
  `forward_backward_custom`).

### 3.2 Evaluation (two tiers - a test/validation firewall)
- **Benchmark** - the *test set*, scored **once after** training; model
  selection must never key off it.
- **Validation** - scored **in-loop** every N steps to drive selection.
- Both harbor-format; scored via the **Metric** / **Verifier** registries.

### 3.3 Metrics & Verifiers
- **`Metric`**: `compute(predictions, targets) -> float`.
- **`Verifier`**: `verify(prompt, completion, target) -> reward` (RL reward /
  per-task scoring).
- **`InferenceClient`**: `generate(...)` - how eval/RL query a model.

---

## 4. Customizability & main design

The recurring pattern - **implement a protocol → register under a `kind` →
reference it in YAML.** No subclassing the library; no fork.

```mermaid
flowchart LR
    subgraph PROTO["implement a protocol"]
        P1["Transform"]
        P2["Algorithm / StepBuilder + loss"]
        P3["Backend (tinker-compatible)"]
        P4["Verifier / Metric"]
        P5["DataStore / LogStore / InferenceClient"]
    end
    PROTO --> DEC["@register_* decorator<br/>(or entry point: evsys_sdk.<plural>)"]
    DEC --> REG["registry (kind → class)"]
    REG --> YAML["referenced by 'kind:' in ExperimentConfig"]
    YAML --> RUN["runner instantiates + wires"]
```

**Any tinker-compatible backend.** Two backend notions:
- **Framework `Backend`** (`prepare()/teardown()`) - selected by
  `backend.kind`; provisions compute, doesn't train.
- **Training `Backend`** (the loop's executor) - `forward_backward_async`,
  `forward_backward_custom_async`, `optim_step_async`,
  `snapshot_sampling_client`, `get_tokenizer`. **"Tinker-compatible" = implement
  this protocol;** `TrainingLoop` then runs unchanged (a `MockBackend` proves
  it in tests).

| Surface | Implement | Register |
|---|---|---|
| reshape data | `Transform` | `@register_transform` |
| new storage | `DataStore` | `@register_data_store` |
| new method | `Algorithm` (+ `StepBuilder`/loss) | `@register_algorithm` |
| new compute | `Backend` | `@register_backend` |
| reward / scoring | `Verifier` / `Metric` | `@register_verifier` / `@register_metric` |
| generation | `InferenceClient` | `@register_inference` |
| metric sink | `LogStore` | `@register_log_store` |

---

## 5. The registries

One registry per extension point. The YAML `kind` resolves into it, and
`schema_for(kind, name)` exposes the legal `params` (used by evolutionary
search to mutate configs safely).

```mermaid
flowchart TB
    YAML["ExperimentConfig (kind: …)"]
    YAML --> R1["algorithm"]
    YAML --> R2["backend"]
    YAML --> R3["transform"]
    YAML --> R4["data_store"]
    YAML --> R5["log_store"]
    YAML --> R6["metric"]
    YAML --> R7["verifier"]
    YAML --> R8["inference_client"]
```

| Registry | Decorator | `kind` used in |
|---|---|---|
| algorithm | `@register_algorithm` | `run.algorithm.kind` |
| backend | `@register_backend` | `run.backend.kind` |
| transform | `@register_transform` | `data.transforms[].kind` |
| data_store | `@register_data_store` | `data_store.kind` |
| log_store | `@register_log_store` | `log_store.kind` |
| metric | `@register_metric` | `eval.metrics[]` / `validation.metrics[]` |
| verifier | `@register_verifier` | verifier specs / RL reward |
| inference_client | `@register_inference` | `eval.inference.kind` (+ per-backend default factories) |

---

## Summary

The **Experiment** carries a hypothesis, expands into one or more **training
runs** (`RunConfig` arms, optionally seeded into variance **groups**), and
produces a **best_arm** + **conclusion** against a `success_metric`. Each run
plugs together two surfaces: the **data surface** turns raw, lineage-cached
sources through ordered `Transform`s into standardized typed rows that carry
only data; the **algorithm surface** is one contract - `Algorithm.train(ctx) ->
RunResult` - with an optional `TrainingLoop`/`StepBuilder` toolkit for gradient
methods over any tinker-compatible `Backend`, plus pluggable `Evaluation`,
`Metric`, and `Verifier`. Everything is a protocol registered under a `kind`
across eight registries, so users (and third-party packages) extend the system
without forking it.

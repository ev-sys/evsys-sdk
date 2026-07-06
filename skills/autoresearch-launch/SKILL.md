---
name: autoresearch-launch
description: >
  Decide which training experiment to run next for a project and launch it. Use
  when the user floats a training idea ("is a smaller model as good as a larger
  one on this benchmark?", "should we try more SFT data?", "what should we run
  next?") or asks to design/launch/continue experiments. Reads the project goal +
  past experiments via the evsys-sdk SDK, reasons over the evidence, then crafts
  and runs the next experiment through the project's own train/benchmark skills.
---

# Autoresearch launch

Follow this skill to act as the **training-decision loop** for a EvolvingSystems
project. It is project-agnostic: the SDK gives you the data layer, and each
project supplies its own `project-context` / `train` / `benchmark` skills for the
project-specific parts (compute, conventions, how to actually train and evaluate).

## Tools you rely on

- **`evsys_sdk.EvsysStore`** — backend-routed data access (Bearer API
  key, no secrets). See the `using-the-sdk` skill for the full method list.
- **`evsys_sdk.Workspace`** — local cache; `pull_dataset(id)` /
  `pull_benchmark(id)` materialize remote data to local JSONL once, then training
  reads locally (remote-first, cache-on-pull, gitignored).
- **Project skills (a fixed contract, called by name):**
  - `project-context` → the project's `project_id`, current goal, and how
    training/benchmarks run here. **Invoke this first.**
  - `train` → launch training for this project (Modal/Tinker/local — project's choice).
  - `benchmark` → run evals for this project.

## Lifecycle

0. **Load context.** Invoke `project-context` for `project_id` + conventions.
   Then `store.current_goal(project_id)` and `store.list_goals(project_id)`.
1. **Recall (broad).** `store.experiment_summaries(project_id)` →
   `{id, name, hypothesis, conclusion, is_valid}` for every experiment. Find what's
   relevant to the idea and what's already been concluded. Ignore `is_valid=False`
   experiments as evidence (you may mention them).
2. **Drill down (targeted).** For the most relevant experiment(s):
   `store.experiment_detail(exp_id)` → groups → runs (`run_config` = hyperparameters)
   → evals + checkpoints (`include_metrics=True` for train-log series). Understand
   what configs/benchmarks/results already exist so you don't repeat them.
3. **Propose.** Craft: a one-line **hypothesis**; **groups** = the arms being
   compared (e.g. "Qwen 4B" vs "Qwen 9B"), each with model + dataset + recipe +
   hyperparameters; **runs** = replicate seeds per group; the **benchmark** to
   evaluate on; the **success metric**. Align to the *current goal* (read-only).
   Present the plan and get approval before spending compute.
4. **Materialize → run.** On approval, prefer the **OOP path**:
   * `evsys new-experiment <slug>` → `experiments/<yyyymmdd>_<slug>/{config.yaml,run.py}`.
   * Fill in `config.yaml`:
     - `name`, `metadata.hypothesis`, `metadata.tags`,
       `metadata.project_goal_id`, `metadata.success_metric`.
     - `metadata.benchmark.path` (local) + `metadata.benchmark.id` (from
       `evsys benchmark upload data/benchmark/<name>` if not yet registered).
     - Either a single `run:` block or a `matrix:` sweep — one arm per
       comparison cell (model, hyperparameter, seed).
   * Place any project-specific verifiers / metrics / transforms in
     `scripts/{verifiers,metrics,transforms}.py` (registered via the
     `@register_*` decorators so the YAML's `kind:` lookups resolve).
   * `python experiments/<dir>/run.py` calls `Experiment.from_yaml(...).run()`,
     which creates the experiment + per-arm run records, runs training (per
     arm with failure isolation), auto-forwards local `metrics.jsonl` to the
     store, scores each arm on the benchmark, and writes a conclusion +
     `best_score` on the experiment.

   Only drop down to direct `EvsysStore` / `ExperimentRun` calls when the
   project genuinely needs something the `Experiment` class doesn't model
   (custom rollout loops, post-hoc patching). For SFT/RL sweeps the OOP path
   is enough.
5. **Conclude.** `Experiment.run()` writes a default conclusion + `best_score`
   automatically. Patch over it with `store.set_conclusion(exp_id, ...)` if a
   richer takeaway is warranted — this is what future-you reads first. If you
   discover a bug in a run, `store.invalidate_experiment(exp_id, reason=...)`.

## Hard rules

- **🚨 NEVER let any information from a benchmark / eval set leak into training.**
  This is non-negotiable. Concretely you MUST NOT:
    * Open, read, sample, summarize, or derive features from any benchmark
      jsonl, eval queries file, or eval-side metadata when building training
      data or training-time prompts.
    * Use a benchmark's task list, slug coverage, expected answers, or any
      file derived from those (e.g. `eval_queries_v2.json`,
      `benchmark/<name>/tasks.jsonl`) as a source for things shown to the
      model at train time — including in-context tool lists, candidate sets,
      few-shot exemplars, system prompts, or augmentations.
    * Build "toolkit → candidate list" / "schema → fields" / "category →
      items" maps from the benchmark for use during training. If you need
      such a map, derive it from a training-time source (the model
      provider's full tool catalog, the dataset's own metadata, a separate
      project-supplied registry) — never from the eval suite.
  Even partial overlap is contamination: if 50% of your training rows
  show the model a list whose contents come from the benchmark, the post-
  training pass-rate is meaningless because the model has implicitly
  memorized the benchmark's coverage. **An invalid eval is worse than no eval.**
  If you find yourself reaching for a benchmark file to fill a training-side
  field, STOP and ask the user for a benchmark-independent source.
- **Never change the project goal** (`store.set_goal`) unless the user *explicitly*
  asks to update it. Goals are read-only context for you.
- **Don't spend compute without approval** — propose the plan first (Phase 3),
  then launch.
- **base_model is a per-run hyperparameter** (in `run_config`), never an
  experiment field — a comparison experiment has multiple models across groups.
- **Build on history**: read summaries + drill into relevant experiments before
  proposing, so you never rediscover a dead end already in `conclusion`.
- Pull data to the Workspace once and train locally; never stream rows per-step.

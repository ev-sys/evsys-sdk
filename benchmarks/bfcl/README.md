# BFCL — sandbox-free benchmark harness

Scores any Tinker checkpoint on the **Berkeley Function-Call Leaderboard**
(BFCL), using a faithful, `inspect_ai`-free port of the official AST / abstention
scorer. Scope is the **13 sandbox-free categories** — pure AST function-call
matching plus the two abstention categories. No code execution, no SQL/REST, no
multi-turn state (those are deferred — see [TODO](#todo--deferred-categories)).

**Two harnesses, one shared core.** The point of this dir is to validate the SDK
harbor path against an independent baseline by evaluating the SAME 500 tasks under
IDENTICAL conditions:

* **`evaluate.py` — RAW baseline.** Does NOT import `evsys_sdk` or `harbor`.
  Samples each task straight from the Tinker checkpoint with the `tinker_cookbook`
  renderer + `tinker.SamplingParams` (a hand-mirror of `harbor/llms/tinker.py`).
* **`run_bfcl.py` — SDK harness.** Scores through `Benchmark.score_via_harbor`
  (the real harbor rollout path).

Both import the same **`bfcl_core.py`** for the prompt, the rollout params, the
tool-call parse, and the scoring — so they are byte-identical by construction
(see [Parity guarantee](#parity-guarantee)).

```
benchmarks/bfcl/
  bfcl_core.py       # SHARED single source of truth: build_messages, ROLLOUT_PARAMS,
                     #   parse_tool_calls, score (both harnesses import this)
  build_dataset.py   # raw gorilla BFCL → data/benchmark/bfcl-noexec-500/
  verifier.py        # registers bfcl_match (the ported AST/abstention scorer)
  evaluate.py        # Harness 1 — RAW baseline (no evsys_sdk / harbor)
  run_bfcl.py        # Harness 2 — SDK, via Benchmark.score_via_harbor
  parity_check.py    # smoke test: offline 3-way token-id parity (+ optional live)
  self_test.py       # offline scorer round-trip (no Tinker/harbor/network)
```

## Parity guarantee

To validate the SDK harness against the RAW one, the model input after the chat
template must be **token-identical** between them — and ideally identical to
`inspect_ai`'s native function-calling render too. We get all three from one fact,
verified empirically (`parity_check.py`):

> For the SAME `[{system: tool block}, {user: query}]` messages,
> `inspect_ai`'s native FC render (`tok.apply_chat_template([user], tools=oai,
> add_generation_prompt=True)`) and harbor's `get_renderer("qwen3",
> tok).build_generation_prompt([system, user])` produce **identical token ids**.

So `bfcl_core.build_messages(task)` is the ONLY place a prompt is built: a
**system** message = Qwen3's own native-FC tool block for the task's tools
(extracted verbatim from the chat template, so byte-identical to inspect), and a
**user** message = the BFCL query. Both harnesses render THOSE messages with the
SAME `tinker_cookbook` `"qwen3"` renderer → identical input tokens.

**Per-task system prompts (the SDK side).** Every BFCL task has its own tools, so
each needs its own system message. `Benchmark.score_via_harbor` takes only a
job-level `system_prompt`, which can't carry per-task tools. We added a small,
additive, backward-compatible SDK field — **`HarborTask.system_prompt`** — that
the harbor adapter packs into the per-task `instruction.md` (behind a sentinel)
and `BasicLoopAgent` splits back into `(system, user)`. So `build_dataset.py`
bakes each task's canonical tool block into `system_prompt`, and `run_bfcl.py`
runs the whole 500 in **one** harbor job (no grouping, no per-task jobs). This
mirrors how harbor's own hub BFCL dataset (`harbor run -d gorilla/bfcl`) varies
per task: each task dir is independent and carries its own prompt content.

**Verified parity-check result** (offline, `python parity_check.py --n 20`):
`token-id identical for all 20 tasks: YES` — inspect native-FC == RAW harness ==
SDK harness, byte-for-byte. `ROLLOUT_PARAMS` and the renderer stop sequence
(`[151645]`, read by BOTH harnesses from the renderer) also match.

## How inspect evaluates BFCL (what we ported)

The reference is `inspect_evals/bfcl` (gorilla commit
`dac44e7ac9db5ff26a01ab0c1ec5de5a1e703b7a`). In **FC (function-calling) mode**
the model is given native tools and emits tool calls; the calls are matched
against ground truth — *not executed* — by a category-specific scorer:

| inspect file | what it does | our port (`verifier.py`) |
|---|---|---|
| `score/scorer.py :: ast_match` | dispatch simple/parallel/multiple | `bfcl_match` + `matching_function` |
| `… :: tool_call_matches_possible_answers` | one-call match: name, required params, no-extras, per-param possible values, type rules | `tool_call_matches_possible_answers` |
| `… :: _match_parallel` | all expected calls present, **order-free** | `_match_parallel` |
| `… :: _match_multiple` | exactly one call, one answer | `_match_multiple` |
| `… :: _standardize_string` / `_value_matches` | case-insensitive, strip `, . / - _ * ^`, `'`→`"` | `_standardize_string` / `_value_matches` |
| `… :: _apply_numeric_type_rules` | Python int→float ok; Java/JS need literal float; float-for-int invalid | `_apply_numeric_type_rules` + `_check_array_element_types` |
| `score/scorer.py :: irrelevance_match` / `relevance_match` | abstention: 1.0 iff no call (irrelevance) / a call (relevance) | dispatched in `bfcl_match` |
| `utils/tool_parsing.py :: get_type` / `normalize_function_name` | BFCL types → JSON-schema; dots→underscores | `get_type` / `normalize_function_name` |
| `utils/task_categories.py :: matching_function` | category → scorer name | `matching_function` |
| `data.py :: record_to_sample`, `_func_doc_language_specific_pre_processing` | record → sample; Java/JS param flattening + language hints | `build_dataset.py` |

**How we match inspect's native FC mode:** inspect gives the model native tools
via the chat template's `tools=` argument, which Qwen3 renders as a **system**
message (`# Tools … <tools>{…}</tools> … <tool_call>…</tool_call>`). We reproduce
that exact system block by feeding Qwen3's own chat template the tool schemas and
extracting it verbatim (`bfcl_core.build_system_prompt`), then render
`[{system}, {user}]` with harbor's `tinker_cookbook` `"qwen3"` renderer — which
produces **identical token ids** to inspect's native FC (proven in
`parity_check.py`). The model emits hermes `<tool_call>{…}</tool_call>` blocks and
`verifier.py` parses them before matching. The matching itself is byte-for-byte
the inspect logic.

## Source: raw gorilla data (and why not the harbor hub dataset)

The harbor hub hosts `gorilla/bfcl@latest` (3,641 tasks). We verified it **is**
exactly this non-sandbox subset — per-category counts and ground truth match the
raw gorilla data (parity spot-checked on `simple_python_266`, `parallel_38`,
`multiple_66`, `irrelevance_205`, `live_irrelevance_351-81-12`). **But we do not
use it as the run source**, for two reasons:

1. **Container, not host-side.** Each hub task is a Docker task: the model must
   write `/app/result.json` and a `tests/test.sh` runs an evaluator *inside the
   container*. The SDK's `score_via_harbor` runs host-side with a
   `NoOpEnvironment` + an in-process verifier — it cannot run a container
   `test.sh`. Using the hub dataset directly would require harbor's native
   container path, not the SDK benchmark path this harness targets.
2. **The hub evaluator is not faithful.** Its `evaluate.py` matcher is
   order-*dependent* for parallel categories (`zip(pred, gt)`) and has no
   int/float language rules or string standardization — it diverges from the
   official BFCL scorer.

So `build_dataset.py` sources instructions + tool schemas + ground truth from
raw gorilla (sparse-cloning the same pinned commit) and scores with the faithful
ported `bfcl_match`. The hub dataset stands as the parity reference.

## Field-by-field mapping: BFCL record → HarborTask

| BFCL record (raw gorilla) | HarborTask field |
|---|---|
| `question[0]` (user turn text) | `instruction` (JUST the user query) |
| `function[]` schemas → Qwen3 native-FC tool block | `system_prompt` (the per-task system message) |
| — | `verifier.kind = "in_process"`, `verifier.fn_name = "bfcl_match"` |
| `possible_answer[id].ground_truth` (AST possible answers; `[]` for abstention) | `verifier.expected.ground_truth` |
| `function[]` (after Java/JS flattening + language hints) | `verifier.expected.tools` |
| category → `python`/`java`/`js` | `verifier.expected.language` |
| category name | `verifier.expected.category` **and** `metadata.category` |
| `id` | `task_id = "bfcl-<id>"`, `metadata.bfcl_id = <id>` |

`metadata.category` is what makes `breakdown_keys=["category"]` produce the
per-category accuracy table.

## Stratified-500 sample (deterministic, seed `20240624`)

Largest-remainder apportionment of 500 across the 13 categories proportional to
each category's pool (total pool = 3,641):

| category | pool | drew | | category | pool | drew |
|---|---:|---:|---|---|---:|---:|
| simple_python | 400 | 55 | | live_simple | 258 | 35 |
| simple_java | 100 | 14 | | live_multiple | 1053 | 145 |
| simple_javascript | 50 | 7 | | live_parallel | 16 | 2 |
| multiple | 200 | 28 | | live_parallel_multiple | 24 | 3 |
| parallel | 200 | 28 | | live_relevance | 16 | 2 |
| parallel_multiple | 200 | 27 | | live_irrelevance | 884 | 121 |
| irrelevance | 240 | 33 | | **total** | **3641** | **500** |

## Build

```bash
# uses the pinned-commit data already on disk
python benchmarks/bfcl/build_dataset.py --data-dir <gorilla>/bfcl_eval/data
# or sparse-clone the pin automatically
python benchmarks/bfcl/build_dataset.py
```
Writes `data/benchmark/bfcl-noexec-500/{tasks.jsonl,metadata.yaml}`.

## Run on a checkpoint

Both harnesses take `--model-path <tinker-checkpoint>` (omit for base Qwen3-4B)
and `--limit N` for a fast smoke. Defaults come from `bfcl_core.ROLLOUT_PARAMS`
(`temperature=0.0`, `max_tokens=2048`, `num_samples=1`) + the `"qwen3"` renderer.
Both need a Tinker API key; the SDK harness also needs `harbor` installed.

```bash
# SDK harness (Benchmark.score_via_harbor — ONE harbor job over all 500)
python benchmarks/bfcl/run_bfcl.py  --model-path <tinker-checkpoint>
python benchmarks/bfcl/run_bfcl.py  --model-path <ckpt> --limit 20      # smoke

# RAW baseline (no evsys_sdk / harbor — samples tinker directly)
python benchmarks/bfcl/evaluate.py  --model-path <tinker-checkpoint>
python benchmarks/bfcl/evaluate.py  --model-path <ckpt> --limit 20      # smoke
```

Both print overall accuracy + the per-category breakdown. Because they share
`bfcl_core`, the model sees byte-identical input under both (see
[Parity guarantee](#parity-guarantee)).

## Parity check (smoke test)

```bash
# offline: 3-way token-id parity over 20 tasks (no API key needed)
python benchmarks/bfcl/parity_check.py --n 20
# live (optional): set TINKER_API_KEY to also run 3 tasks through BOTH harnesses
TINKER_API_KEY=... python benchmarks/bfcl/parity_check.py
```
The offline part asserts inspect-native-FC == RAW == SDK token ids for every
sampled task (prints the first divergence on any mismatch) and that
`ROLLOUT_PARAMS` + the renderer stop sequence agree. Current result: **identical
for all 20**. The live part (key-gated) reports per-task RAW vs SDK scores and
whether they agree; without a key it prints a clear skip message.

## Offline self-test (no Tinker)

```bash
python benchmarks/bfcl/self_test.py
```
For ~30 scored tasks it synthesizes the exact GT completion (asserts 1.0) and a
corrupted call (asserts 0.0); for irrelevance, empty → 1.0 and a spurious call →
0.0; for relevance, a call → 1.0 and empty → 0.0. Also asserts
`Benchmark.from_dir` loads all 500 tasks. Current result: **84/84 pass**.

## Fidelity notes / caveats

* **Faithfully ported:** simple/parallel/multiple AST matching (order-free
  parallel), per-param possible-values, required-param + no-extra-param checks,
  string standardization, the Python/Java/JS int↔float rules, one-level array
  element typing, irrelevance/relevance abstention, function-name normalization.
* **Single-turn only.** All 13 in-scope categories are single-turn; the harbor
  rollout is one turn (`max_turns=1`), matching the inspect single-turn solver.
* **Prompt-mode, not native FC.** We surface tools in the prompt (the SDK path
  has no native-tool channel) and parse hermes blocks. A model that emits valid
  calls in another wrapper than `<tool_call>` would be under-scored; the parser
  tolerates minor noise (missing closing tag, double-encoded `arguments`, stray
  prose around the JSON) but not a different envelope.

## TODO — deferred categories

A later phase should add the execution + stateful categories, which need harbor
**container** environments / backends, not the host-side in-process path:

* `exec_simple`, `exec_multiple`, `exec_parallel`, `exec_parallel_multiple` —
  run the predicted call and compare program output. Use the harbor hub's
  container tasks (`gorilla/bfcl`, `tests/test.sh`) via harbor's native job
  path, or an `E2BVerifier`.
* `rest` — live REST API execution (network).
* `sql` — SQL execution against a fixture DB.
* `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`,
  `multi_turn_long_context`, `multi_turn_composite` — stateful multi-turn
  rollouts against the BFCL backend classes (GorillaFileSystem, TradingBot, …);
  need a multi-turn harbor environment that holds state across turns.
```

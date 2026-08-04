---
name: running-benchmarks
description: Run the SkyRL cost benchmarks on rented GPUs and analyse the results. Use when asked to measure throughput or cost per token, fill gaps in the benchmark matrix, or interpret existing benchmark data.
---

# Running the benchmarks

Everything lives in `experiments/`. The goal is one number: **$/M training
tokens**, against Tinker's published **$0.737/M**.

## Shape of a run

1. `experiments/launchers/gaprun.py` acquires boxes and drives them. Set
   `EVSYS_SSH_KEY`, `EVSYS_SRC`, `EVSYS_SCRATCH`.
2. It pushes `experiments/servers/setup_remote.sh` plus a chain script, and runs
   the chain in the background.
3. Each chain starts a server, runs cells, and writes one JSON per cell.
4. `experiments/launchers/aggregate.py` folds every JSON into `all_rows.json`;
   `build_html.py` renders the report.

## Rules that are not optional

**A fresh server per cell that can OOM.** A server that OOMs never recovers — it
silently poisoned 11 consecutive cells once. Every chain restarts between cells.

**Kill Ray properly.** `pkill -f skyrl.tinker.api` leaves Megatron policy workers
holding 121 GiB and stale adapter slots; the next "fresh" server dies inside
`swap_to_adapter` with a CUDA error mentioning nothing about Ray. Always
`ray stop --force` plus `pkill -9 -f 'ray::'` and `raylet`.

**Never `pkill -f <pattern>` over SSH** where the pattern appears in your own
command line — it matches and kills your own shell. Put the kill in a script
whose filename cannot match, and scp it. Heredocs piped into ssh silently do not
apply either.

**Never overwrite a running bash script.** Bash reads by byte offset; replacing
the file mid-run makes it jump to a wrong offset. Write a new filename.

**Pull results as each cell lands**, so a preemption costs one cell, not a box.

## Reading the numbers

**`tok_s` is aggregate across adapters, not per adapter** — `tokens_per_step =
seq_len × batch × tenants`. Adding adapters splits the card; each runs at ~1/n
speed. Dividing cost by adapter count produces a number that is wrong by n×.

**Packing substitutes for batch size.** The gain tracks batch, not utilisation:
batch=64 gains nothing, batch=16 gains 14%, batch=1 gains 36%. Where memory
forces batch=1 (long context) it is the only way left to fill the card. One H100
with 8 adapters is ~9% cheaper per token than eight separate ones, with each
experiment ~7× slower in wall clock.

**Trust `price_hr`, not the `gpu` field.** The scripts hardcoded
`"H200 SXM5 141GB"` for most of this work, mis-attributing three files including
every 8B and 9B run. `aggregate.py` derives the card from launch price.

**Tinker's $0.737/M is quoted at 64K context.** Comparing 8K numbers to it
flatters us; the 64K rows are the honest comparison.

## Configuration that made long context possible

`fused_lm_head_logprob: true` — without it a `[seq, vocab]` fp32 logits tensor is
materialised: 74 GiB at 128K. With it, 64K went 134.6 → 35.9 GiB at identical
throughput. `logprobs_chunk_size` does **not** substitute; it chunks computation
*from* logits that already exist.

`recompute_*` are TransformerConfig fields and must go via
`transformer_config_kwargs`, not the top level.

Under colocation, vLLM's `gpu_memory_utilization` is a **pre-allocation**, not a
ceiling it grows into. At the 0.8 default the engine reserves almost everything
and the policy has nowhere to live; 0.25 works.

## What is measured, and what is missing

Covered: 4B and 8B SFT across RTX PRO 6000 / H100 / A100 / H200, 512→262144,
batch and adapter sweeps, some 4B RL.

Missing: **8B RL entirely** — RL needs vLLM, which rules out A100 and RTX PRO
6000 in this stack, and has only ever produced numbers on H200. Also Qwen3.5-9B,
whose GDN TileLang kernels fail to JIT on both Ampere and Blackwell.

---

# Addendum: the benchmarking doctrine (2026-08)

Everything below was learned running the 8B/9B campaigns from a no-SSH
environment. It supersedes nothing above — it adds to it.

## Rule zero: benchmark on the REAL stack

**Every experiment benchmark runs on SkyRL (tinker server) + vLLM.** A
lightweight HF+PEFT proxy is allowed only as a feasibility probe (does the
model load? where is the OOM wall?), and its numbers must be labeled
non-comparable — the proxy showed 9B "$0.42/M" where honest server numbers
differ, and it cannot see vLLM multiLoRA batching (punica/SGMV), Megatron
sequence parallelism, or rollout economics at all. If someone asks for
experiment numbers, the answer comes from the real stack.

## Running without SSH (blind boxes)

The chains never needed SSH for the work — only for push/pull. Replace both:

1. Embed setup + server scripts + bench clients (base64) in a **cloud-init
   startup script** (Verda `POST /scripts`, then `startup_script_id`).
2. Stream progress and results over HTTPS to an ntfy topic:
   staged `say` calls, and gzip+base64 result files in ≤3000B chunks with
   `sleep 2` between (ntfy silently drops rapid-fire messages).
3. On any failure, stream the **server log tail** — a truncated 260-char
   client error is useless; the server log has the real traceback.
4. Dead-man `shutdown -h` timer sized to the budget; the controller reaps
   instance AND volume after (volumes outlive instances and bill ~$0.08/hr).
5. `uv sync --extra tinker --extra megatron` takes **~90s** on Verda's
   standard image (Ubuntu 22.04 CUDA 12.8) — it is not the risk it looks.
   `git clone` IS flaky — always retry with backoff.

`experiments/spot_limits/launch_skyrl.py` is the reference implementation.

## The experiment matrix (what "benchmarked" means)

Target: Tinker's **$0.737/M at 64K context**. A model×card claim needs:

- **Realistic cells**: batch {32, 256} × seq {4k, 16k, 64k, 256k}. Cells that
  OOM or exceed wall budget are recorded as limits — that IS the data.
- **≥15–20 measured steps** per standard cell (short cells lie; warmup excluded).
  Long-context cells may report fewer under a wall budget — record `steps_done`.
- **Always record** busy-util %, peak memory GiB, watts (longctx's Prof does
  this natively), tok/s, $/M.
- **multiLoRA saturation sweep** (tenants 1,2,4,8,… until aggregate throughput
  stops rising) at the throughput-optimal cell — this is how cost per
  *experiment* is amortized: N adapters ≈ one adapter's memory, each at ~1/N
  speed, aggregate ~9% cheaper per token.
- **Both classes**: non-rollout (SFT/distill via `longctx.py`) and rollout
  (RL via `rlmatrix.py` — rollout economics dominate; generation is ~10× the
  update cost without a fast engine).

## Accelerator policy

- **Spot first, always.** Same silicon, ~⅓ the price; ~1 preemption per
  successful launch, so pull results per cell, never per box.
- **Default ceiling: H100** (1× or 2× spot). Use H200/B200 or on-demand only
  when (a) explicitly authorized, or (b) the experiment is infeasible below it
  — documented cases: RL has only ever produced numbers on H200-class; 9B
  needs TP≥2.
- Per-GPU price is the unit for $/M (`--price-hr` is per GPU, not per box).
- Availability listings lag reality; the per-SKU endpoint is the authority,
  and stock churns minute-to-minute — retry, or take the equivalent card.

## Qwen3.5-9B: the recipe (and the traps)

- Megatron **TP=1 fails at `MegatronPolicyWorkerBase.init_model()`** — this
  looked like "9B doesn't run" but is a sharding floor, not an arch wall.
  Upstream runs 9B DAPO with **TP=4 on 8×H100** (full FT + Adam); LoRA-only
  serving wants TP=2 minimum. TP>1 auto-enables sequence parallelism.
- vLLM 0.23 serves 9B with `engine_init_kwargs={"gdn_prefill_backend":
  "triton"}` and `enforce_eager=true`.
- `trainer.remove_microbatch_padding=false` — **sample packing is not
  supported for GDN layers in Megatron** (Megatron-LM #2644), so the packing
  gains documented above do not apply to 9B.
- GDN TileLang kernels fail to JIT on Ampere and Blackwell; Hopper is the
  only lane.

## Known limits (agent knowledge — update after every campaign)

| Model | Card | Stack | Best measured | Context ceiling | Notes |
|---|---|---|---|---|---|
| 4B | H200 spot | SkyRL/Megatron | **$0.050/M** @8k×8 (14.7× vs Tinker) | 256K (fused logprob) | validated blind 2026-08 |
| 4B | RTX PRO 6000 spot | SkyRL | $0.0318/M | — | branch headline, 23.2× |
| 8B | RTX PRO 6000 / A100 | SkyRL | $0.0468–0.0625/M | — | branch |
| 8B RL | H200 only | SkyRL+vLLM | — | — | A100/RTX ruled out; single H100 died post-KV-alloc |
| 9B | any, TP=1 | SkyRL/Megatron | **fails init** | — | needs TP≥2 |
| 9B | 1×H100 | HF proxy (non-comparable) | $0.99/M @2k×1 | 32k (chunked-CE) | feasibility only |
| 9B RL | 2×H200 | HF generate proxy | $2.47/M @64 seqs | — | rollout-bound 10:1; real vLLM numbers pending |

## Checkpointing between preemptions

`src/evsys_sdk/checkpoint_delta.py`: store base weights once, each checkpoint
as `compress(state XOR base)` — lossless, sparse fine-tune deltas ~377×
smaller; `DeltaCheckpointer(keep_last=k)` bounds disk for overwrite-in-place
on a persistent volume. Status: unit-tested (96% cov). Cross-node
restore-from-volume on Verda: **not yet validated** — do that before relying
on it for preemption recovery.

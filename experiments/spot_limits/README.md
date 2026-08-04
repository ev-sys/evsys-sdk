# Spot 8B/9B context-length limit sweep

Finds the memory/throughput **limit** (OOM ceiling) for `Qwen3-8B` and
`Qwen3.5-9B` across Verda spot cards, measuring utilisation, peak memory and
cost per rung. Autonomous, no inbound SSH: each box runs `bench.py` from a
cloud-init startup script and streams every rung to an ntfy topic over HTTPS.

**Method (matters for reading the numbers):** LoRA r16 adapters, HF
Transformers + SDPA attention, `device_map="auto"`, synthetic token batches,
batch=1. This measures the **kernel/memory envelope and where it OOMs** — it is
deliberately *not* the SkyRL/Megatron pipeline, so `usd_per_M` here is **not**
comparable to the earlier `experiments/results/*.json` ($/M) figures.

## Measured ceilings (context before OOM)

| Model | 1×A100 80GB (spot) | 2×H100 160GB (spot) | 1×H200 141GB (on-demand) |
|---|---|---|---|
| Qwen3-8B  | ~2k (OOM @ 8k)  | ~8k (OOM @ 16k)  | ~8k (OOM @ 16k) |
| Qwen3.5-9B | ~2k (OOM @ 8k) | ~4k (OOM @ 8k)   | ~4k (OOM @ 8k) |

(2×H200 spot was intermittently out of stock during the run; the H200 point is
a single 141 GB card on-demand — same ceiling as 2×H100, reached faster since a
single card has no cross-GPU pipeline overhead.)

## Notes
- **Memory-bound, not compute-bound on H100:** 8B on 2×H100 sits at ~56% GPU
  util — it OOMs with compute to spare. On 1×A100 it is compute-bound (100%).
- **A100 spot is the cheapest per token at short context** ($0.042/M vs
  $0.075/M for 8B @ 2k) purely because A100 spot is ~3.6× cheaper/hour — but it
  OOMs by 8k, so H100 is required for longer context.
- **Qwen3.5-9B runs on both Ampere and Hopper via the HF/SDPA reference path**,
  bypassing the SkyRL TileLang GDN kernels that fail to JIT — but only in an
  unoptimised, memory-heavy, low-throughput mode (5–9× slower than 8B). Not a
  practical fix for the fast-path problem, but the model itself is runnable.

Run one card: `python launch_card.py <sku> <region> <label> <price_hr> <8b_rungs> <9b_rungs>`
(needs VERDA_CLIENT_ID/SECRET in env and a registered ssh key id in key_id.txt).

"""Self-measuring context-length LIMIT sweep for a Qwen model on this box.

Fire-and-forget: no inbound SSH. Streams every rung to an ntfy topic over HTTPS
(the only egress the controller can read). Finds where throughput/memory stops
being worth it and where it OOMs — the "limit" the run is after. Uses LoRA
adapters (matches the earlier adapter sweeps) and SDPA attention (memory-
efficient, no flash-attn build). Synthetic token batches: this measures the
kernel/memory envelope, not a workload — stated plainly so nobody over-reads it.
"""
import json, os, time, threading, subprocess, urllib.request, traceback

TOPIC = os.environ["BENCH_TOPIC"]
NT = f"https://ntfy.sh/{TOPIC}"
MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen3-8B")
PRICE_HR = float(os.environ.get("PRICE_HR", "0"))
CARD = os.environ.get("CARD", "?")
RUNGS = [int(x) for x in os.environ.get("RUNGS", "2048,8192,32768,65536").split(",")]
STEPS = int(os.environ.get("STEPS", "12"))


def say(title, msg):
    try:
        urllib.request.urlopen(urllib.request.Request(
            NT, data=str(msg).encode()[:3800], headers={"Title": str(title)[:200]}), timeout=15)
    except Exception:
        pass


def smi():
    """(avg_util%, peak_mem_MiB) across all GPUs, one instantaneous sample."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=10).strip().splitlines()
        us = [tuple(int(v) for v in l.split(",")) for l in out]
        return (sum(u for u, _ in us) / len(us), max(m for _, m in us))
    except Exception:
        return (0, 0)


class Sampler(threading.Thread):
    """Background nvidia-smi sampler → peak util / peak mem for the current rung."""
    def __init__(self):
        super().__init__(daemon=True); self.on = True; self.pu = 0; self.pm = 0
    def reset(self): self.pu = 0; self.pm = 0
    def run(self):
        while self.on:
            u, m = smi()
            self.pu = max(self.pu, u); self.pm = max(self.pm, m)
            time.sleep(1)


def main():
    say("setup", f"importing torch on {CARD} for {MODEL}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    ngpu = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(ngpu)]
    say("gpu", f"{ngpu} GPU(s): {names}")

    t0 = time.time()
    say("load", f"loading {MODEL} bf16 sdpa device_map=auto …")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", trust_remote_code=True)
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      lora_dropout=0.0, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    say("load", f"loaded+LoRA in {time.time()-t0:.0f}s; trainable "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    samp = Sampler(); samp.start()
    results, prev_toks = [], None
    for seq in RUNGS:
        for i in range(ngpu): torch.cuda.reset_peak_memory_stats(i)
        samp.reset()
        try:
            ids = torch.randint(0, tok.vocab_size or 150000, (1, seq), device="cuda:0")
            # warmup
            out = model(ids, labels=ids); out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(STEPS):
                out = model(ids, labels=ids); out.loss.backward()
                opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.time() - t
            toks = seq * STEPS / dt
            peak = sum(torch.cuda.max_memory_allocated(i) for i in range(ngpu)) / 1e9
            usd_per_M = (PRICE_HR / 3600.0) / toks * 1e6 if toks else None
            r = dict(card=CARD, model=MODEL, seq=seq, tok_s=round(toks, 1),
                     peak_mem_gb=round(peak, 1), peak_util=samp.pu, peak_mem_mib=samp.pm,
                     usd_per_M=round(usd_per_M, 4) if usd_per_M else None,
                     loss=round(float(out.loss), 3))
            results.append(r)
            say(f"rung seq={seq}", json.dumps(r))
            # Run every rung until the card OOMs — the OOM point IS the limit.
            prev_toks = toks
        except torch.cuda.OutOfMemoryError:
            say(f"OOM seq={seq}", json.dumps(dict(card=CARD, model=MODEL, seq=seq,
                limit="OOM", last_ok=results[-1] if results else None)))
            for i in range(ngpu): torch.cuda.empty_cache()
            break
        except Exception as e:
            say(f"error seq={seq}", f"{type(e).__name__}: {e}\n{traceback.format_exc()[:1500]}")
            break
    samp.on = False
    say("SWEEP_DONE", json.dumps(dict(card=CARD, model=MODEL, rungs=results)))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        say("FATAL", f"{type(e).__name__}: {e}\n{traceback.format_exc()[:2000]}")

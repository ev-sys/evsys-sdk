"""9B multi-axis sweep with OOM-prevention, mirroring SkyRL's long-context config.

Implements the two knobs that made 128K/256K possible in the earlier 4B runs
(server_colo.sh): a CHUNKED cross-entropy that never materialises the full
[B, T, vocab] logits tensor (SkyRL: fused_lm_head_logprob + logprobs_chunk_size
1024), plus gradient checkpointing for layer activations. Sweeps seq_len x batch
(and LoRA rank), measuring util / peak-mem / tok_s / $/M per cell, to OOM.
Streams every cell to ntfy over HTTPS. Synthetic tokens: measures the envelope.
"""
import json, os, time, threading, subprocess, traceback, urllib.request

TOPIC = os.environ["BENCH_TOPIC"]
NT = f"https://ntfy.sh/{TOPIC}"
MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen3.5-9B")
PRICE_HR = float(os.environ.get("PRICE_HR", "0"))
CARD = os.environ.get("CARD", "?")
CHUNK = int(os.environ.get("CHUNK", "1024"))     # == SkyRL logprobs_chunk_size
WARMUP = int(os.environ.get("WARMUP", "1"))
STEPS = int(os.environ.get("STEPS", "6"))        # measured steps — "small decent number"
CELL_BUDGET = float(os.environ.get("CELL_BUDGET", "75"))  # s: stop a cell early, report steps done
# (seq_len, batch, lora_rank) cells. Primary: seq ladder at b=1; then batch at 2k/8k; then rank.
CELLS = json.loads(os.environ.get("CELLS", json.dumps(
    [[2048, 1, 16], [8192, 1, 16], [16384, 1, 16], [32768, 1, 16], [65536, 1, 16],
     [2048, 2, 16], [2048, 4, 16], [2048, 8, 16], [8192, 2, 16], [8192, 4, 16], [2048, 1, 64]])))


def say(title, msg):
    try:
        urllib.request.urlopen(urllib.request.Request(
            NT, data=str(msg).encode()[:3800], headers={"Title": str(title)[:200]}), timeout=15)
    except Exception:
        pass


def smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=10).strip().splitlines()
        us = [tuple(int(v) for v in l.split(",")) for l in out]
        return (sum(u for u, _ in us) / len(us), max(m for _, m in us))
    except Exception:
        return (0, 0)


class Sampler(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.on = True; self.pu = 0; self.pm = 0
    def reset(self): self.pu = 0; self.pm = 0
    def run(self):
        while self.on:
            u, m = smi(); self.pu = max(self.pu, u); self.pm = max(self.pm, m); time.sleep(1)


def chunked_ce(hidden, weight, targets, chunk):
    """Cross-entropy without materialising full logits: project + CE per seq-chunk.

    hidden [B,T,H], weight [V,H] (lm_head), targets [B,T]. Sums loss over chunks so
    the [B, chunk, V] logits tensor is the only one alive — the whole point.
    """
    import torch, torch.nn.functional as F
    B, T, H = hidden.shape
    h = hidden.reshape(B * T, H); tgt = targets.reshape(B * T)
    total = h.new_zeros(())
    n = 0
    for i in range(0, B * T, chunk):
        hc = h[i:i + chunk]
        logits = torch.nn.functional.linear(hc, weight)      # [chunk, V]
        total = total + F.cross_entropy(logits.float(), tgt[i:i + chunk], reduction="sum")
        n += hc.shape[0]
    return total / max(n, 1)


def inner(peft_model):
    """Return (transformer, lm_head) from a PEFT-wrapped HF causal LM."""
    cm = peft_model.base_model.model          # e.g. Qwen3ForCausalLM (LoRA-injected)
    return cm.model, cm.lm_head


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    ngpu = torch.cuda.device_count()
    say("gpu", f"{CARD}: {ngpu}x {torch.cuda.get_device_name(0)}; chunk={CHUNK} steps={STEPS}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    say("load", f"loaded {MODEL} in {time.time()-t0:.0f}s; grad-checkpoint on, chunked-CE on")

    samp = Sampler(); samp.start()
    results = []
    cur_rank = None
    peft = None
    for seq, batch, rank in CELLS:
        try:
            import torch
            if rank != cur_rank:
                base = model
                lora = LoraConfig(r=rank, lora_alpha=rank * 2,
                                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                  lora_dropout=0.0, task_type="CAUSAL_LM")
                peft = get_peft_model(base, lora); peft.train()
                peft.gradient_checkpointing_enable()
                opt = torch.optim.AdamW([p for p in peft.parameters() if p.requires_grad], lr=1e-4)
                transformer, lm_head = inner(peft)
                cur_rank = rank
            for i in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(i)
            samp.reset()
            ids = torch.randint(0, tok.vocab_size or 150000, (batch, seq), device="cuda:0")
            tgt = torch.roll(ids, -1, dims=1)

            def one():
                out = transformer(input_ids=ids, use_cache=False)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                loss = chunked_ce(hidden, lm_head.weight, tgt, CHUNK)
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                return float(loss)

            for _ in range(WARMUP): one()
            torch.cuda.synchronize(); t = time.time()
            last = 0.0; done = 0
            for _ in range(STEPS):
                last = one(); done += 1
                if time.time() - t > CELL_BUDGET:   # bound slow (long-ctx) cells
                    break
            torch.cuda.synchronize(); dt = time.time() - t
            toks = seq * batch * done / dt
            peak = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9
            r = dict(card=CARD, model=MODEL, seq=seq, batch=batch, rank=rank,
                     tok_s=round(toks, 1), peak_mem_gb=round(peak, 1), peak_util=samp.pu,
                     usd_per_M=round((PRICE_HR / 3600.0) / toks * 1e6, 4) if toks else None,
                     steps=done, loss=round(last, 3))
            results.append(r); say(f"cell s{seq}xb{batch}r{rank}", json.dumps(r))
        except torch.cuda.OutOfMemoryError:
            say(f"OOM s{seq}xb{batch}r{rank}", json.dumps(dict(card=CARD, seq=seq, batch=batch, rank=rank, limit="OOM")))
            torch.cuda.empty_cache()
        except Exception as e:
            say(f"err s{seq}xb{batch}", f"{type(e).__name__}: {e}\n{traceback.format_exc()[:1200]}")
            torch.cuda.empty_cache()
    samp.on = False
    say("SWEEP_DONE", json.dumps(dict(card=CARD, model=MODEL, cells=results)))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        say("FATAL", f"{type(e).__name__}: {e}\n{traceback.format_exc()[:2000]}")

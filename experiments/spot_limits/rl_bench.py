"""Minimal GRPO-style RL throughput benchmark (HF generate; no vLLM).

RL is not SFT: each step is (1) ROLLOUT — sample G completions per prompt,
(2) REWARD, (3) group-relative ADVANTAGE, (4) POLICY UPDATE on the sampled
sequences. My earlier benches only did (4), which is why "RL" was missing.

This runs the whole loop so we get real RL numbers: rollout vs train time split,
tokens/s, memory, utilization, $/M. It deliberately uses HF ``generate`` (with KV
cache) rather than vLLM, so it *runs anywhere* — but rollout will be slow, which
is exactly the point the SkyRL/vLLM stack exists to fix. Streams to ntfy.
"""
import json, os, time, threading, subprocess, traceback, urllib.request

TOPIC = os.environ["BENCH_TOPIC"]; NT = f"https://ntfy.sh/{TOPIC}"
MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen3.5-9B")
PRICE_HR = float(os.environ.get("PRICE_HR", "0")); CARD = os.environ.get("CARD", "?")
CHUNK = int(os.environ.get("CHUNK", "1024"))
# cells: (num_prompts, group_size, prompt_len, gen_len). group_size G = completions/prompt.
CELLS = json.loads(os.environ.get("RL_CELLS", json.dumps(
    [[4, 4, 256, 128], [8, 4, 256, 128], [8, 8, 256, 128], [16, 8, 128, 128], [8, 8, 256, 256]])))
STEPS = int(os.environ.get("STEPS", "3"))


def say(t, m):
    try:
        urllib.request.urlopen(urllib.request.Request(NT, data=str(m).encode()[:3800],
            headers={"Title": str(t)[:200]}), timeout=15)
    except Exception:
        pass


def smi():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"], text=True, timeout=10).strip().splitlines()
        us = [tuple(int(v) for v in l.split(",")) for l in o]
        return sum(u for u, _ in us) / len(us), max(m for _, m in us)
    except Exception:
        return 0, 0


class Sampler(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.on = True; self.pu = 0
    def run(self):
        while self.on:
            u, _ = smi(); self.pu = max(self.pu, u); time.sleep(0.5)


def chunked_logprobs(hidden, weight, targets, chunk):
    """Sum log p(target) over the sequence without full logits (as in SFT bench)."""
    import torch, torch.nn.functional as F
    weight = weight.to(hidden.device)
    B, T, H = hidden.shape
    h = hidden.reshape(B * T, H); tgt = targets.reshape(B * T).to(hidden.device)
    lp = h.new_zeros(B * T)
    for i in range(0, B * T, chunk):
        logits = F.linear(h[i:i + chunk], weight).float()
        lp[i:i + chunk] = torch.log_softmax(logits, -1).gather(1, tgt[i:i + chunk, None]).squeeze(1)
    return lp.reshape(B, T)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    say("gpu", f"{CARD}: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto", trust_remote_code=True)
    peft = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    peft.train()
    peft.gradient_checkpointing_enable()      # the fix: RL policy forward was storing all activations -> OOM
    peft.enable_input_require_grads()
    cm = peft.base_model.model
    opt = torch.optim.AdamW([p for p in peft.parameters() if p.requires_grad], lr=1e-5)
    say("load", f"loaded {MODEL} in {time.time()-t0:.0f}s; RL loop (HF generate)")
    samp = Sampler(); samp.start()
    V = tok.vocab_size or 150000
    results = []
    for nprompt, G, plen, glen in CELLS:
        try:
            samp.pu = 0
            for i in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(i)
            prompts = torch.randint(0, V, (nprompt, plen), device="cuda:0")
            rollout_t = train_t = 0.0; gen_tokens = train_tokens = 0
            for _ in range(STEPS):
                # 1) ROLLOUT: G samples per prompt
                rep = prompts.repeat_interleave(G, 0)
                torch.cuda.synchronize(); r0 = time.time()
                with torch.no_grad():
                    cm.config.use_cache = True
                    out = cm.generate(rep, max_new_tokens=glen, do_sample=True, temperature=1.0,
                                      pad_token_id=tok.eos_token_id or 0)
                torch.cuda.synchronize(); rollout_t += time.time() - r0
                comp = out[:, rep.shape[1]:]
                gen_tokens += comp.numel()
                torch.cuda.empty_cache()          # release the generation KV cache before the update
                # 2) REWARD (synthetic) + 3) group-relative advantage (GRPO)
                rew = comp.float().mean(1).view(nprompt, G)
                adv = ((rew - rew.mean(1, keepdim=True)) / (rew.std(1, keepdim=True) + 1e-6)).view(-1)
                # 4) POLICY UPDATE on prompt+completion
                cm.config.use_cache = False
                torch.cuda.synchronize(); u0 = time.time()
                full = out
                tr = cm.model(input_ids=full, use_cache=False)
                hidden = tr.last_hidden_state if hasattr(tr, "last_hidden_state") else tr[0]
                lp = chunked_logprobs(hidden[:, :-1], cm.lm_head.weight, full[:, 1:], CHUNK)
                comp_lp = lp[:, rep.shape[1] - 1:].mean(1)         # mean logprob over completion
                loss = -(adv.to(comp_lp.device) * comp_lp).mean()
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); train_t += time.time() - u0
                train_tokens += full.numel()
            tot = rollout_t + train_t
            r = dict(card=CARD, model=MODEL, kind="rl", nprompt=nprompt, group=G, plen=plen, glen=glen,
                     rollout_s=round(rollout_t, 1), train_s=round(train_t, 1),
                     rollout_tok_s=round(gen_tokens / rollout_t, 1) if rollout_t else None,
                     e2e_tok_s=round((gen_tokens + train_tokens) / tot, 1) if tot else None,
                     peak_mem_gb=round(sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9, 1),
                     peak_util=samp.pu,
                     usd_per_M=round((PRICE_HR / 3600.0) / (gen_tokens / tot) * 1e6, 4) if tot and gen_tokens else None)
            results.append(r); say(f"rl n{nprompt}g{G}p{plen}c{glen}", json.dumps(r))
        except torch.cuda.OutOfMemoryError:
            say(f"OOM rl n{nprompt}g{G}", json.dumps(dict(card=CARD, kind="rl", nprompt=nprompt, group=G, limit="OOM")))
            torch.cuda.empty_cache()
        except Exception as e:
            say(f"err rl n{nprompt}", f"{type(e).__name__}: {e}\n{traceback.format_exc()[:1200]}")
            torch.cuda.empty_cache()
    samp.on = False
    say("RL_DONE", json.dumps(dict(card=CARD, model=MODEL, cells=results)))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        say("FATAL", f"{type(e).__name__}: {e}\n{traceback.format_exc()[:2000]}")

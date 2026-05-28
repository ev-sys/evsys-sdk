"""Run the SkyRL + vLLM multi-LoRA "always-hot" Tinker loop on Modal.

This brings up SkyRL's Tinker-compatible server (`skyrl.tinker.api`) with the
Megatron backend + vLLM rollouts and multi-LoRA enabled, then launches N
concurrent training clients — one LoRA adapter each — against the single warm
endpoint (`http://localhost:8000`). It is the self-hosted, pay-GPU-per-hour
equivalent of the hosted Tinker service: many experiments multiplex on one set
of GPUs instead of paying per client.

Architecture (see the field report "Multi-LoRA Training for Continual Learning"):
  * Trainer:  SkyRL-Train (Megatron) holds the base weights + per-tenant LoRA
              adapters and runs forward_backward / optim_step.
  * Sampler:  vLLM serves all adapters hot (SGMV multi-LoRA decode).
  * Sync:     save_weights_for_sampler broadcasts updated LoRA into vLLM.
  * Interface: the Tinker HTTP API, so any Tinker client (incl. our SDK's
              TinkerBackend with base_url=...) is a drop-in.

Image is the prebuilt `novaskyai/skyrl-train-ray-...cu12.8` (the blog's
recommended env) so we don't fight Megatron/CUDA builds. Repos are cloned +
installed into image layers; the run logic is parameterized so it can be tuned
at `modal run` time without rebuilding.

Usage:
    modal run examples/modal/skyrl_multilora.py \
        --n-loras 2 --max-steps 2 --gpu "H100:2" --model "Qwen/Qwen3-4B-Instruct-2507"
"""

from __future__ import annotations

import json

import modal

SKYRL_REF = "skyrl-v0.2.0"
COOKBOOK_REF = "aa602f5"  # 2026-04-23, one day after skyrl-v0.2.0 (API-compat)
REMOTE = "/root"
HF_CACHE = "/root/.cache/huggingface"
LORA_SYNC = "/tmp/lora_sync/multilora"

hf_volume = modal.Volume.from_name("skyrl-hf-cache", create_if_missing=True)

# LoRA training on SkyRL requires the MEGATRON backend (the FSDP worker has no
# prime_optimizer_state — LoRA priming is Megatron-only). The `megatron` extra
# pulls transformer-engine==2.11.0 with no prebuilt wheel, so it compiles from
# source and needs cuDNN dev headers — installed via apt from the cuda-devel
# image's NVIDIA repo. flash-attn/mamba/causal-conv1d come as prebuilt wheels.
EXTRA = "megatron"
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential", "ca-certificates", "libnuma1", "numactl")
    # cuDNN dev headers + clang (the transformer-engine-torch extension build
    # invokes clang++) so the TE source build compiles.
    .run_commands(
        "apt-get update && apt-get install -y clang "
        "&& (apt-get install -y libcudnn9-dev-cuda-12 || apt-get install -y libcudnn9-dev-cuda-13 || true)",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
    )
    .env({
        "HF_HOME": HF_CACHE,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "UV_LINK_MODE": "copy",
        "CUDNN_PATH": "/usr",
        "CPATH": "/usr/include:/usr/local/cuda/include",
        "PATH": "/root/.local/bin:/usr/local/cuda/bin:${PATH}",
    })
    .run_commands(
        # SkyRL: tag clone (depth 1 OK for tags). tinker-cookbook: SHA, so full
        # clone + checkout (git clone --depth 1 -b <SHA> doesn't work for SHAs).
        f"cd {REMOTE} && git clone --depth 1 -b {SKYRL_REF} https://github.com/NovaSky-AI/SkyRL.git",
        f"cd {REMOTE} && git clone https://github.com/thinking-machines-lab/tinker-cookbook.git "
        f"&& cd tinker-cookbook && git checkout {COOKBOOK_REF}",
        f"cd {REMOTE}/SkyRL && uv sync --extra tinker --extra {EXTRA}",
        f"cd {REMOTE}/tinker-cookbook && uv sync --extra math-rl",
        gpu="any",
    )
)

app = modal.App("tl-skyrl-multilora")


def _server_backend_config(n_loras: int, train_gpus: int, infer_gpus: int) -> str:
    """Megatron + vLLM multi-LoRA backend config (non-colocated split).

    Scaled down from the field-report 4+4 recipe to 2 train + 2 infer on
    H100:4 — the 1+1 layout hung at actor-group init. max_loras hot adapters
    served by vLLM; lora_sync_path is the trainer→vLLM weight-sync channel."""
    return json.dumps({
        "strategy": "megatron",
        "trainer.placement.colocate_all": False,
        "trainer.placement.policy_num_gpus_per_node": train_gpus,
        "trainer.policy.megatron_config.tensor_model_parallel_size": train_gpus,
        "trainer.policy.megatron_config.lora_config.merge_lora": False,
        "trainer.micro_train_batch_size_per_gpu": 8,
        "trainer.micro_forward_batch_size_per_gpu": 8,
        "trainer.policy.model.lora.max_loras": n_loras,
        "trainer.policy.model.lora.max_cpu_loras": n_loras,
        "trainer.policy.model.lora.lora_sync_path": LORA_SYNC,
        "generator.inference_engine.run_engines_locally": True,
        "generator.inference_engine.num_engines": 1,
        "generator.inference_engine.tensor_parallel_size": infer_gpus,
        "generator.inference_engine.gpu_memory_utilization": 0.8,
        "generator.inference_engine.max_num_seqs": 128,
    })


@app.function(image=image, timeout=20 * 60)
def smoke() -> dict:
    """CPU-only probe of the prebuilt novaskyai env: which python, what's already
    installed (skyrl/megatron/vllm/...), where cuDNN/CUDA live, and whether the
    SkyRL repo has the tinker server. Informs how to install without recompiling."""
    import subprocess

    probe = r'''
import importlib.util as u
mods = ["skyrl","skyrl.tinker.api","skyrl_train","vllm","megatron","megatron.core",
        "transformer_engine","flash_attn","torch","jax","fastapi","sqlmodel","apex",
        "ray","sglang","uv"]
for m in mods:
    try:
        print(("OK  " if u.find_spec(m) else "MISS"), m)
    except Exception as e:
        print("ERR ", m, type(e).__name__)
import sys; print("python:", sys.executable, sys.version.split()[0])
'''
    out: dict[str, str] = {}
    for key, cmd in {
        "which_python": ["bash", "-lc", "which python python3 uv; python3 --version"],
        "modules": ["python3", "-c", probe],
        "pip_pkgs": ["bash", "-lc",
                     "python3 -m pip list 2>/dev/null | grep -iE "
                     "'skyrl|megatron|vllm|transformer.?engine|flash.?attn|^torch|jax|fastapi|sqlmodel|ray|sglang' || true"],
        "cudnn": ["bash", "-lc",
                  "ls -1 /usr/local/cuda*/include/cudnn.h 2>/dev/null; "
                  "find / -name cudnn.h 2>/dev/null | head -3; echo CUDA_HOME=$CUDA_HOME"],
        "skyrl_tinker_dir": ["bash", "-lc", "ls /root/SkyRL/skyrl/tinker/ 2>&1 | head"],
    }.items():
        r = subprocess.run(cmd, capture_output=True, text=True)
        out[key] = (r.stdout + r.stderr).strip()[-1200:]
    return out


@app.function(image=image, gpu="H100:2", timeout=60 * 60,
              volumes={HF_CACHE: hf_volume})
def run_multilora(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    n_loras: int = 2,
    max_steps: int = 2,
    train_gpus: int = 1,
    infer_gpus: int = 1,
    server_warmup_s: int = 1500,
) -> dict:
    import os
    import signal
    import subprocess
    import threading
    import time
    import urllib.request

    os.makedirs(LORA_SYNC, exist_ok=True)
    # UV_NO_SYNC skips `uv run`'s auto-sync so the Ray worker reuses each
    # project's prebuilt venv instead of re-syncing ~3GB. Don't set
    # UV_PROJECT_ENVIRONMENT — it would force the client (cwd=tinker-cookbook)
    # to use SkyRL's venv (which has no chz). Each `uv run` auto-picks the
    # right .venv based on cwd.
    env = {
        **os.environ, "HOME": "/root", "TINKER_API_KEY": "tml-dummy",
        "TINKER_BASE_URL": "http://127.0.0.1:8000",
        "UV_NO_SYNC": "1",
    }
    skyrl = f"{REMOTE}/SkyRL"
    cookbook = f"{REMOTE}/tinker-cookbook"

    # ---- 1. Launch the SkyRL multi-LoRA Tinker server ----
    server_log: list[str] = []
    server_cmd = [
        "uv", "run", "--extra", "tinker", "--extra", EXTRA,
        "-m", "skyrl.tinker.api",
        "--base-model", model, "--backend", EXTRA, "--port", "8000",
        "--backend-config", _server_backend_config(n_loras, train_gpus, infer_gpus),
    ]
    print(">>> [modal] starting SkyRL multi-LoRA server", flush=True)
    server = subprocess.Popen(server_cmd, cwd=skyrl, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              start_new_session=True)

    def _pump(stream, tag, sink):
        for line in stream:
            sink.append(line)
            print(f"[{tag}] {line}", end="", flush=True)
    threading.Thread(target=_pump, args=(server.stdout, "server", server_log), daemon=True).start()

    def _healthy() -> bool:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/api/v1/healthz", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    # ---- 2. Wait for healthz ----
    deadline = time.time() + server_warmup_s
    ok = False
    while time.time() < deadline:
        if server.poll() is not None:
            return {"status": "failed", "stage": "server_boot",
                    "server_tail": "".join(server_log[-80:])}
        if _healthy():
            ok = True
            break
        time.sleep(5)
    if not ok:
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        return {"status": "failed", "stage": "healthz_timeout",
                "server_tail": "".join(server_log[-80:])}
    print(">>> [modal] server healthy; launching concurrent LoRA clients", flush=True)

    # ---- 3. Launch N concurrent training clients (one LoRA adapter each) ----
    clients = []
    for i in range(n_loras):
        cmd = [
            "uv", "run", "--extra", "math-rl", "-m", "tinker_cookbook.recipes.math_rl.train",
            f"base_url={env['TINKER_BASE_URL']}", f"model_name={model}", "env=gsm8k",
            f"log_path=/tmp/ml-{i}", "groups_per_batch=8", "group_size=4",
            "lora_rank=8", f"max_steps={max_steps}", "eval_every=0", "save_every=0",
            "max_tokens=256", "behavior_if_log_dir_exists=delete", f"seed={i}",
        ]
        log: list[str] = []
        p = subprocess.Popen(cmd, cwd=cookbook, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        threading.Thread(target=_pump, args=(p.stdout, f"client{i}", log), daemon=True).start()
        clients.append((i, p, log))
        time.sleep(3)

    results = {}
    for i, p, log in clients:
        rc = p.wait()
        results[f"client{i}_rc"] = rc
        results[f"client{i}_tail"] = "".join(log[-30:])

    try:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass

    all_ok = all(results[f"client{i}_rc"] == 0 for i in range(n_loras))
    return {"status": "completed" if all_ok else "failed",
            "n_loras": n_loras, "max_steps": max_steps, **results,
            "server_tail": "".join(server_log[-40:])}


@app.local_entrypoint()
def check():
    print("==== SMOKE ====")
    for k, v in smoke.remote().items():
        print(f"\n--- {k} ---\n{v}" if k in ("stdout", "stderr") else f"{k}: {v}")


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-4B-Instruct-2507", n_loras: int = 2,
         max_steps: int = 2, gpu: str = "H100:2",
         train_gpus: int = 1, infer_gpus: int = 1):
    res = run_multilora.with_options(gpu=gpu).remote(
        model=model, n_loras=n_loras, max_steps=max_steps,
        train_gpus=train_gpus, infer_gpus=infer_gpus)
    print("\n==== RESULT ====")
    for k, v in res.items():
        if k.endswith("_tail"):
            print(f"\n--- {k} ---\n{v}")
        else:
            print(f"{k}: {v}")

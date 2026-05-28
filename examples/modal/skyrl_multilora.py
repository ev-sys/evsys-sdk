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
        --n-loras 2 --max-steps 2 --gpu "H100:2" --model "Qwen/Qwen2.5-1.5B-Instruct"
"""

from __future__ import annotations

import json

import modal

SKYRL_REF = "main"
COOKBOOK_REF = "main"
REMOTE = "/root"
HF_CACHE = "/root/.cache/huggingface"
LORA_SYNC = "/tmp/lora_sync/multilora"

hf_volume = modal.Volume.from_name("skyrl-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8", add_python=None
    )
    .apt_install("git", "curl")
    .env({
        "HF_HOME": HF_CACHE,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "UV_LINK_MODE": "copy",
        "PATH": "/root/.local/bin:/usr/local/cuda/bin:${PATH}",
    })
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        f"cd {REMOTE} && git clone --depth 1 -b {SKYRL_REF} https://github.com/NovaSky-AI/SkyRL.git",
        f"cd {REMOTE} && git clone --depth 1 -b {COOKBOOK_REF} "
        "https://github.com/thinking-machines-lab/tinker-cookbook.git",
        # Resolve SkyRL with the tinker + megatron extras (multi-LoRA needs megatron).
        f"cd {REMOTE}/SkyRL && uv sync --extra tinker --extra megatron",
        # The math-rl client recipe lives in tinker-cookbook.
        f"cd {REMOTE}/tinker-cookbook && uv sync --extra math-rl",
        gpu="any",
    )
)

app = modal.App("tl-skyrl-multilora")


def _server_backend_config(n_loras: int, infer_gpus: int) -> str:
    """Minimal Megatron + vLLM multi-LoRA backend config (scaled down from the
    field-report 8-GPU recipe). colocate_all=False => train + infer on
    separate GPUs; bump tensor_model_parallel_size for bigger models."""
    return json.dumps({
        "strategy": "megatron",
        "trainer.placement.colocate_all": False,
        "trainer.placement.policy_num_gpus_per_node": 1,
        "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
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


@app.function(image=image, gpu="H100:2", timeout=60 * 60,
              volumes={HF_CACHE: hf_volume})
def run_multilora(
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    n_loras: int = 2,
    max_steps: int = 2,
    infer_gpus: int = 1,
    server_warmup_s: int = 1200,
) -> dict:
    import os
    import signal
    import subprocess
    import threading
    import time
    import urllib.request

    os.makedirs(LORA_SYNC, exist_ok=True)
    env = {**os.environ, "HOME": "/root", "TINKER_API_KEY": "tml-dummy",
           "TINKER_BASE_URL": "http://127.0.0.1:8000"}
    skyrl = f"{REMOTE}/SkyRL"
    cookbook = f"{REMOTE}/tinker-cookbook"

    # ---- 1. Launch the SkyRL multi-LoRA Tinker server ----
    server_log: list[str] = []
    server_cmd = [
        "uv", "run", "--extra", "tinker", "--extra", "megatron",
        "-m", "skyrl.tinker.api",
        "--base-model", model, "--backend", "megatron", "--port", "8000",
        "--backend-config", _server_backend_config(n_loras, infer_gpus),
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
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", n_loras: int = 2,
         max_steps: int = 2, gpu: str = "H100:2", infer_gpus: int = 1):
    res = run_multilora.with_options(gpu=gpu).remote(
        model=model, n_loras=n_loras, max_steps=max_steps, infer_gpus=infer_gpus)
    print("\n==== RESULT ====")
    for k, v in res.items():
        if k.endswith("_tail"):
            print(f"\n--- {k} ---\n{v}")
        else:
            print(f"{k}: {v}")

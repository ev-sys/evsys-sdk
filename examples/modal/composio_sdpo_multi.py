"""Composio multi-arm SDPO on Modal SkyRL — small validation.

Runs N concurrent SDPO arms against ONE SkyRL Tinker server. Each arm trains
its own LoRA adapter with a different *teacher* prompt template (the student
template is identical across arms). Tests whether giving the teacher extra
"privileged information" (full tool descriptions) produces a sharper top-K
signal that the student can absorb.

Architecture: identical to examples/modal/skyrl_multilora.py — same image
(cached), same server-boot path. The only swap is the client subprocess:
instead of `tinker_cookbook.recipes.math_rl.train`, each arm runs the local
`_composio_sdpo_arm.py` script (mounted via add_local_file).

Knobs are in `composio_sdpo_multi.yaml`. CLI flags can override.

Usage:
    modal run examples/modal/composio_sdpo_multi.py::main \
        --max-steps 5 --rank 8 --batch-size 4 --n-rows 32
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# --- Pinned to the same SkyRL ref / Cookbook ref as PR #9 so the image
# --- layer is reused — no rebuild for the validation run.
SKYRL_REF = "de55355e72a6bcc04f17b971ab211a306900c08b"
COOKBOOK_REF = "main"
EXTRA = "megatron"
REMOTE = "/root"
HF_CACHE = "/root/.cache/huggingface"
LORA_SYNC = "/tmp/lora_sync/composio_sdpo"
DATA_MOUNT = "/data"

HERE = Path(__file__).resolve().parent
ARM_SCRIPT = HERE / "_composio_sdpo_arm.py"
TOPK_PATCH = HERE / "patch_skyrl_topk.sh"

hf_volume = modal.Volume.from_name("skyrl-hf-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("trajectory-data", create_if_missing=False)

# Reuse the exact image layers from skyrl_multilora — must match byte-for-byte
# so the cache hits.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential", "ca-certificates", "libnuma1", "numactl")
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
        f"cd {REMOTE} && git clone https://github.com/NovaSky-AI/SkyRL.git "
        f"&& cd SkyRL && git checkout {SKYRL_REF}",
        f"cd {REMOTE} && git clone https://github.com/thinking-machines-lab/tinker-cookbook.git "
        f"&& cd tinker-cookbook && git checkout {COOKBOOK_REF}",
        f"cd {REMOTE}/SkyRL && uv sync --extra tinker --extra {EXTRA}",
        f"cd {REMOTE}/tinker-cookbook && uv sync --extra math-rl",
        gpu="any",
    )
    # PG-gate fix for 1+1 actor init (same patch as PR #9).
    .run_commands(
        "find /root/SkyRL -path '*/skyrl/backends/skyrl_train/workers/worker.py' "
        "-exec sed -i 's|if raw_pg is None and self._num_gpus_per_node > 1:|"
        "if raw_pg is None:|' {} +",
    )
    # topk_prompt_logprobs plumbing patch. The cookbook's SDFT recipe calls
    # `teacher_client.sample_async(..., topk_prompt_logprobs=K)` to recover
    # the teacher's top-K distribution at each prompt position. Stock SkyRL
    # accepts the field but silently drops it before reaching vLLM (which
    # natively supports prompt_logprobs=K via its OpenAI extension). The patch
    # threads the field through SampleInput → api.py → forwarding payload and
    # parses vLLM's response back onto SampleOutput.
    .add_local_file(str(TOPK_PATCH), remote_path=f"{REMOTE}/patch_skyrl_topk.sh",
                    copy=True)
    .run_commands(f"bash {REMOTE}/patch_skyrl_topk.sh {REMOTE}/SkyRL")
    # Ship the arm script into the image so subprocesses can launch it.
    .add_local_file(str(ARM_SCRIPT), remote_path=f"{REMOTE}/_composio_sdpo_arm.py",
                    copy=True)
)

app = modal.App("tl-composio-sdpo-multi")


def _server_backend_config(n_arms: int, train_gpus: int, infer_gpus: int) -> str:
    return json.dumps({
        "strategy": "megatron",
        "trainer.placement.colocate_all": False,
        "trainer.placement.policy_num_gpus_per_node": train_gpus,
        "trainer.policy.megatron_config.tensor_model_parallel_size": train_gpus,
        "trainer.policy.megatron_config.lora_config.merge_lora": False,
        "trainer.micro_train_batch_size_per_gpu": 8,
        "trainer.micro_forward_batch_size_per_gpu": 8,
        "trainer.policy.model.lora.max_loras": n_arms,
        "trainer.policy.model.lora.max_cpu_loras": n_arms,
        "trainer.policy.model.lora.lora_sync_path": LORA_SYNC,
        "generator.inference_engine.run_engines_locally": True,
        "generator.inference_engine.num_engines": 1,
        "generator.inference_engine.tensor_parallel_size": infer_gpus,
        "generator.inference_engine.gpu_memory_utilization": 0.8,
        "generator.inference_engine.max_num_seqs": 128,
    })


@app.function(
    image=image, gpu="H100:2", timeout=60 * 60,
    volumes={HF_CACHE: hf_volume, DATA_MOUNT: data_volume},
)
def run_validation(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    arms: list[str] = ["vanilla", "desc"],  # noqa: B006 — explicit default arms
    rank: int = 8,
    max_steps: int = 5,
    batch_size: int = 4,
    max_tokens: int = 128,
    n_rows: int = 32,
    learning_rate: float = 1e-4,
    max_desc_tools: int = 20,
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
    env = {
        **os.environ, "HOME": "/root", "TINKER_API_KEY": "tml-dummy",
        "TINKER_BASE_URL": "http://127.0.0.1:8000",
        "SKYRL_DUMP_INFRA_LOG_TO_STDOUT": "1",
        "CUDNN_PATH": "/usr/lib/x86_64-linux-gnu",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
    }
    skyrl = f"{REMOTE}/SkyRL"
    cookbook = f"{REMOTE}/tinker-cookbook"

    # ---- 1. SkyRL Tinker server ----
    server_log: list[str] = []
    n_arms = len(arms)
    server_cmd = [
        "uv", "run", "--extra", "tinker", "--extra", EXTRA,
        "-m", "skyrl.tinker.api",
        "--base-model", model, "--backend", EXTRA, "--port", "8000",
        "--backend-config", _server_backend_config(n_arms, train_gpus, infer_gpus),
    ]
    print(">>> [modal] starting SkyRL multi-LoRA server", flush=True)
    server = subprocess.Popen(server_cmd, cwd=skyrl, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              start_new_session=True)

    def _pump(stream, tag, sink):
        for line in stream:
            sink.append(line)
            print(f"[{tag}] {line}", end="", flush=True)
    threading.Thread(target=_pump, args=(server.stdout, "server", server_log),
                     daemon=True).start()

    def _healthy() -> bool:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/api/v1/healthz", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    deadline = time.time() + server_warmup_s
    while time.time() < deadline:
        if server.poll() is not None:
            return {"status": "failed", "stage": "server_boot",
                    "server_tail": "".join(server_log[-80:])}
        if _healthy():
            break
        time.sleep(5)
    else:
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        return {"status": "failed", "stage": "healthz_timeout",
                "server_tail": "".join(server_log[-80:])}
    print(">>> [modal] server healthy; launching SDPO arms", flush=True)

    # ---- 2. Spawn N arm subprocesses ----
    clients = []
    for i, arm in enumerate(arms):
        log_path = f"/tmp/sdpo_arm_{arm}"
        cmd = [
            "uv", "run", "--extra", "math-rl",
            "python", f"{REMOTE}/_composio_sdpo_arm.py",
            "--arm", arm,
            "--base-url", env["TINKER_BASE_URL"],
            "--model", model,
            "--data-path", f"{DATA_MOUNT}/composio/sft_train_4000.jsonl",
            "--rank", str(rank),
            "--learning-rate", str(learning_rate),
            "--max-steps", str(max_steps),
            "--batch-size", str(batch_size),
            "--max-tokens", str(max_tokens),
            "--n-rows", str(n_rows),
            "--max-desc-tools", str(max_desc_tools),
            "--log-path", log_path,
            "--seed", str(i),
        ]
        log: list[str] = []
        # We run under tinker-cookbook's venv so the tinker SDK + cookbook
        # helpers (build_topk_distillation_datums) resolve.
        p = subprocess.Popen(cmd, cwd=cookbook, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        threading.Thread(target=_pump, args=(p.stdout, f"arm-{arm}", log),
                         daemon=True).start()
        clients.append((arm, p, log, log_path))
        time.sleep(3)  # stagger to avoid create_model race

    results: dict = {"status": "completed", "arms": {}}
    for arm, p, log, log_path in clients:
        rc = p.wait()
        # Drain metrics.jsonl if present
        metrics_path = Path(log_path) / "metrics.jsonl"
        per_step = []
        if metrics_path.exists():
            for line in metrics_path.read_text().splitlines():
                if line.strip():
                    per_step.append(json.loads(line))
        results["arms"][arm] = {
            "rc": rc,
            "n_steps_logged": len(per_step),
            "first_step": per_step[0] if per_step else None,
            "last_step": per_step[-1] if per_step else None,
            "stdout_tail": "".join(log[-40:]),
        }

    # ---- 3. Clean up server ----
    try:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass

    return results


@app.local_entrypoint()
def main(
    max_steps: int = 5,
    rank: int = 8,
    batch_size: int = 4,
    n_rows: int = 32,
    max_tokens: int = 128,
):
    res = run_validation.remote(
        max_steps=max_steps, rank=rank, batch_size=batch_size,
        n_rows=n_rows, max_tokens=max_tokens,
    )
    print("\n==== RESULT ====")
    print(json.dumps(res, indent=2, default=str)[:5000])

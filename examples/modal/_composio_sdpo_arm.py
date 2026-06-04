"""One SDPO arm: multi-LoRA top-K distillation against a Tinker endpoint.

Runs as a subprocess inside the Modal container (see composio_sdpo_multi.py).
Talks to the local SkyRL Tinker server at $TINKER_BASE_URL using the tinker
SDK, so the same script also works against the hosted Tinker service.

What "arm" means: each arm trains its own LoRA adapter, with a different
*teacher prompt template*. The student prompt is always the same
(`system + tool_names + query`); only the teacher's privileged-information
context differs:

  --arm vanilla : teacher sees the same prompt as the student plus the gold
                  tool name as a worked example (no extra information beyond
                  what the student gets).
  --arm desc    : teacher additionally sees the FULL descriptions of every
                  tool in the row's toolkit, then the gold tool. Tests whether
                  this privileged context produces a sharper top-K signal that
                  the student can absorb.

Loss: top-K forward-KL distillation over the gold response tokens (offline /
teacher-forced — student doesn't roll out; we use the dataset's gold completion
as the "student completion" the teacher scores). This is enough to validate the
multi-arm pipeline; on-policy rollout SDFT is a v2.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tinker
from tinker_cookbook.distillation.sdft import build_topk_distillation_datums


# Templates: only the teacher diff matters. Student template is shared.
DEMO_TEMPLATES = {
    "vanilla": (
        "{question}\n\n"
        "Reference answer:\n{golden_answer}\n\n"
        "Now produce your own answer."
    ),
    "desc": (
        "{question}\n\n"
        "Tool descriptions (privileged information):\n{golden_answer}\n\n"
        "Now produce your own answer."
    ),
}

ANSWER_RE = re.compile(r"<answer>([^<]+)</answer>")


def _toolkit_description_map(rows: list[dict]) -> dict[str, dict[str, str]]:
    """Aggregate {toolkit: {tool_slug: description}} from the rows.

    Descriptions are embedded in each row's assistant `<think>...</think>` after
    'because:'; we recover them so the `desc` arm can present descriptions of
    every tool in the toolkit (not just the gold tool) as privileged info.
    """
    by_toolkit: dict[str, dict[str, str]] = defaultdict(dict)
    think_re = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    for r in rows:
        msgs = r["messages"]
        asst = msgs[-1]["content"]
        am = ANSWER_RE.search(asst)
        tm = think_re.search(asst)
        if not (am and tm):
            continue
        tool = am.group(1).strip()
        toolkit = tool.split("_", 1)[0]  # SLACK_..., OUTLOOK_..., etc.
        think = tm.group(1)
        # "because:" splits "<gold> fits because: <description>"
        if "because:" in think:
            desc = think.split("because:", 1)[1].strip()
            by_toolkit[toolkit].setdefault(tool, desc)
    return dict(by_toolkit)


def _record_for_row(
    row: dict, arm: str, tool2desc: dict[str, dict[str, str]],
    max_desc_tools: int,
) -> dict[str, Any] | None:
    """Convert a composio row into an SDPO record with arm-specific feedback."""
    msgs = row["messages"]
    if len(msgs) < 3 or msgs[-1]["role"] != "assistant":
        return None
    asst = msgs[-1]["content"]
    am = ANSWER_RE.search(asst)
    if not am:
        return None
    tool = am.group(1).strip()
    toolkit = tool.split("_", 1)[0]

    student_messages = msgs[:-1]
    response = asst
    question = msgs[1]["content"] if len(msgs) >= 2 else ""

    if arm == "vanilla":
        feedback = asst  # gold response, no extra info
    elif arm == "desc":
        kit_descs = tool2desc.get(toolkit, {})
        if not kit_descs:
            return None
        items = sorted(kit_descs.items())[:max_desc_tools]
        body = "\n".join(f"{slug}: {desc}" for slug, desc in items)
        feedback = f"{body}\n\nGold tool: <answer>{tool}</answer>"
    else:
        raise ValueError(f"unknown arm: {arm}")

    return {
        "response": response,
        "student_messages": student_messages,
        "question": question,
        "feedback": feedback,
        "tool": tool,
        "toolkit": toolkit,
    }


def _student_datum(
    tokenizer, rec: dict, demo_template: str, max_tokens: int,
) -> tuple[Any, Any] | tuple[None, None]:
    """Build (student_datum, teacher_prompt) for one record."""
    def _ids(msgs, add_gen):
        out = tokenizer.apply_chat_template(msgs, add_generation_prompt=add_gen, tokenize=True)
        if hasattr(out, "input_ids"):
            out = out.input_ids
        elif hasattr(out, "keys"):
            out = out["input_ids"]
        out = list(out)
        if out and isinstance(out[0], (list, tuple)):
            out = out[0]
        return [int(t) for t in out]

    history = rec["student_messages"]
    prompt_ids = _ids(history, True)
    full_ids = _ids([*history, {"role": "assistant", "content": rec["response"]}], False)
    if list(full_ids[: len(prompt_ids)]) != list(prompt_ids):
        full_ids = list(prompt_ids) + tokenizer.encode(rec["response"])
    completion_ids = list(full_ids[len(prompt_ids):])[:max_tokens]
    if not completion_ids:
        return None, None
    full_ids = list(prompt_ids) + completion_ids
    mask = [0.0] * (len(prompt_ids) - 1) + [1.0] * len(completion_ids)
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full_ids[:-1]),
        loss_fn_inputs={
            "target_tokens": np.asarray(full_ids[1:], dtype=np.int64),
            "mask": np.asarray(mask, dtype=np.float32),
        },
    )
    teacher_user = demo_template.format(question=rec["question"], golden_answer=rec["feedback"])
    t_ids = _ids([{"role": "user", "content": teacher_user}], True)
    return datum, tinker.ModelInput.from_ints(t_ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(DEMO_TEMPLATES))
    ap.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data-path", default="/data/composio/sft_train_4000.jsonl")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-desc-tools", type=int, default=30,
                    help="Cap on tools whose descriptions enter the teacher's priv-info "
                         "for the `desc` arm. Keeps the teacher prompt under context.")
    ap.add_argument("--n-rows", type=int, default=None,
                    help="Truncate dataset to this many rows (validation runs).")
    ap.add_argument("--log-path", default="/tmp/sdpo_arm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arm = args.arm
    log_dir = Path(args.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.jsonl"

    # 1) Load composio data
    rows: list[dict] = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.n_rows:
        rows = rows[: args.n_rows]
    if not rows:
        print(f"[arm={arm}] no rows loaded from {args.data_path}", flush=True)
        return 2

    # 2) Build the {toolkit: {tool: description}} map from the corpus
    tool2desc = _toolkit_description_map(rows)
    print(f"[arm={arm}] loaded {len(rows)} rows, "
          f"{sum(len(v) for v in tool2desc.values())} unique tools across "
          f"{len(tool2desc)} toolkits", flush=True)

    # 3) Build per-row records with arm-specific feedback
    records: list[dict] = []
    for r in rows:
        rec = _record_for_row(r, arm, tool2desc, args.max_desc_tools)
        if rec is not None:
            records.append(rec)
    if not records:
        print(f"[arm={arm}] no valid records (after arm filter)", flush=True)
        return 2
    print(f"[arm={arm}] {len(records)} valid records", flush=True)

    # 4) Connect to the Tinker endpoint, create training + teacher clients
    if "TINKER_API_KEY" not in os.environ:
        os.environ["TINKER_API_KEY"] = "tml-dummy"  # SkyRL ignores; hosted Tinker needs it
    service_client = tinker.ServiceClient(base_url=args.base_url)
    print(f"[arm={arm}] connecting to {args.base_url}", flush=True)
    training_client = service_client.create_lora_training_client(args.model, rank=args.rank)
    teacher_client = service_client.create_sampling_client(base_model=args.model)
    tokenizer = training_client.get_tokenizer()
    vocab = len(tokenizer)
    print(f"[arm={arm}] training_client + teacher_client ready (rank={args.rank}, vocab={vocab})",
          flush=True)

    # 4b) Kickstart the vLLM inference engine. SkyRL's non-colocated forwarding
    #     path serves /asample directly to vLLM via a proxy URL it reads from
    #     EngineStateDB. But the URL is only written when the engine
    #     subprocess actually starts vLLM, which happens lazily on the first
    #     forward_backward / sample / save_sampler_checkpoint call routed
    #     through the engine queue (NOT through the forwarder). Sample calls
    #     bypass the queue and so cannot kickstart vLLM themselves —
    #     deadlock. save_weights_for_sampler is a queue-routed call that hits
    #     backend.save_sampler_checkpoint → _ensure_inference_engines, which
    #     publishes the proxy URL. After it completes, sample_async works.
    print(f"[arm={arm}] kickstarting vLLM via save_weights_for_sampler...", flush=True)
    t_kickstart = time.time()
    _ = training_client.save_weights_for_sampler(f"arm-{arm}-kickstart").result()
    print(f"[arm={arm}] vLLM kickstart complete in {time.time()-t_kickstart:.1f}s", flush=True)

    # 5) Training loop
    adam = tinker.AdamParams(learning_rate=args.learning_rate)
    rng = random.Random(args.seed)
    demo_template = DEMO_TEMPLATES[arm]
    t_start = time.time()
    for step in range(args.max_steps):
        batch = rng.sample(records, min(args.batch_size, len(records)))
        data_D, meta_D, teacher_prompts = [], [], []
        for rec in batch:
            datum, tprompt = _student_datum(tokenizer, rec, demo_template, args.max_tokens)
            if datum is None:
                continue
            teacher_prompts.append(tprompt)
            meta_D.append({"group_idx": len(teacher_prompts) - 1})
            data_D.append(datum)
        if not data_D:
            print(f"[arm={arm}] step {step}: empty batch (skipped)", flush=True)
            continue

        # Soft top-K forward-KL distillation. The cookbook builds datums with
        # (N, K) target_tokens + weights = renormalized teacher top-K probs.
        # Our SkyRL multi-target CE patch (see patch_skyrl_topk.sh) detects
        # these shapes and computes
        #   L = -sum_k weights[N, K] * log p_student(target[N, K])
        # by re-gathering the student's logprobs K times from the same model
        # forward.
        ce_datums, build_m = asyncio.run(build_topk_distillation_datums(
            data_D, meta_D, teacher_client=teacher_client,
            teacher_prompts_P=teacher_prompts, topk=args.topk,
            vocab_size=vocab, skip_first_n_tokens=3,
        ))

        fb = training_client.forward_backward(ce_datums, "cross_entropy").result()
        training_client.optim_step(adam).result()

        metrics = {
            "arm": arm,
            "step": step,
            "elapsed_s": round(time.time() - t_start, 2),
            "batch_size": len(data_D),
            **{k: float(v) for k, v in (build_m or {}).items() if isinstance(v, (int, float))},
            **{k: float(v) for k, v in getattr(fb, "metrics", {}).items() if isinstance(v, (int, float))},
        }
        print(f"[arm={arm}] {metrics}", flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    print(f"[arm={arm}] DONE in {time.time() - t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

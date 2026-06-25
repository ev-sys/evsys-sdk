"""Harness 1 — the RAW BFCL baseline. Does NOT import ``evsys_sdk`` or ``harbor``.

This is the independent baseline the SDK harness (``run_bfcl.py``) is validated
against. It evaluates the SAME 500 tasks under IDENTICAL conditions by reusing
the shared :mod:`bfcl_core` for the prompt, the sampling params, the tool-call
parse, and the scoring — and by sampling from the Tinker checkpoint with the
exact same ``tinker_cookbook`` renderer + ``tinker.SamplingParams`` that harbor's
``TinkerLLM`` uses (see ``harbor/llms/tinker.py``), reconstructed here WITHOUT
importing harbor:

  per task:
    bfcl_core.build_messages(task)                    # [system(tool block), user(query)]
      → renderer.build_generation_prompt(messages)    # tinker_cookbook "qwen3"
      → sampling_client.sample_async(...)             # tinker, ROLLOUT_PARAMS + FIXED
      → renderer.parse_response(tokens) → text
      → bfcl_core.parse_tool_calls(text)              # shared hermes parser
      → bfcl_core.score(text, expected, {})           # shared faithful AST scorer

Run::

    python benchmarks/bfcl/evaluate.py --model-path tinker://<checkpoint>
    python benchmarks/bfcl/evaluate.py --limit 20            # base Qwen3-4B, 20 tasks

Needs ``TINKER_API_KEY`` set (it issues real sampling calls). The offline render
+ scoring paths are exercised key-free by ``parity_check.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bfcl_core  # noqa: E402 — the shared single source of truth


def _build_renderer(model_name: str, renderer_name: str):
    """The SAME renderer harbor's TinkerLLM builds: get_renderer(name, tokenizer)."""
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model_name)
    return get_renderer(renderer_name, tokenizer)


async def _make_sampling_client(model_name: str, model_path: str | None):
    """Mirror TinkerLLM._ensure_client: a tinker SamplingClient on the checkpoint
    (``model_path``) or the base model. No harbor."""
    import tinker

    service = tinker.ServiceClient()
    if model_path:
        return await service.create_sampling_client_async(model_path=model_path)
    return await service.create_sampling_client_async(base_model=model_name)


async def _sample_one(client, renderer, messages, max_tokens, temperature):
    """One rollout, mirroring TinkerLLM.call's body (sans harbor wrappers)."""
    import tinker

    model_input = renderer.build_generation_prompt(messages)
    stop_sequences = renderer.get_stop_sequences()  # same stop harbor uses
    params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_sequences,
        seed=bfcl_core.FIXED_SAMPLING["seed"],
        top_k=bfcl_core.FIXED_SAMPLING["top_k"],
        top_p=bfcl_core.FIXED_SAMPLING["top_p"],
    )
    resp = await client.sample_async(prompt=model_input, num_samples=1, sampling_params=params)
    tokens = resp.sequences[0].tokens
    parsed, _ok = renderer.parse_response(tokens)
    return parsed.get("content", "") or ""


async def _run(args: argparse.Namespace) -> None:
    tasks = bfcl_core.load_tasks(args.bench_dir)
    if args.limit is not None:
        tasks = tasks[: max(0, int(args.limit))]
    print(f"[raw] loaded {len(tasks)} tasks from {args.bench_dir}")
    print(f"[raw] model_name={args.model_name} model_path={args.model_path or '(base)'}")
    print(f"[raw] renderer={bfcl_core.RENDERER_NAME} params={bfcl_core.ROLLOUT_PARAMS} "
          f"fixed={bfcl_core.FIXED_SAMPLING}")

    renderer = _build_renderer(args.model_name, bfcl_core.RENDERER_NAME)
    client = await _make_sampling_client(args.model_name, args.model_path)

    max_tokens = bfcl_core.ROLLOUT_PARAMS["max_tokens"]
    temperature = bfcl_core.ROLLOUT_PARAMS["temperature"]

    sem = asyncio.Semaphore(args.n_concurrent)
    rewards: list[float] = [0.0] * len(tasks)

    async def _do(i: int, task: dict) -> None:
        async with sem:
            messages = bfcl_core.build_messages(task, args.model_name)
            completion = await _sample_one(client, renderer, messages, max_tokens, temperature)
            expected = bfcl_core._task_expected(task)
            rewards[i] = float(bfcl_core.score(completion, expected, {}))

    await asyncio.gather(*(_do(i, t) for i, t in enumerate(tasks)))

    overall = sum(rewards) / len(rewards) if rewards else 0.0
    by_cat: dict[str, list[float]] = defaultdict(list)
    for t, r in zip(tasks, rewards):
        by_cat[t["metadata"]["category"]].append(r)

    print(f"\n=== Overall (RAW harness) ===  accuracy={overall:.3f}  n={len(rewards)}")
    print("\n=== Per category ===")
    for name in sorted(by_cat):
        rs = by_cat[name]
        print(f"  {name:26s} n={len(rs):4d}  pass_rate={sum(rs) / len(rs):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", default=bfcl_core.DEFAULT_MODEL_NAME)
    ap.add_argument("--model-path", default=None,
                    help="Tinker checkpoint to score; omit for the base model.")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap tasks scored (first N) for a fast smoke run.")
    ap.add_argument("--n-concurrent", type=int, default=8)
    ap.add_argument("--bench-dir", type=Path, default=bfcl_core.BENCH_DIR)
    args = ap.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

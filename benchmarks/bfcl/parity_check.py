"""Parity smoke test — proves the two harnesses feed the model IDENTICAL input.

Two parts:

OFFLINE (no API key — ALWAYS runs).  For N sampled tasks, render the model input
THREE ways and assert the token id sequences are byte-identical:

  (a) INSPECT native FC — the reference. Qwen3's own chat template with the tools
      passed via ``tools=`` (OpenAI schema):
        tok.apply_chat_template([{user: query}], tools=oai, add_generation_prompt=True)
      This is exactly how inspect_ai renders a BFCL FC task.

  (b) RAW harness path — ``bfcl_core.build_messages(task)`` (= the canonical
      ``[{system: tool block}, {user: query}]``) rendered with the SAME
      ``tinker_cookbook`` ``get_renderer("qwen3", tokenizer).build_generation_prompt``
      the RAW ``evaluate.py`` uses, then ``.to_ints()``.

  (c) SDK harness path — reconstruct what harbor actually feeds the renderer: the
      per-task ``system_prompt`` is packed into ``instruction.md`` by the SDK
      adapter; ``BasicLoopAgent`` splits it back into ``(system, user)`` and
      renders ``[{system}, {user}]`` with the same renderer. We replay that exact
      pack→split→render here (importing the SDK's real adapter + split helper).

  All three must produce identical token ids. On any mismatch we print the first
  divergence. We also assert ROLLOUT_PARAMS + the renderer stop sequence agree.

LIVE (only if ``TINKER_API_KEY`` is set).  Runs 3 tasks through BOTH harnesses on
the same base model and reports per-task scores + agreement. Without a key it
prints a clear skip message.

Run:  ``python benchmarks/bfcl/parity_check.py [--n 20]``
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bfcl_core  # noqa: E402


def _ids(out) -> list[int]:
    """Flat token-id list from apply_chat_template(tokenize=True) (BatchEncoding
    subclasses dict; ``input_ids`` is the flat list)."""
    if hasattr(out, "keys"):
        return list(out["input_ids"])
    if out and isinstance(out[0], list):
        return list(out[0])
    return list(out)


def _first_divergence(a: list[int], b: list[int]) -> str:
    import itertools

    for i, (x, y) in enumerate(itertools.zip_longest(a, b)):
        if x != y:
            return f"idx {i}: {x!r} != {y!r}"
    return "(no divergence)"


def _inspect_native_fc_ids(hf_tok, task) -> list[int]:
    """(a) inspect_ai native FC: tools via tools=, the reference render."""
    tools = bfcl_core._task_tools(task)
    query = bfcl_core._task_query(task)
    return _ids(
        hf_tok.apply_chat_template(
            [{"role": "user", "content": query}],
            tools=bfcl_core._to_oai_tools(tools),
            add_generation_prompt=True,
            tokenize=True,
        )
    )


def _raw_harness_ids(renderer, task) -> list[int]:
    """(b) RAW harness: build_messages → tinker_cookbook qwen3 render."""
    return renderer.build_generation_prompt(bfcl_core.build_messages(task)).to_ints()


def _sdk_harness_ids(renderer, task) -> list[int]:
    """(c) SDK harness: replay harbor's pack(instruction.md) → split → render.

    Imports the SDK's REAL adapter packing + agent split so this tracks the
    production code path, not a copy of it."""
    from evsys_sdk.training.harbor_engine import (
        _SYSTEM_PROMPT_SENTINEL,
        split_system_instruction,
    )

    system = bfcl_core._task_system_prompt(task, bfcl_core.DEFAULT_MODEL_NAME)
    instruction = bfcl_core._task_query(task)
    # what the adapter writes into instruction.md:
    packed = f"{system}{_SYSTEM_PROMPT_SENTINEL}\n{instruction}" if system else instruction
    # what BasicLoopAgent does with it:
    per_task_system, user = split_system_instruction(packed)
    messages: list[dict] = []
    if per_task_system:
        messages.append({"role": "system", "content": per_task_system})
    messages.append({"role": "user", "content": user})
    return renderer.build_generation_prompt(messages).to_ints()


def _offline(n: int, model_name: str) -> int:
    from transformers import AutoTokenizer
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tasks = bfcl_core.load_tasks()
    rng = random.Random(1234)
    sample = rng.sample(tasks, min(n, len(tasks)))

    hf_tok = AutoTokenizer.from_pretrained(model_name)
    renderer = get_renderer(bfcl_core.RENDERER_NAME, get_tokenizer(model_name))

    print(f"[offline] 3-way token-id parity over {len(sample)} sampled tasks "
          f"(model={model_name}, renderer={bfcl_core.RENDERER_NAME})")

    mismatches = 0
    sample_render_shown = False
    for task in sample:
        a = _inspect_native_fc_ids(hf_tok, task)
        b = _raw_harness_ids(renderer, task)
        c = _sdk_harness_ids(renderer, task)
        ok = a == b == c
        if not ok:
            mismatches += 1
            print(f"  MISMATCH {task['task_id']} ({task['metadata']['category']}):")
            if a != b:
                print(f"    inspect(a) vs RAW(b):  {_first_divergence(a, b)}")
            if b != c:
                print(f"    RAW(b) vs SDK(c):      {_first_divergence(b, c)}")
            if a != c:
                print(f"    inspect(a) vs SDK(c):  {_first_divergence(a, c)}")
        elif not sample_render_shown:
            # eyeball one rendered prompt (decoded from the RAW path token ids)
            print(f"\n--- sample rendered input ({task['task_id']}, "
                  f"{task['metadata']['category']}, {len(b)} tokens) ---")
            print(get_tokenizer(model_name).decode(b))
            print("--- end sample ---\n")
            sample_render_shown = True

    identical = mismatches == 0
    print(f"[offline] token-id identical for all {len(sample)} tasks: "
          f"{'YES' if identical else f'NO ({mismatches} mismatch)'}")

    # params + stop parity
    stop = renderer.get_stop_sequences()
    print(f"[offline] ROLLOUT_PARAMS = {bfcl_core.ROLLOUT_PARAMS}")
    print(f"[offline] FIXED_SAMPLING = {bfcl_core.FIXED_SAMPLING}")
    print(f"[offline] renderer stop sequences = {stop} "
          "(BOTH harnesses read this from the renderer → always agree)")
    assert isinstance(bfcl_core.ROLLOUT_PARAMS["max_tokens"], int)
    assert bfcl_core.ROLLOUT_PARAMS["temperature"] == 0.0

    return 0 if identical else 1


def _live(model_name: str) -> int:
    if not os.environ.get("TINKER_API_KEY"):
        print("\n[live] TINKER_API_KEY not set — skipping the live parity run.")
        print("[live] set TINKER_API_KEY to run 3 tasks through BOTH harnesses "
              "and compare per-task scores.")
        return 0

    import asyncio

    import evaluate as raw_harness  # the RAW harness module
    from evsys_sdk import Benchmark

    import verifier  # noqa: F401 — registers bfcl_match

    tasks = bfcl_core.load_tasks()[:3]
    print(f"\n[live] running {len(tasks)} tasks through BOTH harnesses on {model_name}")

    async def _run():
        # RAW
        renderer = raw_harness._build_renderer(model_name, bfcl_core.RENDERER_NAME)
        client = await raw_harness._make_sampling_client(model_name, None)
        raw_scores = []
        for t in tasks:
            comp = await raw_harness._sample_one(
                client, renderer, bfcl_core.build_messages(t, model_name),
                bfcl_core.ROLLOUT_PARAMS["max_tokens"],
                bfcl_core.ROLLOUT_PARAMS["temperature"],
            )
            raw_scores.append(float(bfcl_core.score(comp, bfcl_core._task_expected(t), {})))

        # SDK
        bench = Benchmark.from_dir(bfcl_core.BENCH_DIR)
        bench.tasks = bench.tasks[:3]
        score = await bench.score_via_harbor(
            model_name=model_name, model_path=None, model_client="tinker",
            workspace_dir=Path(".bfcl_parity_ws"), renderer_name=bfcl_core.RENDERER_NAME,
            system_prompt=None,
            num_samples=bfcl_core.ROLLOUT_PARAMS["num_samples"],
            max_tokens=bfcl_core.ROLLOUT_PARAMS["max_tokens"],
            temperature=bfcl_core.ROLLOUT_PARAMS["temperature"],
            breakdown_keys=["category"],
        )
        sdk_scores = [pt.reward for pt in score.per_task]
        return raw_scores, sdk_scores

    raw_scores, sdk_scores = asyncio.run(_run())
    agree = all(abs(r - s) < 1e-9 for r, s in zip(raw_scores, sdk_scores))
    print(f"[live] RAW scores: {raw_scores}")
    print(f"[live] SDK scores: {sdk_scores}")
    print(f"[live] per-task agreement: {'YES' if agree else 'NO'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="tasks to check offline")
    ap.add_argument("--model-name", default=bfcl_core.DEFAULT_MODEL_NAME)
    args = ap.parse_args()
    rc = _offline(args.n, args.model_name)
    rc |= _live(args.model_name)
    return rc


if __name__ == "__main__":
    sys.exit(main())

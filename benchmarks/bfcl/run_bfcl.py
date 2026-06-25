"""Harness 2 — the SDK BFCL harness. Scores a checkpoint through the harbor path.

The SDK counterpart of the RAW ``evaluate.py``. It feeds the model the SAME input
as the RAW harness and drives the rollout via :meth:`Benchmark.score_via_harbor`
— ONE harbor job over all tasks.

The parity contract (proved by ``parity_check.py``): every task carries its own
``HarborTask.system_prompt`` (the canonical Qwen3 native-FC tool block for that
task's tools, baked in by ``build_dataset.py``). harbor's ``BasicLoopAgent`` uses
that per-task system prompt as the system message and the task ``instruction`` as
the user turn, then renders ``[{system}, {user}]`` with the SAME
``tinker_cookbook`` ``"qwen3"`` renderer the RAW harness uses. So the rendered
input token ids are byte-identical to ``bfcl_core.build_messages(task)`` — which
is itself byte-identical to inspect_ai's native FC render.

Because the system message is per-task, the job-level ``system_prompt`` is left
``None`` and a SINGLE ``score_via_harbor`` call covers the whole benchmark (no
grouping, no per-task jobs).

Examples::

    python benchmarks/bfcl/run_bfcl.py                       # base Qwen3-4B
    python benchmarks/bfcl/run_bfcl.py --model-path tinker://<checkpoint>
    python benchmarks/bfcl/run_bfcl.py --limit 20            # fast smoke

Requires a Tinker API key + ``harbor`` installed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bfcl_core  # noqa: E402 — shared single source of truth

from evsys_sdk import Benchmark  # noqa: E402

import verifier  # noqa: E402,F401 — importing registers bfcl_match


async def _run(args: argparse.Namespace) -> None:
    bench = Benchmark.from_dir(args.bench_dir)
    print(f"[sdk] loaded {len(bench.tasks)} tasks from {args.bench_dir}")
    print(f"[sdk] model_name={args.model_name} model_path={args.model_path or '(base)'}")
    print(f"[sdk] renderer={bfcl_core.RENDERER_NAME} params={bfcl_core.ROLLOUT_PARAMS}")
    print("[sdk] per-task system_prompt carries each task's tool block → ONE job")

    score = await bench.score_via_harbor(
        model_name=args.model_name,
        model_path=args.model_path,
        model_client="tinker",
        workspace_dir=args.workspace_dir,
        renderer_name=bfcl_core.RENDERER_NAME,
        system_prompt=None,  # per-task HarborTask.system_prompt supplies it
        num_samples=bfcl_core.ROLLOUT_PARAMS["num_samples"],
        max_tokens=bfcl_core.ROLLOUT_PARAMS["max_tokens"],
        temperature=bfcl_core.ROLLOUT_PARAMS["temperature"],
        limit=args.limit,
        breakdown_keys=["category"],
        n_concurrent=args.n_concurrent,
    )

    print("\n=== Overall (SDK harness) ===")
    for k, v in score.metrics.items():
        print(f"  {k:18s} {v}")

    cat = score.breakdowns.get("category", {})
    print("\n=== Per category ===")
    for name in sorted(cat):
        b = cat[name]
        print(f"  {name:26s} n={int(b['n']):4d}  pass_rate={b['pass_rate']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", default=bfcl_core.DEFAULT_MODEL_NAME)
    ap.add_argument("--model-path", default=None,
                    help="Tinker checkpoint to score; omit for the base model.")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap tasks scored (first N) for a fast smoke run.")
    ap.add_argument("--n-concurrent", type=int, default=8)
    ap.add_argument("--bench-dir", type=Path, default=bfcl_core.BENCH_DIR)
    ap.add_argument("--workspace-dir", type=Path, default=Path(".bfcl_workspace"),
                    help="scratch dir harbor writes task dirs + job output to.")
    args = ap.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

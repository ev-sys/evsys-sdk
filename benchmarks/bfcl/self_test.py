"""Offline self-test for the BFCL harness — no Tinker, no harbor, no network.

Proves the ported scorer is faithful by round-tripping the ground truth through
:func:`verifier.bfcl_match`:

  * For ~30 sampled scored tasks (simple / parallel / multiple), synthesize the
    EXACT ground-truth completion (emit hermes ``<tool_call>`` blocks built from
    the first possible answer of each GT call) and assert it scores 1.0; then
    corrupt one call (wrong arg value) and assert 0.0.
  * For irrelevance tasks, assert an empty completion → 1.0 and a spurious call
    → 0.0.  For relevance tasks, assert a call → 1.0 and empty → 0.0.

Also sanity-checks that ``Benchmark.from_dir`` loads all 500 tasks.

Run:  ``python benchmarks/bfcl/self_test.py``
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evsys_sdk import Benchmark  # noqa: E402

import verifier  # noqa: E402,F401 — registers bfcl_match on import
from verifier import bfcl_match  # noqa: E402

BENCH_DIR = Path("data/benchmark/bfcl-noexec-500")
N_SCORED_SAMPLE = 30
SEED = 7


def _first_value(possible: list) -> object:
    """First concrete (non-empty-string) possible value, falling back to ''."""
    for v in possible:
        if v != "":
            return v
    return possible[0] if possible else ""


def _gt_completion(ground_truth: list[dict]) -> str:
    """Build a hermes completion from the first possible value of each GT call."""
    blocks: list[str] = []
    for call in ground_truth:
        name = next(iter(call))
        params = call[name]
        args = {p: _first_value(vals) for p, vals in params.items()}
        # Drop optional params whose only "value" is "" (model would omit them).
        args = {p: v for p, v in args.items() if not (v == "" )}
        blocks.append(json.dumps({"name": name, "arguments": args}))
    return "\n".join(f"<tool_call>{b}</tool_call>" for b in blocks)


def _corrupt_completion(ground_truth: list[dict]) -> str:
    """A wrong call: keep the function name, poison one argument value."""
    call = ground_truth[0]
    name = next(iter(call))
    params = call[name]
    args = {p: _first_value(vals) for p, vals in params.items()}
    if args:
        first = next(iter(args))
        args[first] = "__definitely_wrong_value_zzz__"
    else:
        args = {"__bogus__": "__bogus__"}
    return f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>'


def main() -> int:
    bench = Benchmark.from_dir(BENCH_DIR)
    assert len(bench.tasks) == 500, f"expected 500 tasks, got {len(bench.tasks)}"
    print(f"[self-test] Benchmark.from_dir loaded {len(bench.tasks)} tasks OK")

    by_kind: dict[str, list] = {"scored": [], "irrelevance": [], "relevance": []}
    for t in bench.tasks:
        exp = t.verifier.expected
        cat = exp["category"]
        if "irrelevance" in cat:
            by_kind["irrelevance"].append(exp)
        elif "relevance" in cat:
            by_kind["relevance"].append(exp)
        else:
            by_kind["scored"].append(exp)

    passed = 0
    total = 0
    rng = random.Random(SEED)

    # --- scored categories: GT → 1.0, corrupted → 0.0 ---
    scored = by_kind["scored"]
    rng.shuffle(scored)
    for exp in scored[:N_SCORED_SAMPLE]:
        gt = exp["ground_truth"]
        comp = _gt_completion(gt)
        total += 1
        ok_pos = bfcl_match(comp, exp, {}) == 1.0
        passed += ok_pos
        if not ok_pos:
            print(f"  FAIL[+] {exp['category']} {gt}\n        completion={comp!r}")

        total += 1
        bad = _corrupt_completion(gt)
        ok_neg = bfcl_match(bad, exp, {}) == 0.0
        passed += ok_neg
        if not ok_neg:
            print(f"  FAIL[-] {exp['category']} corrupted scored 1.0: {bad!r}")

    # --- irrelevance: empty → 1.0, spurious call → 0.0 ---
    for exp in by_kind["irrelevance"][:10]:
        total += 1
        passed += bfcl_match("I cannot help with that.", exp, {}) == 1.0
        total += 1
        spurious = '<tool_call>{"name": "foo", "arguments": {"x": 1}}</tool_call>'
        passed += bfcl_match(spurious, exp, {}) == 0.0

    # --- relevance: call → 1.0, empty → 0.0 ---
    for exp in by_kind["relevance"][:5]:
        total += 1
        call = '<tool_call>{"name": "foo", "arguments": {"x": 1}}</tool_call>'
        passed += bfcl_match(call, exp, {}) == 1.0
        total += 1
        passed += bfcl_match("Sorry, no.", exp, {}) == 0.0

    rate = passed / total if total else 0.0
    print(f"[self-test] verifier round-trip: {passed}/{total} = {rate:.1%} pass")
    print(f"[self-test] (scored sampled={min(N_SCORED_SAMPLE, len(scored))}, "
          f"irrelevance={min(10, len(by_kind['irrelevance']))}, "
          f"relevance={min(5, len(by_kind['relevance']))})")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

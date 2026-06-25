"""Build the ``bfcl-noexec-500`` harbor benchmark from the raw BFCL data.

BFCL (Berkeley Function-Call Leaderboard) ships its dataset as per-category
JSONL at the gorilla repo's pinned commit. This script ports inspect_evals'
``record_to_sample`` for the **13 sandbox-free categories** (pure AST + the two
abstention categories — no execution, no SQL/REST, no multi-turn state), draws a
deterministic stratified sample of 500 tasks proportional to each category's
pool size, and writes a harbor-format benchmark dir:

    data/benchmark/bfcl-noexec-500/
        tasks.jsonl       # one HarborTask per line
        metadata.yaml     # provenance + the per-category draw counts

Why build from raw gorilla rather than the harbor-hub ``gorilla/bfcl`` dataset?
The hub dataset IS the same 3,641-task non-sandbox subset (verified by per-
category parity against this raw data), but each hub task is a **Docker
container task**: the model must write ``/app/result.json`` and a ``tests/test.sh``
runs an evaluator *inside the container*. The SDK's :meth:`Benchmark.score_via_harbor`
runs host-side with a ``NoOpEnvironment`` + an in-process verifier — it cannot
run a container ``test.sh``. The hub evaluator is also a simplified matcher
(order-*dependent* for parallel, no int/float language rules), so it is not a
faithful BFCL scorer. We therefore source instructions + ground truth + tool
schemas from raw gorilla and score with the faithful ported :mod:`verifier`.

Each emitted :class:`~evsys_sdk.data_types.HarborTask`:

  * ``instruction`` — JUST the BFCL user query (the tool/function schemas live in
    ``verifier.expected.tools`` and are surfaced as the **system** message by the
    shared :func:`bfcl_core.build_messages`). Splitting the prompt this way is
    what makes the two harnesses byte-identical: the SDK harbor path is given
    ``system_prompt = build_system_prompt(tools)`` and this ``instruction`` as the
    user turn, so harbor's ``[system, user]`` assembly equals ``build_messages``,
    and the RAW harness renders ``build_messages`` directly. The model is asked
    (in the system message) to emit calls as Qwen3 / hermes
    ``<tool_call>{"name": …, "arguments": {…}}</tool_call>`` blocks.
  * ``verifier`` — ``{kind: in_process, fn_name: "bfcl_match",
    expected: {category, ground_truth, tools, language}, params: {}}``.
  * ``metadata`` — ``{category, bfcl_id}`` so ``breakdown_keys=["category"]``
    buckets the score per category.

Run:  ``python benchmarks/bfcl/build_dataset.py [--data-dir <raw gorilla data>]``
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# The 13 sandbox-free categories (pure AST + abstention). Excludes every
# exec_*/rest/sql (need execution) and multi_turn_* (need stateful backends).
CATEGORIES: list[str] = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_relevance",
    "live_irrelevance",
]

# Categories with no possible_answer file — only call *presence* is scored.
ABSTENTION = {"irrelevance", "live_irrelevance", "live_relevance"}

# Gorilla source (pinned, matches inspect_evals + the harbor-hub dataset).
GITHUB_REPO_URL = "https://github.com/ShishirPatil/gorilla.git"
BFCL_GITHUB_COMMIT = "dac44e7ac9db5ff26a01ab0c1ec5de5a1e703b7a"
GITHUB_DATA_PATH = "berkeley-function-call-leaderboard/bfcl_eval/data"

FILENAME_PREFIX = "BFCL_v4_"
FILENAME_SUFFIX = ".json"

SEED = 20240624  # deterministic stratified draw
TARGET_N = 500


# ---------------------------------------------------------------------------
# Language-specific function-doc preprocessing
#
# Ports inspect_evals/bfcl/data.py::_func_doc_language_specific_pre_processing.
# Java / JS function params are flattened to "string" type with a hint appended
# to the description, mirroring the official handler. This matters because the
# verifier's type rules key off the (possibly rewritten) schema type.
# ---------------------------------------------------------------------------


def _language(category: str) -> str:
    if "java" in category and "javascript" not in category:
        return "java"
    if "javascript" in category:
        return "js"
    return "python"


def _language_hint(language: str) -> str:
    if language == "java":
        return " Note that the provided function is in Java 8 SDK syntax."
    if language == "js":
        return " Note that the provided function is in JavaScript syntax."
    return " Note that the provided function is in Python 3 syntax."


def _preprocess_functions(functions: list[dict], language: str) -> list[dict]:
    """Add language hints + flatten Java/JS param types to ``string``."""
    for item in functions:
        item["description"] = item.get("description", "") + _language_hint(language)
        properties = item.get("parameters", {}).get("properties", {})
        if language == "java":
            for _key, value in properties.items():
                if value.get("type") == "any":
                    value["description"] = value.get("description", "") + (
                        " This parameter can be of any type of Java object in string representation."
                    )
                else:
                    value["description"] = value.get("description", "") + (
                        f" This is Java {value.get('type')} type parameter in string representation."
                    )
                if value.get("type") in ("ArrayList", "Array"):
                    value["description"] += (
                        f" The list elements are of type {value['items']['type']}; "
                        "they are not in string representation."
                    )
                    value.pop("items", None)
                value["type"] = "string"
        elif language == "js":
            for _key, value in properties.items():
                if value.get("type") == "any":
                    value["description"] = value.get("description", "") + (
                        " This parameter can be of any type of JavaScript object in string representation."
                    )
                else:
                    value["description"] = value.get("description", "") + (
                        f" This is JavaScript {value.get('type')} type parameter in string representation."
                    )
                if value.get("type") == "array":
                    value["description"] += (
                        f" The list elements are of type {value['items']['type']}; "
                        "they are not in string representation."
                    )
                    value.pop("items", None)
                if value.get("type") == "dict" and "properties" in value:
                    value["description"] += (
                        " The dictionary entries have the following schema; they are not "
                        f"in string representation. {json.dumps(value.get('properties', {}))}"
                    )
                    value.pop("properties", None)
                value["type"] = "string"
    return functions


# ---------------------------------------------------------------------------
# Raw record → HarborTask dict  (ports record_to_sample for the 13 categories)
#
# NOTE: the tools are NOT rendered into the instruction here. The instruction is
# JUST the user query; the tool schemas are carried on ``verifier.expected.tools``
# and surfaced as the SYSTEM message by ``bfcl_core.build_messages`` / the SDK
# harness's ``system_prompt``. This system/user split is the parity contract
# (see bfcl_core.py): both harnesses build the SAME ``[system, user]`` messages.
# ---------------------------------------------------------------------------


def _extract_question(question: Any) -> str:
    """BFCL ``question`` is ``list[list[{role,content}]]`` (one turn here).

    We surface the user turn(s) as plain text. Single-turn categories carry one
    turn; we join any user messages (system content, if present, is prepended).
    """
    # Normalize list[dict] → list[list[dict]] (some live cats store the inner form).
    turns = question
    if turns and isinstance(turns[0], dict):
        turns = [turns]
    parts: list[str] = []
    for turn in turns:
        for msg in turn:
            role, content = msg.get("role"), msg.get("content", "")
            if role == "system":
                parts.append(content)
            elif role in ("user", "assistant"):
                parts.append(content)
    return "\n\n".join(p for p in parts if p)


def record_to_task(record: dict, ground_truth: list, category: str) -> dict:
    """Convert one raw BFCL record (+ its GT) to a HarborTask dict."""
    language = _language(category)
    functions = list(record.get("function") or [])
    functions = _preprocess_functions(functions, language)

    # The instruction is JUST the user query — the tools become the system
    # message downstream (bfcl_core.build_messages / the SDK system_prompt).
    instruction = _extract_question(record.get("question", [])).strip()

    # Abstention categories carry no ground truth (only call presence is scored).
    gt = [] if category in ABSTENTION else list(ground_truth or [])

    # The per-task system message: the canonical Qwen3 native-FC tool block for
    # this task's tools (byte-identical to inspect_ai's FC system message — proved
    # in parity_check.py). The SDK harbor path uses this as the task's
    # system_prompt; the RAW harness renders the same [system, user] messages.
    from bfcl_core import build_system_prompt  # local: shared single source

    return {
        "task_id": f"bfcl-{record['id']}",
        "instruction": instruction,
        "system_prompt": build_system_prompt(functions),
        "verifier": {
            "kind": "in_process",
            "fn_name": "bfcl_match",
            "expected": {
                "category": category,
                "ground_truth": gt,
                "tools": functions,
                "language": language,
            },
            "params": {},
        },
        "metadata": {"category": category, "bfcl_id": record["id"]},
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_category(data_dir: Path, category: str) -> list[dict]:
    """Load all tasks for one category, joined with ground truth, as task dicts."""
    qfile = data_dir / f"{FILENAME_PREFIX}{category}{FILENAME_SUFFIX}"
    records = _load_jsonl(qfile)

    gt_by_id: dict[str, list] = {}
    if category not in ABSTENTION:
        gfile = data_dir / "possible_answer" / f"{FILENAME_PREFIX}{category}{FILENAME_SUFFIX}"
        for g in _load_jsonl(gfile):
            gt_by_id[g["id"]] = g["ground_truth"]

    tasks: list[dict] = []
    for rec in records:
        tasks.append(record_to_task(rec, gt_by_id.get(rec["id"], []), category))
    return tasks


def _ensure_data_dir(data_dir: Path | None) -> Path:
    """Use ``--data-dir`` if it has the BFCL files; else sparse-clone the pin."""
    if data_dir is not None:
        if (data_dir / f"{FILENAME_PREFIX}simple_python{FILENAME_SUFFIX}").is_file():
            return data_dir
        raise FileNotFoundError(
            f"--data-dir {data_dir} has no {FILENAME_PREFIX}simple_python file"
        )

    # Sparse-checkout the pinned commit into a temp dir (matches inspect_evals).
    tmp = Path(tempfile.mkdtemp(prefix="bfcl_gorilla_"))
    print(f"[build] sparse-cloning gorilla@{BFCL_GITHUB_COMMIT[:10]} → {tmp}")
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--sparse", "--no-checkout",
         GITHUB_REPO_URL, str(tmp)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(["git", "checkout", BFCL_GITHUB_COMMIT], cwd=tmp,
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "sparse-checkout", "set", GITHUB_DATA_PATH], cwd=tmp,
                   check=True, capture_output=True, text=True)
    return tmp / GITHUB_DATA_PATH


# ---------------------------------------------------------------------------
# Stratified sampling — proportional to each category's pool size, seeded
# ---------------------------------------------------------------------------


def stratified_counts(pool: dict[str, int], target: int) -> dict[str, int]:
    """Largest-remainder apportionment of ``target`` across categories by pool
    size. Deterministic; every category with a nonzero pool gets ≥ 0, and the
    counts sum to exactly ``target`` (capped at the pool size per category)."""
    total = sum(pool.values())
    raw = {c: target * n / total for c, n in pool.items()}
    floored = {c: int(v) for c, v in raw.items()}
    remainder = target - sum(floored.values())
    # Hand out the leftover seats to the largest fractional parts.
    order = sorted(pool, key=lambda c: raw[c] - floored[c], reverse=True)
    for c in order[:remainder]:
        floored[c] += 1
    # Never draw more than a category actually has.
    for c in floored:
        floored[c] = min(floored[c], pool[c])
    return floored


def build(data_dir: Path | None, out_root: Path) -> dict[str, int]:
    src = _ensure_data_dir(data_dir)
    by_cat: dict[str, list[dict]] = {c: load_category(src, c) for c in CATEGORIES}
    pool = {c: len(v) for c, v in by_cat.items()}
    draw = stratified_counts(pool, TARGET_N)

    rng = random.Random(SEED)
    sampled: list[dict] = []
    for c in CATEGORIES:
        rng.shuffle(by_cat[c])
        sampled.extend(by_cat[c][: draw[c]])
    rng.shuffle(sampled)  # interleave categories so a partial run is balanced

    out_dir = out_root / "bfcl-noexec-500"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tasks.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in sampled) + "\n"
    )

    import yaml  # pyyaml is a required SDK dep
    metadata = {
        "name": "bfcl-noexec-500",
        "description": (
            "Berkeley Function-Call Leaderboard — sandbox-free subset "
            "(AST + abstention), stratified 500-task sample."
        ),
        "source": "gorilla BFCL_v4",
        "source_commit": BFCL_GITHUB_COMMIT,
        "verifier_fn": "bfcl_match",
        "seed": SEED,
        "n_tasks": len(sampled),
        "categories": CATEGORIES,
        "pool_counts": pool,
        "sampled_counts": draw,
    }
    (out_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))

    print(f"[build] wrote {len(sampled)} tasks → {out_dir}/tasks.jsonl")
    print(f"[build] pool total = {sum(pool.values())}; per-category draw:")
    for c in CATEGORIES:
        print(f"    {c:26s} pool={pool[c]:5d}  drew={draw[c]:4d}")
    return draw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir", type=Path, default=None,
        help="raw gorilla BFCL data dir (with BFCL_v4_*.json); sparse-clones the "
             "pinned commit when omitted.",
    )
    ap.add_argument(
        "--out-root", type=Path, default=Path("data/benchmark"),
        help="benchmark root; the suite is written to <out-root>/bfcl-noexec-500/.",
    )
    args = ap.parse_args()
    build(args.data_dir, args.out_root)


if __name__ == "__main__":
    sys.exit(main())

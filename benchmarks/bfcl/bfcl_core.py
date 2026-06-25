"""Shared BFCL core — the SINGLE source of truth both harnesses import.

The two harnesses (``evaluate.py`` = RAW baseline, no ``evsys_sdk``; and
``run_bfcl.py`` = SDK ``score_via_harbor`` path) must evaluate the SAME 500
tasks under IDENTICAL conditions. The only way to guarantee that is to make the
model input, the sampling params, the tool-call parse, and the scoring come from
ONE place. That place is this module.

What this module owns (and nothing else should re-implement):

  * :func:`build_messages` — the canonical chat ``messages`` for a BFCL task:
    a ``system`` message carrying the tool/function schemas in the Qwen/hermes
    tool format + the instruction to emit ``<tool_call>{...}</tool_call>`` blocks,
    and a ``user`` message with the BFCL query. This is the ONLY place a BFCL
    prompt is constructed. Both harnesses render THESE messages with the SAME
    ``tinker_cookbook`` renderer (``get_renderer("qwen3", tokenizer)`` →
    ``build_generation_prompt``), so the input token ids are identical by
    construction (proved by ``parity_check.py``).

  * :data:`ROLLOUT_PARAMS` — the sampling params both harnesses use. Chosen to
    match inspect_ai / PostTrainBench's AST-mode defaults: greedy decoding
    (``temperature=0.0``), ``max_tokens=2048``, ``num_samples=1``. ``top_p`` /
    ``top_k`` / ``seed`` are pinned to harbor's ``TinkerLLM`` defaults
    (``top_p=1.0``, ``top_k=-1``, ``seed=None``) so the RAW harness reproduces
    harbor's sampler exactly — the SDK path can't override those, so we don't.

  * the hermes :func:`parse_tool_calls` and the faithful :func:`score`
    (= ``bfcl_match``), re-exported from :mod:`verifier`, so both harnesses parse
    and score with byte-identical logic.

Why the system/user split matters for parity: the SDK harbor path builds its
messages as ``[{system: system_prompt} (if set), {user: instruction}]`` (see
``harbor_agents.BasicLoopAgent.run`` → ``Chat.chat`` → ``TinkerLLM.call``). So if
the SDK harness passes ``system_prompt = build_system_prompt(task)`` and the task
``instruction = <user query>``, harbor feeds the renderer the EXACT messages
:func:`build_messages` returns. The RAW harness renders :func:`build_messages`
directly. Same messages → same renderer → same token ids.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Re-export the shared parse + score from the faithful verifier. Both harnesses
# import these names from here so there is one parser and one scorer.
from verifier import bfcl_match as score  # noqa: E402
from verifier import parse_tool_calls  # noqa: E402

# ---------------------------------------------------------------------------
# Rollout params — identical for BOTH harnesses.
#
# These are the params that BOTH harnesses can set. The SDK path
# (``score_via_harbor``) accepts exactly ``num_samples``, ``max_tokens``,
# ``temperature`` for the tinker sampler; the rest (top_p/top_k/seed/stop) are
# fixed by harbor's ``TinkerLLM`` and the qwen3 renderer. The RAW harness mirrors
# those fixed values too (see ``FIXED_SAMPLING`` + STOP_FROM_RENDERER) so the two
# samplers are configured identically end-to-end.
# ---------------------------------------------------------------------------

ROLLOUT_PARAMS: dict[str, Any] = {
    "temperature": 0.0,   # greedy — matches inspect/PostTrainBench AST mode
    "max_tokens": 2048,   # generous single-turn FC budget
    "num_samples": 1,     # one rollout per task; AST match is deterministic
}

# Params harbor's TinkerLLM fixes (the SDK path does not expose them). The RAW
# harness sets these explicitly so its tinker.SamplingParams equals harbor's.
FIXED_SAMPLING: dict[str, Any] = {
    "top_p": 1.0,
    "top_k": -1,
    "seed": None,
}

# The renderer name both harnesses use. The stop sequence is NOT hardcoded here —
# both harnesses read it from ``renderer.get_stop_sequences()`` (see
# ``parity_check.py``), exactly as harbor's TinkerLLM does, so they always agree.
RENDERER_NAME = "qwen3"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"

BENCH_DIR = Path("data/benchmark/bfcl-noexec-500")


# ---------------------------------------------------------------------------
# Prompt construction — the ONE place a BFCL prompt is built.
#
# We do NOT hand-roll the tool-list text. To be BYTE-IDENTICAL to inspect_ai's
# native function-calling mode, we let Qwen3's OWN chat template render the tool
# block: ``apply_chat_template([{user}], tools=<oai schemas>, ...)`` emits Qwen3's
# canonical ``# Tools ... <tools>{...}</tools> ... <tool_call>{...}</tool_call>``
# system message. We extract that system block verbatim and use it as BOTH the
# RAW harness's system message AND the SDK harness's ``system_prompt``.
#
# Verified (parity_check.py): for the SAME [system(block), user(query)] messages,
#   tok.apply_chat_template([user], tools=oai, add_generation_prompt=True)   (inspect FC)
#   == get_renderer("qwen3", tok).build_generation_prompt([system, user])    (harbor)
# produce identical token ids. So matching inspect reduces to: build this system
# block from the tokenizer, then both harnesses render [system, user].
# ---------------------------------------------------------------------------

_SYS_OPEN = "<|im_start|>system\n"
_IM_END = "<|im_end|>"

# Module-level tokenizer cache (keyed by model name): apply_chat_template is the
# source of the canonical tool block, and load is the one slow step.
_TOKENIZER_CACHE: dict[str, Any] = {}


def _get_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    tok = _TOKENIZER_CACHE.get(model_name)
    if tok is None:
        from transformers import AutoTokenizer  # local: only needed to render

        tok = AutoTokenizer.from_pretrained(model_name)
        _TOKENIZER_CACHE[model_name] = tok
    return tok


def _to_oai_tools(tools: list[dict]) -> list[dict]:
    """BFCL function dicts (already language-preprocessed by build_dataset) →
    the OpenAI ``{"type":"function","function":{name,description,parameters}}``
    tool schema the chat template's ``tools=`` argument expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            },
        }
        for fn in tools
    ]


def build_system_prompt(tools: list[dict], model_name: str = DEFAULT_MODEL_NAME) -> str:
    """The canonical Qwen3 tool-calling **system** message for ``tools``.

    Rendered by Qwen3's own chat template (``tools=`` argument) and extracted
    verbatim, so it is byte-identical to inspect_ai's native FC system block.

    Returns ``""`` when ``tools`` is empty: Qwen3's template emits NO system block
    with no tools (matching inspect's native FC with no tools), so there is no
    system message — ``build_messages`` then omits the system turn entirely. A few
    live_irrelevance tasks legitimately have zero tools (the correct answer is to
    abstain), so this is expected, not an error.
    """
    if not tools:
        return ""
    tok = _get_tokenizer(model_name)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": ""}],
        tools=_to_oai_tools(tools),
        add_generation_prompt=True,
        tokenize=False,
    )
    if _SYS_OPEN not in rendered:
        raise RuntimeError(
            "Qwen3 chat template did not emit a system block for a non-empty tool "
            "list; cannot build the canonical FC system prompt."
        )
    return rendered.split(_SYS_OPEN, 1)[1].split(_IM_END, 1)[0]


def _task_tools(task: Any) -> list[dict]:
    """The (already language-preprocessed) tool schemas carried on a task.

    Accepts either a raw task dict (``{"verifier": {"expected": {...}}}``, as in
    ``tasks.jsonl``) or an SDK :class:`HarborTask` (``task.verifier.expected``).
    """
    expected = _task_expected(task)
    return list(expected.get("tools") or [])


def _task_expected(task: Any) -> dict:
    verifier = task["verifier"] if isinstance(task, dict) else task.verifier
    expected = verifier["expected"] if isinstance(verifier, dict) else verifier.expected
    return dict(expected or {})


def _task_query(task: Any) -> str:
    """The BFCL user query for a task (the task ``instruction``)."""
    return (task["instruction"] if isinstance(task, dict) else task.instruction) or ""


def _task_system_prompt(task: Any, model_name: str) -> str:
    """The task's system message: the stored ``system_prompt`` if the dataset
    baked one in (the canonical Qwen3 FC tool block), else regenerated from the
    task's tools. Prefer the stored value so the rendered messages are exactly
    what the SDK harness feeds harbor (its per-task ``system_prompt``)."""
    stored = (
        task.get("system_prompt") if isinstance(task, dict)
        else getattr(task, "system_prompt", None)
    )
    if stored:
        return stored
    return build_system_prompt(_task_tools(task), model_name)


def build_messages(task: Any, model_name: str = DEFAULT_MODEL_NAME) -> list[dict[str, str]]:
    """The canonical chat messages for a BFCL task: ``[{system}, {user}]``.

    ``system`` = Qwen3's native tool-calling block for the task's tools (so it is
    byte-identical to inspect_ai's FC system message); ``user`` = the BFCL query.
    This is the ONLY place a BFCL prompt is constructed. Both harnesses feed THESE
    messages to the same renderer, so input tokens match exactly.

    Accepts a raw task dict (from ``tasks.jsonl``) or an SDK ``HarborTask``.
    A task with no tools (a few live_irrelevance tasks) has no system message —
    the messages are just ``[{user}]`` (matching inspect's no-tools FC render).
    """
    system = _task_system_prompt(task, model_name)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": _task_query(task)})
    return messages


def load_tasks(bench_dir: Path | None = None) -> list[dict]:
    """Load the raw task dicts from ``<bench_dir>/tasks.jsonl``."""
    bench_dir = bench_dir or BENCH_DIR
    path = Path(bench_dir) / "tasks.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


__all__ = [
    "build_messages",
    "build_system_prompt",
    "ROLLOUT_PARAMS",
    "FIXED_SAMPLING",
    "RENDERER_NAME",
    "DEFAULT_MODEL_NAME",
    "BENCH_DIR",
    "parse_tool_calls",
    "score",
    "load_tasks",
    "_task_expected",
    "_task_query",
]

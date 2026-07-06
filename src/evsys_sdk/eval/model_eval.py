"""Model eval — generate completions over the eval set via an InferenceClient.

Each row in the eval set has 3 queries; the model predicts a tool slug for
each. Pass@1/pass@3/pass^3 are computed in :mod:`.report`.

Generation calls are wrapped in the retry helper too — Tinker / remote
inference can occasionally raise transient errors that shouldn't kill the
whole run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..protocols import InferenceClient
from .retry import RetryReport, call_with_retry

_ANSWER_RE = re.compile(r"<answer>\s*([\w]+)\s*</answer>")
_FALLBACK_SLUG_RE = re.compile(r"\b[A-Z][A-Z0-9_]{6,}\b")


def extract_predicted_slug(text: str) -> str:
    """Pull the predicted slug from ``<answer>...</answer>`` tags;
    fall back to the longest ALL_CAPS_TOKEN if tags are missing.
    """
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    candidates = _FALLBACK_SLUG_RE.findall(text)
    return max(candidates, key=len) if candidates else ""


class PromptBuilder(Protocol):
    """Builds the model's input from a (query, toolkit, expected_slug) tuple."""

    def __call__(self, *, query: str, toolkit: str, expected_slug: str) -> str: ...


DEFAULT_SYSTEM = (
    "You are a tool search engine. Match user queries to the correct API tool. "
    "Think step by step inside <think></think> tags, then give your answer "
    "inside <answer></answer> tags."
)

DEFAULT_SYSTEM_NO_THINK = (
    "You are a tool search engine. Match user queries to the correct API tool. "
    "Give your answer inside <answer></answer> tags."
)


def qwen_chat_prompt(*, query: str, toolkit: str = "", expected_slug: str = "") -> str:
    """Qwen2/Qwen3 chat-template wrapper.

    DEPRECATED hand-built form (missing the auto-injected `<think>` scaffold
    that Qwen3.5 adds via its chat template). Retained for backwards
    compatibility with older eval runs. New code should use
    :func:`qwen3_chat_template_prompt` which round-trips through
    ``apply_chat_template`` and supports ``enable_thinking``.
    """
    return (
        f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\nQuery: {query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


_QWEN_TOKENIZERS: dict[str, Any] = {}


def _get_qwen_tokenizer(model_name: str):
    """Cached lookup of a HF tokenizer for chat-template rendering."""
    if model_name not in _QWEN_TOKENIZERS:
        from transformers import AutoTokenizer

        _QWEN_TOKENIZERS[model_name] = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    return _QWEN_TOKENIZERS[model_name]


def qwen3_chat_template_prompt(
    *,
    query: str,
    toolkit: str = "",
    expected_slug: str = "",
    model_name: str = "Qwen/Qwen3.5-4B",
    enable_thinking: bool = True,
    system_prompt: str | None = None,
) -> str:
    """Build the inference prompt via the model's official chat template.

    For Qwen3-family models, this correctly emits the ``<think>``-scaffold
    suffix matching how the model was trained:

    * ``enable_thinking=True``  → prompt ends with ``<|im_start|>assistant\\n<think>\\n``
      (model continues from inside the think block).
    * ``enable_thinking=False`` → prompt ends with
      ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`` (model emits
      ``<answer>`` directly).
    """
    sys_msg = system_prompt
    if sys_msg is None:
        sys_msg = DEFAULT_SYSTEM if enable_thinking else DEFAULT_SYSTEM_NO_THINK
    tokenizer = _get_qwen_tokenizer(model_name)
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"Query: {query}"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


@dataclass
class ModelEvalConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    max_attempts: int = 5
    """Retries per generation call (transient inference errors)."""
    batch_size: int = 1
    """When >1 and the client exposes ``generate_batch``, submit prompts in
    chunks of this size and collect them concurrently. Falls back to
    sequential ``generate`` calls if the client doesn't support batching."""
    prompt_builder: PromptBuilder = qwen_chat_prompt


@dataclass
class ModelEvalResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    """Per-row results: {tool_slug, toolkit, queries: [{query, completion,
    predicted, error}]}."""
    retry_report: RetryReport = field(default_factory=RetryReport)


def _row_qkeys(eval_rows: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for ri, row in enumerate(eval_rows):
        for qi, q in enumerate(row.get("queries", [])):
            out.append((ri, qi, q))
    return out


def run_model_eval(
    eval_rows: list[dict[str, Any]],
    *,
    client: InferenceClient,
    config: ModelEvalConfig | None = None,
    progress: bool = True,
) -> ModelEvalResult:
    cfg = config or ModelEvalConfig()
    report = RetryReport()

    # Build all (row,query) prompts up-front so we can choose between
    # sequential or batched submission.
    jobs = _row_qkeys(eval_rows)
    prompts: list[str] = []
    for ri, qi, query in jobs:
        row = eval_rows[ri]
        prompts.append(
            cfg.prompt_builder(
                query=query,
                toolkit=row.get("toolkit", ""),
                expected_slug=row.get("tool_slug", ""),
            )
        )

    completions: list[str | None] = [None] * len(jobs)
    use_batch = cfg.batch_size > 1 and hasattr(client, "generate_batch")
    if use_batch:
        # Submit in chunks; retry the whole chunk on transient error.
        idx = 0
        while idx < len(prompts):
            chunk = prompts[idx : idx + cfg.batch_size]
            ctx = f"model_gen_batch:{idx}-{idx + len(chunk) - 1}"
            result = call_with_retry(
                client.generate_batch,  # type: ignore[attr-defined]
                prompts=chunk,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                max_attempts=cfg.max_attempts,
                report=report,
                context=ctx,
            )
            if result is None:
                # Whole batch failed — mark each prompt as retry-exhausted.
                for j in range(len(chunk)):
                    completions[idx + j] = None
            else:
                for j, c in enumerate(result):
                    completions[idx + j] = c
            idx += cfg.batch_size
            if progress and idx % (cfg.batch_size * 4) == 0:
                done = min(idx, len(prompts))
                print(f"  model eval: {done}/{len(prompts)} prompts done")
    else:
        for i, p in enumerate(prompts):
            ri, qi, _ = jobs[i]
            ctx = f"model_gen:row{ri}:q{qi}"
            completions[i] = call_with_retry(
                client.generate,
                prompt=p,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                max_attempts=cfg.max_attempts,
                report=report,
                context=ctx,
            )
            if progress and (i + 1) % 60 == 0:
                print(f"  model eval: {i + 1}/{len(prompts)} prompts done")

    # Reassemble per-row results.
    out_rows: list[dict[str, Any]] = []
    cursor = 0
    for ri, row in enumerate(eval_rows):
        expected = row.get("tool_slug", "")
        toolkit = row.get("toolkit", "")
        q_outs: list[dict[str, Any]] = []
        for query in row.get("queries", []):
            comp = completions[cursor]
            cursor += 1
            if comp is None:
                q_outs.append({"query": query, "completion": "", "predicted": "", "error": "retry_exhausted"})
            else:
                q_outs.append(
                    {
                        "query": query,
                        "completion": comp,
                        "predicted": extract_predicted_slug(comp),
                        "error": None,
                    }
                )
        out_rows.append({"tool_slug": expected, "toolkit": toolkit, "queries": q_outs})

    return ModelEvalResult(rows=out_rows, retry_report=report)

"""SFT data shaping — turn ``{messages: [...]}`` rows into ``tinker.Datum``s.

Extracted from ``evsys_sdk/algorithms/tinker_sft.py:_row_to_datum``. Pure
function over the tokenizer + max_seq_len + enable_thinking; no tinker_cookbook
imports, no global state. The new ``SFTStepBuilder`` (commit #61) calls
``sft_tokenize`` once at construction time and reads tokenized Datums out of
the result in each ``build_batch`` call.

Output shape: ``list[tinker.Datum]`` where each ``Datum`` has

  * ``model_input``       — token ids of the *entire* chat-templated sequence
  * ``loss_fn_inputs["weights"]`` — per-position float mask, ``1.0`` on
    assistant tokens, ``0.0`` elsewhere.

This is the shape tinker's server-side ``cross_entropy`` loss consumes;
see ``training_client.forward_backward_async(loss_fn="cross_entropy")``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import tinker
import torch

from .templates import Message, apply_template

logger = logging.getLogger(__name__)


def sft_tokenize(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_len: int,
    enable_thinking: bool | None = None,
) -> list[tinker.Datum]:
    """Tokenize every row → list of ``tinker.Datum``.

    Rows must have ``messages: list[{"role", "content"}]``. Rows whose
    rendering produces zero assistant tokens (e.g. system+user only — no
    label) are silently dropped; a warning is logged with the count.
    """
    data: list[tinker.Datum] = []
    skipped = 0
    for r in rows:
        d = row_to_datum(r, tokenizer, max_seq_len=max_seq_len,
                         enable_thinking=enable_thinking)
        if d is None:
            skipped += 1
            continue
        data.append(d)
    if skipped:
        logger.warning("sft_tokenize: skipped %d/%d rows with no assistant span",
                       skipped, len(rows))
    if not data:
        raise ValueError(
            "sft_tokenize: all rows produced empty (no assistant tokens). "
            "Check your data — at least one row needs an assistant turn."
        )
    return data


def row_to_datum(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_seq_len: int,
    enable_thinking: bool | None = None,
) -> tinker.Datum | None:
    """Single-row tokenize + assistant-span loss mask → ``tinker.Datum``.

    Strategy: render the full sequence once for the input ids; then walk
    the messages and for each assistant turn, render up to (and not
    including) it with ``add_generation_prompt=True`` to find the prefix
    span, render through it without to find the end. The diff is the
    assistant token span — mark it 1.0 in the weight vector.

    Returns ``None`` when the row has no assistant turn or the assistant
    span ends up empty after truncation to ``max_seq_len``.
    """
    messages: list[Message] = list(row.get("messages") or [])
    if not messages:
        raise ValueError("row has empty messages")
    if not any(m.get("role") == "assistant" for m in messages):
        return None

    full_text = apply_template(
        tokenizer, messages, add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    weights = [0.0] * len(full_ids)
    cursor = 0
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        prefix_text = apply_template(
            tokenizer, messages[:i], add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        through_text = apply_template(
            tokenizer, messages[: i + 1], add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        through_ids = tokenizer.encode(through_text, add_special_tokens=False)
        start = max(cursor, len(prefix_ids))
        end = min(len(weights), len(through_ids))
        for j in range(start, end):
            weights[j] = 1.0
        cursor = end

    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        weights = weights[:max_seq_len]
    if sum(weights) == 0:
        return None

    return _build_datum(full_ids, weights)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_datum(ids: list[int], weights: list[float]) -> tinker.Datum:
    """Construct the ``Datum`` the cross_entropy loss expects.

    Tinker's CE loss is left-shifted (predicts position ``t`` from position
    ``t-1``), so ``model_input`` covers positions ``[0, N-1]``, ``target_tokens``
    covers ``[1, N]``, and ``weights`` aligns with ``target_tokens``.
    """
    model_input = tinker.ModelInput.from_ints(ids[:-1])
    targets = ids[1:]
    target_weights = weights[1:]
    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(
                torch.tensor(targets, dtype=torch.long)
            ),
            "weights": tinker.TensorData.from_torch(
                torch.tensor(target_weights, dtype=torch.float32)
            ),
        },
    )


__all__ = ["row_to_datum", "sft_tokenize"]

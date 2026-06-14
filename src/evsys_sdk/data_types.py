"""Harbor-compatible data shapes.

The schemas runners consume and dashboards render. Mirrors the production
internal types so JSONL files round-trip between this SDK and the internal
serving / dashboard stack without conversion.

Two row formats:
  * ``ChatMessagesRow``  — SFT example  (messages prefix + supervised target)
  * ``HarborTask``       — RL task      (instruction + verifier spec)

Verifier specs (the data the runner serializes for HarborTask):
  * ``InProcessVerifier`` — cheap Python fn lookup
  * ``E2BVerifier``       — sandboxed code execution
  * ``LLMJudgeVerifier``  — judge model + rubric
  * ``VerifierPayload``   — discriminated union of the three (use this in type hints).

These are *data shapes* (Pydantic / dataclasses) describing the verification
plan; the runtime ``Verifier`` Protocol in ``protocols.py`` is what actually
EXECUTES verification. They are deliberately separate concepts.

Multimodal content: messages can carry text + image blocks in OpenAI or
Anthropic style. Use ``image_url_block(url)`` or
``image_base64_block(media_type, b64)`` for convenience.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Union

# ---------------------------------------------------------------------------
# Target formats — what a runner consumes
# ---------------------------------------------------------------------------


class TargetFormat(str, enum.Enum):
    CHAT_MESSAGES   = "chat_messages"      # SFT
    HARBOR_TASK     = "harbor_task"        # RL via verifier rollouts
    PROMPT_DATASET  = "prompt_dataset"     # GEPA prompt tuning (no weight updates)


# ---------------------------------------------------------------------------
# ChatMessagesRow — SFT
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatMessagesRow:
    """One SFT conversation — pure data, no supervision encoded.

    ``messages`` is the full multi-turn conversation (roles: system / user /
    assistant / tool). The row deliberately carries **no** notion of which
    tokens are trained: choosing the supervised span is the *algorithm's* job
    (e.g. ``NativeSFT`` masks assistant turns according to its ``supervise``
    config). Keeping the dataset format free of target/loss metadata lets the
    same conversation feed any SFT variant.

    For multimodal SFT, a message's ``content`` may be either a string OR a
    list of content blocks (mix of text + image blocks). See
    ``image_url_block`` / ``image_base64_block`` helpers below.
    """

    messages: list[dict]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Verifier specs — discriminated by `kind`
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InProcessVerifier:
    """Cheap Python-function verifier.

    The runner looks up ``fn_name`` in a registered verifier-fn map and calls
    it with ``(completion, expected, **params)``. Sub-millisecond — use this
    for tool-call matching, exact-match, boxed-answer parsing.
    """

    fn_name: str
    expected: Any = None
    params: dict = field(default_factory=dict)
    kind: Literal["in_process"] = "in_process"


@dataclass(frozen=True)
class E2BVerifier:
    """Sandboxed code verifier — runs ``test_state.py`` inside an E2B container.

    The runner builds the container per the ``dockerfile``, drops in the
    model's completion + ``test_state.py``, executes ``test_sh`` (default
    ``pytest -q test_state.py``), and reads pass/fail from the exit code.
    """

    dockerfile: str = ""
    test_sh: str = ""
    test_state_py: str = ""
    kind: Literal["e2b"] = "e2b"


@dataclass(frozen=True)
class LLMJudgeVerifier:
    """LLM judge — ``judge_model`` scores the completion against ``rubric``.

    The runner calls the judge model with the rubric + completion and parses a
    numeric score from the response (usually in a ``\\boxed{}`` or
    ``<score>...</score>`` tag).
    """

    judge_model: str = ""
    rubric: str = ""
    kind: Literal["llm_judge"] = "llm_judge"


VerifierPayload = Union[InProcessVerifier, E2BVerifier, LLMJudgeVerifier]
"""Discriminated by ``.kind`` ∈ {'in_process', 'e2b', 'llm_judge'}."""


# ---------------------------------------------------------------------------
# HarborTask — RL
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarborTask:
    """One RL task — instruction + verifier spec.

    Same shape whether the task came from a hand-written eval set, an HF
    dataset, or a generated rollout corpus. Runners materialize the prompt by
    feeding ``instruction`` to the policy, generate a rollout, and score it
    using ``verifier`` (whichever variant — see ``VerifierPayload``).
    """

    task_id: str
    instruction: str
    verifier: VerifierPayload
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PromptExample — GEPA prompt tuning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptExample:
    """One example for prompt-search / GEPA-style optimization.

    ``inputs`` is a dict of named task inputs; ``expected`` is the gold output
    the score function compares the model's completion against.
    """

    inputs: dict
    expected: Any
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Multimodal content block helpers
# ---------------------------------------------------------------------------


def text_block(text: str) -> dict:
    """Standard text block. Works in either OpenAI or Anthropic shape."""
    return {"type": "text", "text": text}


def image_url_block(url: str, *, detail: str | None = None) -> dict:
    """OpenAI-style image block. Works for any vision model that follows the
    OpenAI multimodal content schema (most do)."""
    iu: dict = {"url": url}
    if detail:
        iu["detail"] = detail
    return {"type": "image_url", "image_url": iu}


def image_base64_block(media_type: str, b64_data: str) -> dict:
    """Anthropic-style base64 image block.

    ``media_type`` is the MIME type, e.g. ``"image/png"``.
    """
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": b64_data},
    }


def block_to_image_src(block: Any) -> str | None:
    """Return a renderable image src URL/data-URL for a content block, or None.

    Handles OpenAI ``image_url`` blocks AND Anthropic ``image`` blocks (both
    URL and base64 variants). Used by dashboard renderers.
    """
    if not isinstance(block, dict):
        return None
    if block.get("type") == "image_url":
        iu = block.get("image_url") or {}
        url = iu.get("url")
        if isinstance(url, str):
            return url
    if block.get("type") == "image" and isinstance(block.get("source"), dict):
        src = block["source"]
        if src.get("type") == "url" and isinstance(src.get("url"), str):
            return src["url"]
        if (
            src.get("type") == "base64"
            and isinstance(src.get("data"), str)
            and isinstance(src.get("media_type"), str)
        ):
            return f"data:{src['media_type']};base64,{src['data']}"
    return None


def has_images(row: ChatMessagesRow) -> bool:
    """True iff any message in the row carries at least one image block."""
    for m in row.messages:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            for b in c:
                if block_to_image_src(b) is not None:
                    return True
    return False


# ---------------------------------------------------------------------------
# Row-format discrimination (for dashboards / mixed-format files)
# ---------------------------------------------------------------------------


def detect_format(row: Any) -> str:
    """Returns 'chat_messages' | 'harbor_task' | 'prompt_dataset' | 'unknown'.

    Checked most-specific first: ``harbor_task`` and ``prompt_dataset`` have
    distinctive key pairs; ``chat_messages`` is any row carrying ``messages``
    (the conversation — supervision is decided by the algorithm, not the row).
    """
    if not isinstance(row, dict):
        return "unknown"
    if "task_id" in row and "verifier" in row:
        return "harbor_task"
    if "inputs" in row and "expected" in row:
        return "prompt_dataset"
    if "messages" in row:
        return "chat_messages"
    return "unknown"


# ---------------------------------------------------------------------------
# JSON conversion helpers — round-trip dicts <-> dataclasses
# ---------------------------------------------------------------------------


def _verifier_from_dict(d: dict) -> VerifierPayload:
    kind = d.get("kind")
    if kind == "in_process":
        return InProcessVerifier(
            fn_name=d["fn_name"],
            expected=d.get("expected"),
            params=dict(d.get("params") or {}),
        )
    if kind == "e2b":
        return E2BVerifier(
            dockerfile=d.get("dockerfile", ""),
            test_sh=d.get("test_sh", ""),
            test_state_py=d.get("test_state_py", ""),
        )
    if kind == "llm_judge":
        return LLMJudgeVerifier(
            judge_model=d.get("judge_model", ""),
            rubric=d.get("rubric", ""),
        )
    raise ValueError(f"unknown verifier kind: {kind!r}")


def harbor_task_from_dict(d: dict) -> HarborTask:
    return HarborTask(
        task_id=d["task_id"],
        instruction=d["instruction"],
        verifier=_verifier_from_dict(d["verifier"]),
        metadata=dict(d.get("metadata") or {}),
    )


def chat_messages_row_from_dict(d: dict) -> ChatMessagesRow:
    msgs = d.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("chat_messages row requires a non-empty `messages` list")
    return ChatMessagesRow(
        messages=list(msgs),
        metadata=dict(d.get("metadata") or {}),
    )


def prompt_example_from_dict(d: dict) -> PromptExample:
    return PromptExample(
        inputs=dict(d.get("inputs") or {}),
        expected=d.get("expected"),
        metadata=dict(d.get("metadata") or {}),
    )


def from_dict(row: dict) -> Union[ChatMessagesRow, HarborTask, PromptExample]:
    """Dispatch on row shape — round-trips the JSONL coming off a runner."""
    fmt = detect_format(row)
    if fmt == "chat_messages":  return chat_messages_row_from_dict(row)
    if fmt == "harbor_task":    return harbor_task_from_dict(row)
    if fmt == "prompt_dataset": return prompt_example_from_dict(row)
    raise ValueError(f"can't dispatch row — unknown format: keys={sorted(row.keys())[:6]}")


_ROW_PARSERS = {
    "chat_messages": chat_messages_row_from_dict,
    "harbor_task": harbor_task_from_dict,
    "prompt_dataset": prompt_example_from_dict,
}


def parse_rows(
    rows: Iterable[dict],
    fmt: Union["TargetFormat", str],
) -> list[Union[ChatMessagesRow, HarborTask, PromptExample]]:
    """Strictly parse raw dicts into typed rows for the given ``fmt``.

    This is the standardized boundary between the transform stage and a
    StepBuilder/algorithm: ``raw rows -> transforms -> parse_rows(fmt) -> typed
    rows``. ``fmt`` is a :class:`TargetFormat` (or its string value). Every row
    must match ``fmt`` per :func:`detect_format`; a mismatch raises
    ``ValueError`` naming the offending row — no silent coercion, mirroring the
    SDK's ``extra='forbid'`` philosophy. Tokenization/rollout stays downstream,
    owned by the algorithm.
    """
    want = fmt.value if isinstance(fmt, TargetFormat) else str(fmt)
    parser = _ROW_PARSERS.get(want)
    if parser is None:
        raise ValueError(
            f"parse_rows: unsupported target format {want!r} "
            f"(expected one of {sorted(_ROW_PARSERS)})"
        )
    out: list[Union[ChatMessagesRow, HarborTask, PromptExample]] = []
    for i, r in enumerate(rows):
        got = detect_format(r)
        if got != want:
            keys = sorted(r)[:6] if isinstance(r, dict) else type(r).__name__
            raise ValueError(
                f"parse_rows: row {i} has format {got!r}, expected {want!r} "
                f"(keys={keys})"
            )
        out.append(parser(r))
    return out


def to_dict(
    obj: Union[ChatMessagesRow, HarborTask, PromptExample, VerifierPayload],
) -> dict:
    """Dataclass → plain dict (JSON-serializable)."""
    import dataclasses as _dc
    return _dc.asdict(obj)


def iter_jsonl(path: str) -> Iterable[Union[ChatMessagesRow, HarborTask, PromptExample]]:
    """Iterate a mixed-format JSONL and yield typed rows."""
    import json
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield from_dict(json.loads(line))


__all__ = [
    "TargetFormat",
    "ChatMessagesRow", "HarborTask", "PromptExample",
    "InProcessVerifier", "E2BVerifier", "LLMJudgeVerifier", "VerifierPayload",
    "text_block", "image_url_block", "image_base64_block",
    "block_to_image_src", "has_images",
    "detect_format",
    "harbor_task_from_dict", "chat_messages_row_from_dict", "prompt_example_from_dict",
    "from_dict", "parse_rows", "to_dict", "iter_jsonl",
]

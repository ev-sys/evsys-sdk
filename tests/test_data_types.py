"""Regression tests for the harbor-compatible data shapes.

Cover:
  * Shape construction (every dataclass instantiates with required fields)
  * detect_format() correctly disambiguates by key signature
  * JSONL round-trip: dataclass → dict → JSON → dict → dataclass is lossless
  * Verifier discriminated union: each kind round-trips with the right type
  * Multimodal helpers: text/image_url/image_base64 produce valid blocks +
    block_to_image_src parses them back
  * has_images() correctly flags multimodal chat rows
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from evsys_sdk.data_types import (
    ChatMessagesRow,
    E2BVerifier,
    HarborTask,
    InProcessVerifier,
    LLMJudgeVerifier,
    PromptExample,
    TargetFormat,
    block_to_image_src,
    chat_messages_row_from_dict,
    detect_format,
    from_dict,
    harbor_task_from_dict,
    has_images,
    image_base64_block,
    image_url_block,
    iter_jsonl,
    prompt_example_from_dict,
    text_block,
    to_dict,
)


class TestShapes:
    def test_target_format_values(self):
        assert TargetFormat.CHAT_MESSAGES.value == "chat_messages"
        assert TargetFormat.HARBOR_TASK.value == "harbor_task"
        assert TargetFormat.PROMPT_DATASET.value == "prompt_dataset"

    def test_chat_messages_row_construct(self):
        row = ChatMessagesRow(
            messages=[{"role": "user", "content": "hi"}],
            target_assistant="hello",
        )
        assert row.target_assistant == "hello"
        assert row.metadata == {}

    def test_harbor_task_construct_in_process(self):
        v = InProcessVerifier(fn_name="exact_match", expected="42", params={"strip": True})
        t = HarborTask(task_id="t1", instruction="what is 6*7?", verifier=v)
        assert t.verifier.kind == "in_process"
        assert t.verifier.fn_name == "exact_match"

    def test_harbor_task_construct_e2b(self):
        v = E2BVerifier(dockerfile="FROM python:3.11", test_sh="pytest", test_state_py="def test(): pass")
        t = HarborTask(task_id="t2", instruction="solve it", verifier=v)
        assert t.verifier.kind == "e2b"

    def test_harbor_task_construct_llm_judge(self):
        v = LLMJudgeVerifier(judge_model="claude-sonnet-4-6", rubric="Score 1-5 on clarity.")
        t = HarborTask(task_id="t3", instruction="reply nicely", verifier=v)
        assert t.verifier.kind == "llm_judge"

    def test_prompt_example_construct(self):
        e = PromptExample(inputs={"q": "what is 1+1?"}, expected="2")
        assert e.inputs["q"] == "what is 1+1?"
        assert e.expected == "2"


class TestDetectFormat:
    def test_chat_messages(self):
        assert detect_format({"messages": [], "target_assistant": "x"}) == "chat_messages"

    def test_harbor_task(self):
        assert detect_format({"task_id": "t", "instruction": "i", "verifier": {"kind": "in_process"}}) == "harbor_task"

    def test_prompt_dataset(self):
        assert detect_format({"inputs": {}, "expected": "x"}) == "prompt_dataset"

    def test_unknown(self):
        assert detect_format({"foo": "bar"}) == "unknown"
        assert detect_format("not a dict") == "unknown"
        assert detect_format(None) == "unknown"


class TestRoundTrip:
    def test_chat_messages_round_trip(self):
        row = ChatMessagesRow(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
            target_assistant="hello!",
            metadata={"split": "train", "row_id": "ex_0001"},
        )
        d = to_dict(row)
        j = json.loads(json.dumps(d))
        round_tripped = chat_messages_row_from_dict(j)
        assert round_tripped == row

    @pytest.mark.parametrize("verifier", [
        InProcessVerifier(fn_name="aime_boxed_match", expected="\\frac{1}{4}", params={"normalize_latex": True}),
        E2BVerifier(dockerfile="FROM python:3.11", test_sh="pytest", test_state_py="def t(): pass"),
        LLMJudgeVerifier(judge_model="gpt-4o-2024-08-06", rubric="Score 1-5."),
    ])
    def test_harbor_task_round_trip(self, verifier):
        t = HarborTask(task_id="ex_0042", instruction="do it", verifier=verifier, metadata={"diff": "hard"})
        d = to_dict(t)
        j = json.loads(json.dumps(d))
        round_tripped = harbor_task_from_dict(j)
        assert round_tripped == t
        assert round_tripped.verifier.kind == verifier.kind

    def test_prompt_example_round_trip(self):
        e = PromptExample(inputs={"q": "1+1"}, expected="2", metadata={"src": "tiny"})
        d = to_dict(e)
        j = json.loads(json.dumps(d))
        assert prompt_example_from_dict(j) == e

    def test_from_dict_dispatch(self):
        rows = [
            {"messages": [], "target_assistant": "x"},
            {"task_id": "t", "instruction": "i", "verifier": {"kind": "in_process", "fn_name": "f"}},
            {"inputs": {"q": "a"}, "expected": "b"},
        ]
        out = [from_dict(r) for r in rows]
        assert isinstance(out[0], ChatMessagesRow)
        assert isinstance(out[1], HarborTask)
        assert isinstance(out[2], PromptExample)

    def test_iter_jsonl_mixed(self, tmp_path: Path):
        p = tmp_path / "mixed.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps({"messages": [], "target_assistant": "ok"}) + "\n")
            f.write("\n")  # blank line should be skipped
            f.write(json.dumps({"task_id": "t1", "instruction": "i",
                                "verifier": {"kind": "in_process", "fn_name": "exact"}}) + "\n")
        rows = list(iter_jsonl(str(p)))
        assert len(rows) == 2
        assert isinstance(rows[0], ChatMessagesRow)
        assert isinstance(rows[1], HarborTask)

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="unknown format"):
            from_dict({"foo": 1})


class TestMultimodal:
    def test_text_block(self):
        b = text_block("hello")
        assert b == {"type": "text", "text": "hello"}

    def test_image_url_block(self):
        b = image_url_block("https://x.com/img.png")
        assert b["type"] == "image_url"
        assert b["image_url"]["url"] == "https://x.com/img.png"

    def test_image_url_block_with_detail(self):
        b = image_url_block("https://x.com/img.png", detail="high")
        assert b["image_url"]["detail"] == "high"

    def test_image_base64_block(self):
        b = image_base64_block("image/png", "iVBORw0KGgo")
        assert b["type"] == "image"
        assert b["source"]["type"] == "base64"
        assert b["source"]["media_type"] == "image/png"
        assert b["source"]["data"] == "iVBORw0KGgo"

    @pytest.mark.parametrize("block,expected_prefix", [
        (image_url_block("https://x.com/a.png"), "https://"),
        (image_base64_block("image/png", "AAA"), "data:image/png;base64,"),
        ({"type": "image", "source": {"type": "url", "url": "https://y.com/b.jpg"}}, "https://"),
    ])
    def test_block_to_image_src_known_shapes(self, block, expected_prefix):
        src = block_to_image_src(block)
        assert src is not None
        assert src.startswith(expected_prefix)

    @pytest.mark.parametrize("block", [
        None, "not a dict", {"type": "text", "text": "hi"}, {"type": "image"},
        {"type": "image", "source": {}}, 42, [],
    ])
    def test_block_to_image_src_returns_none(self, block):
        assert block_to_image_src(block) is None

    def test_has_images_text_only(self):
        row = ChatMessagesRow(messages=[{"role": "user", "content": "hi"}], target_assistant="hello")
        assert has_images(row) is False

    def test_has_images_with_image(self):
        row = ChatMessagesRow(
            messages=[{"role": "user", "content": [text_block("look:"), image_url_block("https://x.com/a.png")]}],
            target_assistant="I see it.",
        )
        assert has_images(row) is True


class TestVerifierFromDict:
    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown verifier kind"):
            harbor_task_from_dict({
                "task_id": "t", "instruction": "i",
                "verifier": {"kind": "nope"},
            })

    def test_in_process_minimal(self):
        t = harbor_task_from_dict({
            "task_id": "t", "instruction": "i",
            "verifier": {"kind": "in_process", "fn_name": "f"},
        })
        assert isinstance(t.verifier, InProcessVerifier)
        assert t.verifier.expected is None
        assert t.verifier.params == {}


class TestImmutability:
    def test_dataclasses_are_frozen(self):
        row = ChatMessagesRow(messages=[], target_assistant="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            row.target_assistant = "y"  # type: ignore[misc]

        v = InProcessVerifier(fn_name="f")
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.fn_name = "g"  # type: ignore[misc]

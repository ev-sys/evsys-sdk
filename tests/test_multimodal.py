"""Multimodal (image) message helpers used by the tinker SFT renderer path.

These are the tinker-free pieces of image support: detecting image-bearing rows
and flattening OpenAI/Anthropic image blocks into the ``{type: text|image}``
part shape a chat renderer consumes. The renderer wiring itself lives in
``tinker_sft`` and needs tinker_cookbook installed, so it's exercised
end-to-end against a real backend rather than here.
"""

from __future__ import annotations

from trajectory_labs.data_types import (
    ChatMessagesRow,
    has_images,
    image_base64_block,
    image_url_block,
    messages_have_images,
    normalize_message_images,
    text_block,
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_text_only_messages_have_no_images():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert messages_have_images(msgs) is False


def test_openai_image_block_detected():
    msgs = [{"role": "user", "content": [text_block("look"), image_url_block("http://x/y.png")]}]
    assert messages_have_images(msgs) is True


def test_anthropic_image_block_detected():
    msgs = [{"role": "user", "content": [image_base64_block("image/png", "aGk=")]}]
    assert messages_have_images(msgs) is True


def test_has_images_on_chatmessagesrow_delegates():
    row = ChatMessagesRow(
        messages=[{"role": "user", "content": [image_url_block("http://x/y.png")]}],
        target_assistant="ok",
    )
    assert has_images(row) is True


# ---------------------------------------------------------------------------
# Normalization → renderer part shape
# ---------------------------------------------------------------------------


def test_string_content_passed_through():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert normalize_message_images(msgs) == msgs


def test_openai_blocks_flattened():
    msgs = [{"role": "user", "content": [text_block("look"), image_url_block("http://x/y.png")]}]
    out = normalize_message_images(msgs)
    assert out == [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image", "image": "http://x/y.png"},
    ]}]


def test_anthropic_image_becomes_data_uri():
    msgs = [{"role": "user", "content": [image_base64_block("image/png", "aGk=")]}]
    part = normalize_message_images(msgs)[0]["content"][0]
    assert part["type"] == "image"
    assert part["image"] == "data:image/png;base64,aGk="


def test_role_and_extra_keys_preserved():
    msgs = [{"role": "assistant", "content": "done", "name": "tool_x"}]
    assert normalize_message_images(msgs)[0]["name"] == "tool_x"


def test_unknown_blocks_dropped_text_kept():
    msgs = [{"role": "user", "content": [
        {"type": "weird"},                 # unknown → dropped
        "bare string",                     # bare str → text part
        text_block("kept"),
    ]}]
    out = normalize_message_images(msgs)[0]["content"]
    assert out == [{"type": "text", "text": "bare string"}, {"type": "text", "text": "kept"}]

"""SDPO testable cores: chat slicing + the selected-token surrogate loss.

The Tinker training loop needs a live API key and is not tested here.
"""

from __future__ import annotations

import pytest

from trajectory_labs.algorithms.sdpo import (
    sdpo_surrogate_per_token_loss,
    slice_chat_for_sdpo,
)

torch = pytest.importorskip("torch")


def test_slice_pairs_assistant_with_next_user_feedback():
    messages = [
        {"role": "user", "content": "do X"},
        {"role": "assistant", "content": "attempt 1"},
        {"role": "user", "content": "no, X means Y"},      # feedback for attempt 1
        {"role": "assistant", "content": "attempt 2"},
        {"role": "user", "content": "thanks"},              # feedback for attempt 2
        {"role": "assistant", "content": "final"},          # no following user -> skipped
    ]
    recs = slice_chat_for_sdpo(messages, feedback_prefix="FB: ")
    assert len(recs) == 2
    r0 = recs[0]
    assert r0["response"] == "attempt 1"
    # Student sees only history before the assistant turn (no feedback).
    assert {m["content"] for m in r0["student_messages"]} == {"do X"}
    # Teacher sees the next user turn as feedback.
    assert r0["teacher_messages"][-1] == {"role": "user", "content": "FB: no, X means Y"}
    assert "no, X means Y" not in [m["content"] for m in r0["student_messages"]]


def test_slice_skips_empty_feedback_and_non_followed_turns():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "   "},   # whitespace feedback -> skipped
    ]
    assert slice_chat_for_sdpo(messages, min_feedback_chars=1) == []


def test_surrogate_loss_zero_when_student_equals_teacher():
    lp = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
    loss, metrics = sdpo_surrogate_per_token_loss(lp, lp.detach())
    assert abs(loss.item()) < 1e-6          # log_ratio == 0 -> zero surrogate
    assert abs(metrics["sdpo/mean_log_ratio"]) < 1e-6


def test_surrogate_loss_has_gradient_to_student_only():
    student = torch.tensor([-0.5, -1.5], requires_grad=True)
    teacher = torch.tensor([-1.0, -1.0])
    loss, _ = sdpo_surrogate_per_token_loss(student, teacher)
    loss.backward()
    assert student.grad is not None
    # grad of (s-t).detach()*s wrt s is (s-t).detach()
    expected = (torch.tensor([-0.5, -1.5]) - teacher) / 2.0
    assert torch.allclose(student.grad, expected, atol=1e-6)


def test_surrogate_loss_respects_mask():
    student = torch.tensor([-0.5, -1.5, 99.0], requires_grad=True)
    teacher = torch.tensor([-1.0, -1.0, -1.0])
    mask = torch.tensor([1.0, 1.0, 0.0])  # ignore the 3rd token
    loss, _ = sdpo_surrogate_per_token_loss(student, teacher, mask)
    # only first two tokens contribute
    per = (torch.tensor([-0.5, -1.5]) - torch.tensor([-1.0, -1.0])) * torch.tensor([-0.5, -1.5])
    assert abs(loss.item() - (per.sum() / 2).item()) < 1e-6

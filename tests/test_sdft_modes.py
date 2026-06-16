"""SDFT distillation modes: forward-KL (default), reverse-KL, importance-sampling.

Unit-tests the two new objectives' data shaping + loss math directly (no tinker
server needed). Forward-KL is covered by test_algorithms_sdft.py.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("tinker")
pytest.importorskip("torch")

import tinker  # noqa: E402
import torch  # noqa: E402

from evsys_sdk.algorithms.sdft import SDFTConfig  # noqa: E402
from evsys_sdk.training.sdft_data import (  # noqa: E402
    build_importance_sampling_targets,
    extract_completion_tokens,
    reverse_kl_custom_loss,
    student_datum_from_rollout,
)


# --- config validation -----------------------------------------------------


def test_loss_mode_validation():
    assert SDFTConfig().loss_mode == "forward_kl"
    SDFTConfig(loss_mode="reverse_kl", topk=20)
    SDFTConfig(loss_mode="importance_sampling", topk=0)
    with pytest.raises(Exception):
        SDFTConfig(loss_mode="reverse_kl", topk=0)  # needs top-K
    with pytest.raises(Exception):
        SDFTConfig(loss_mode="bogus")


# --- reverse KL custom loss ------------------------------------------------


def _weights_datum(weights_NK: torch.Tensor) -> tinker.Datum:
    N = weights_NK.shape[0]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(N))),
        loss_fn_inputs={
            "weights": tinker.TensorData.from_torch(weights_NK),
            "target_tokens": tinker.TensorData.from_torch(
                torch.zeros_like(weights_NK, dtype=torch.long)
            ),
        },
    )


def test_reverse_kl_zero_when_student_matches_teacher():
    # Position 0 masked (weights 0); position 1 valid teacher q = [0.7, 0.3].
    weights = torch.tensor([[0.0, 0.0], [0.7, 0.3]])
    datum = _weights_datum(weights)
    # Student logprobs at the 2 top-K slots; row 1 already == log(teacher).
    student_logp = torch.tensor([[0.0, 0.0], [math.log(0.7), math.log(0.3)]])

    loss, metrics = reverse_kl_custom_loss([datum], [student_logp])
    assert metrics["sdft/reverse_kl_positions"] == 1.0
    assert abs(metrics["sdft/reverse_kl_mean"]) < 1e-5   # matched → ~0 KL
    assert abs(loss.item()) < 1e-5


def test_reverse_kl_positive_when_mismatched():
    weights = torch.tensor([[0.0, 0.0], [0.7, 0.3]])
    datum = _weights_datum(weights)
    student_logp = torch.tensor([[0.0, 0.0], [math.log(0.3), math.log(0.7)]])  # swapped
    _loss, metrics = reverse_kl_custom_loss([datum], [student_logp])
    assert metrics["sdft/reverse_kl_mean"] > 0.0


# --- importance-sampling advantages ----------------------------------------


def test_importance_sampling_advantage_is_teacher_minus_student():
    # prompt = 3 tokens, completion = 2 tokens.
    prompt = tinker.ModelInput.from_ints([10, 11, 12])
    datum = student_datum_from_rollout(prompt=prompt, completion_tokens=[20, 21])
    slice_ = extract_completion_tokens(datum, teacher_prompt_len=3, max_context_length=2048)

    student_lps = [-1.0, -2.0]
    # teacher per-position logprobs over the teacher-forced seq (len 5);
    # positions 3,4 are the completion → teacher comp lps = [-0.5, -0.5].
    teacher_lps = [0.0, 0.0, 0.0, -0.5, -0.5]

    datums, metrics = build_importance_sampling_targets(
        student_data=[datum],
        student_logprobs=[student_lps],
        completion_slices=[slice_],
        teacher_logprobs=[teacher_lps],
    )
    adv = datums[0].loss_fn_inputs["advantages"].to_torch()
    lp = datums[0].loss_fn_inputs["logprobs"].to_torch()
    mask = datums[0].loss_fn_inputs["mask"].to_torch()
    comp = torch.where(mask > 0)[0]

    # advantage = teacher_lp - student_lp at completion positions.
    assert torch.allclose(adv[comp], torch.tensor([0.5, 1.5]), atol=1e-6)
    assert torch.allclose(lp[comp], torch.tensor([-1.0, -2.0]), atol=1e-6)
    assert metrics["sdft/is_positions"] == 2.0
    assert abs(metrics["sdft/mean_advantage"] - 1.0) < 1e-6

"""Tests for `evsys_sdk.sweep.Sweep`.

Sweep is the OOP counterpart to the YAML `matrix:` block — same expansion
semantics, used directly from Python by Experiment scripts.
"""

from __future__ import annotations

import pytest

from evsys_sdk.config import (
    AlgorithmConfig,
    DataConfig,
    MatrixSpec,
    ModelConfig,
    RunConfig,
)
from evsys_sdk.sweep import Sweep, expand_runs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_run() -> RunConfig:
    return RunConfig(
        name="base",
        data=DataConfig(source_kind="in_memory", rows=[{"q": "Q", "a": "A"}]),
        model=ModelConfig(name="Qwen/Qwen3-4B"),
        algorithm=AlgorithmConfig(
            kind="mock_sft",
            params={"lora_rank": 0, "learning_rate": 1e-4},
        ),
    )


# ---------------------------------------------------------------------------
# expand() — single axis
# ---------------------------------------------------------------------------


def test_single_axis_one_run_per_value(base_run: RunConfig):
    s = Sweep(base_run, {"algorithm.params.lora_rank": [1, 4, 16]})
    runs = s.expand()
    assert len(runs) == 3
    assert [r.algorithm.params["lora_rank"] for r in runs] == [1, 4, 16]
    # base values not in the axis are preserved
    assert all(r.algorithm.params["learning_rate"] == 1e-4 for r in runs)


def test_len_matches_expand_size(base_run: RunConfig):
    s = Sweep(base_run, {"algorithm.params.lora_rank": [1, 4, 16]})
    assert len(s) == 3
    assert len(s.expand()) == 3


def test_len_with_no_axes_is_one(base_run: RunConfig):
    assert len(Sweep(base_run)) == 1


# ---------------------------------------------------------------------------
# expand() — multi-axis cartesian product
# ---------------------------------------------------------------------------


def test_multi_axis_cartesian(base_run: RunConfig):
    s = Sweep(
        base_run,
        {
            "algorithm.params.lora_rank": [1, 4],
            "algorithm.params.learning_rate": [1e-4, 5e-5],
        },
    )
    runs = s.expand()
    assert len(s) == 4
    assert len(runs) == 4
    combos = {(r.algorithm.params["lora_rank"], r.algorithm.params["learning_rate"])
              for r in runs}
    assert combos == {(1, 1e-4), (1, 5e-5), (4, 1e-4), (4, 5e-5)}


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def test_default_name_suffix_is_deterministic(base_run: RunConfig):
    s = Sweep(base_run, {"algorithm.params.lora_rank": [1, 4]})
    runs = s.expand()
    assert runs[0].name == "base__lora_rank1"
    assert runs[1].name == "base__lora_rank4"


def test_default_name_suffix_floats_use_p_separator(base_run: RunConfig):
    s = Sweep(base_run, {"algorithm.params.learning_rate": [1e-4]})
    runs = s.expand()
    # 1e-4 → "0.0001" → "0p0001"
    assert "learning_rate0p0001" in runs[0].name


def test_name_template_renders_dotted_keys(base_run: RunConfig):
    s = Sweep(
        base_run,
        {"algorithm.params.lora_rank": [1, 4]},
        name_template="{base}__r{algorithm.params.lora_rank}",
    )
    runs = s.expand()
    assert runs[0].name == "base__r1"
    assert runs[1].name == "base__r4"


def test_name_template_unknown_key_raises(base_run: RunConfig):
    s = Sweep(
        base_run,
        {"algorithm.params.lora_rank": [1]},
        name_template="{base}__{not.a.real.axis}",
    )
    with pytest.raises(ValueError, match="unknown key"):
        s.expand()


# ---------------------------------------------------------------------------
# Edge cases / validation
# ---------------------------------------------------------------------------


def test_no_axes_returns_one_copy_of_base(base_run: RunConfig):
    s = Sweep(base_run)
    runs = s.expand()
    assert len(runs) == 1
    assert runs[0].name == "base"
    # different object — independent of the original
    assert runs[0] is not base_run


def test_empty_axis_values_rejected(base_run: RunConfig):
    with pytest.raises(ValueError, match="non-empty list"):
        Sweep(base_run, {"algorithm.params.lora_rank": []})


def test_invalid_axis_value_type_rejected(base_run: RunConfig):
    with pytest.raises(ValueError, match="non-empty list"):
        Sweep(base_run, {"algorithm.params.lora_rank": 1})  # type: ignore[arg-type]


def test_expanded_runs_are_independent(base_run: RunConfig):
    s = Sweep(base_run, {"algorithm.params.lora_rank": [1, 4]})
    runs = s.expand()
    runs[0].algorithm.params["lora_rank"] = 999
    # base_run and the other run are untouched
    assert base_run.algorithm.params["lora_rank"] == 0
    assert runs[1].algorithm.params["lora_rank"] == 4


def test_dotted_axis_traverses_nested_dict(base_run: RunConfig):
    """Setting algorithm.params.x walks AlgorithmConfig.params (a dict)."""
    s = Sweep(base_run, {"algorithm.params.lora_rank": [42]})
    [run] = s.expand()
    assert run.algorithm.params["lora_rank"] == 42


# ---------------------------------------------------------------------------
# Matrix bridge
# ---------------------------------------------------------------------------


def test_to_matrix_round_trip(base_run: RunConfig):
    s = Sweep(
        base_run,
        {"algorithm.params.lora_rank": [1, 4]},
        name_template="{base}__r{algorithm.params.lora_rank}",
    )
    m = s.to_matrix()
    assert isinstance(m, MatrixSpec)
    s2 = Sweep.from_matrix(m)
    assert s2.axes == s.axes
    assert s2.name_template == s.name_template
    # Re-expand: same names
    assert [r.name for r in s.expand()] == [r.name for r in s2.expand()]


def test_from_matrix_preserves_base_run(base_run: RunConfig):
    m = MatrixSpec(base_run=base_run, axes={"algorithm.params.lora_rank": [1]})
    s = Sweep.from_matrix(m)
    assert s.base_run.name == "base"


# ---------------------------------------------------------------------------
# expand_runs as a standalone function (used by yaml_loader)
# ---------------------------------------------------------------------------


def test_expand_runs_public_fn_matches_sweep(base_run: RunConfig):
    axes = {"algorithm.params.lora_rank": [1, 4]}
    via_fn = expand_runs(base_run, axes)
    via_sweep = Sweep(base_run, axes).expand()
    assert [r.name for r in via_fn] == [r.name for r in via_sweep]
    assert [r.algorithm.params["lora_rank"] for r in via_fn] == \
           [r.algorithm.params["lora_rank"] for r in via_sweep]


def test_expand_runs_with_no_axes_returns_base_copy(base_run: RunConfig):
    runs = expand_runs(base_run, {})
    assert len(runs) == 1
    assert runs[0] is not base_run
    assert runs[0].name == "base"


def test_expand_runs_dotted_through_pydantic_attr(base_run: RunConfig):
    """A dotted path that traverses a Pydantic model attribute (not a dict)
    should still resolve. We sweep over model.name, which is on ModelConfig."""
    runs = expand_runs(base_run, {"model.name": ["A", "B"]})
    assert [r.model.name for r in runs] == ["A", "B"]


def test_expand_runs_dotted_path_through_dict_setdefault(base_run: RunConfig):
    """Dotted paths into algorithm.params should walk through the params
    dict (which is what real users do — set algorithm.params.X)."""
    # base has algorithm.params already populated. Test path > 2 segments.
    runs = expand_runs(base_run, {"algorithm.params.new_key": ["v1", "v2"]})
    assert runs[0].algorithm.params["new_key"] == "v1"
    assert runs[1].algorithm.params["new_key"] == "v2"


def test_expand_runs_value_with_slash_is_sanitized_in_name(base_run: RunConfig):
    """Slashes in HF model ids would break filesystem paths if they got
    into run names. The default suffix sanitizes them to underscores."""
    runs = expand_runs(base_run, {"model.name": ["Qwen/Qwen3-4B"]})
    assert "/" not in runs[0].name
    assert "Qwen_Qwen3-4B" in runs[0].name

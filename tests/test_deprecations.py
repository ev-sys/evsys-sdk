"""Verify the legacy tinker_* algorithm wrappers emit a DeprecationWarning
pointing at their `native_*` replacements.

Researchers grepping their CI logs for DeprecationWarning should land
directly on the migration path.
"""

from __future__ import annotations

import warnings

import pytest


def test_tinker_sft_emits_deprecation():
    from evsys_sdk.algorithms.tinker_sft import TinkerSFT
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TinkerSFT()
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, DeprecationWarning)]
    assert any("native_sft" in m for m in msgs)


def test_tinker_sdft_emits_deprecation():
    from evsys_sdk.algorithms.tinker_sdft import TinkerSDFT
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TinkerSDFT()
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, DeprecationWarning)]
    assert any("native_sdft" in m for m in msgs)


def test_tinker_rl_emits_deprecation():
    from evsys_sdk.algorithms.tinker_rl import TinkerRL
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TinkerRL()
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, DeprecationWarning)]
    assert any("native_rl" in m for m in msgs)


def test_native_sft_emits_no_deprecation():
    """The replacements should NOT emit a warning themselves."""
    from evsys_sdk.algorithms.native_sft import NativeSFT
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        NativeSFT()
    assert not [w for w in caught
                if issubclass(w.category, DeprecationWarning)]

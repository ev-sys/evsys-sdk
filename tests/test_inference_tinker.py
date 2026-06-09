"""Tests for ``evsys_sdk.inference.tinker.TinkerInference.from_run_result``
and its registered default-factory.

The classmethod reads ``run_result.artifacts["run_dir"]``, locates the
checkpoints manifest, picks the final sampler checkpoint, and constructs a
``TinkerInference`` with the right (model_name, checkpoint_path). Real
``TinkerInference.__init__`` makes network calls — these tests monkeypatch
it to capture args without standing up a real Tinker session.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evsys_sdk.registry import get_default_inference_factory


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _Res:
    def __init__(self, run_dir: str | None) -> None:
        self.artifacts = {"run_dir": run_dir} if run_dir is not None else {}


class _ModelCfg:
    def __init__(self, name: str = "Qwen/Qwen3.5-4B") -> None:
        self.name = name


class _RunCfg:
    def __init__(self, model_name: str = "Qwen/Qwen3.5-4B") -> None:
        self.model = _ModelCfg(model_name)


def _stub_init(monkeypatch, captured: dict) -> None:
    """Replace ``TinkerInference.__init__`` with a capture-only stub so we
    can exercise ``from_run_result`` without making any real Tinker calls."""
    from evsys_sdk.inference.tinker import TinkerInference

    def fake_init(self, *, model_name: str, checkpoint_path: str | None = None,
                  api_key_env: str = "TINKER_API_KEY") -> None:
        captured["model_name"] = model_name
        captured["checkpoint_path"] = checkpoint_path

    monkeypatch.setattr(TinkerInference, "__init__", fake_init)


def _write_manifest(dir: Path, rows: list[dict]) -> None:
    (dir / "checkpoints.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_from_run_result_happy_path(tmp_path: Path, monkeypatch):
    """A real-looking manifest → constructor called with the final sampler path."""
    _write_manifest(tmp_path, [
        {"label": "step_500", "step": 500,
         "sampler_path": "tinker://abc/sampler_weights/500"},
        {"label": "final", "step": 1000,
         "sampler_path": "tinker://abc/sampler_weights/final"},
    ])
    captured: dict = {}
    _stub_init(monkeypatch, captured)

    from evsys_sdk.inference.tinker import TinkerInference
    inst = TinkerInference.from_run_result(_Res(str(tmp_path)), _RunCfg())
    assert isinstance(inst, TinkerInference)
    assert captured["model_name"] == "Qwen/Qwen3.5-4B"
    assert captured["checkpoint_path"] == "tinker://abc/sampler_weights/final"


def test_from_run_result_threads_run_cfg_model_name(tmp_path: Path, monkeypatch):
    """The model name in run_cfg.model.name must be forwarded verbatim."""
    _write_manifest(tmp_path, [
        {"label": "final", "step": 1, "sampler_path": "tinker://x/sampler/final"},
    ])
    captured: dict = {}
    _stub_init(monkeypatch, captured)

    from evsys_sdk.inference.tinker import TinkerInference
    TinkerInference.from_run_result(_Res(str(tmp_path)),
                                    _RunCfg("Qwen/Qwen3.5-9B"))
    assert captured["model_name"] == "Qwen/Qwen3.5-9B"


# ---------------------------------------------------------------------------
# Error paths — each missing piece raises a clear RuntimeError
# ---------------------------------------------------------------------------


def test_from_run_result_missing_run_dir_raises(monkeypatch):
    """No artifacts['run_dir'] → RuntimeError naming the missing key."""
    captured: dict = {}
    _stub_init(monkeypatch, captured)

    from evsys_sdk.inference.tinker import TinkerInference
    with pytest.raises(RuntimeError, match="no 'run_dir'"):
        TinkerInference.from_run_result(_Res(None), _RunCfg())


def test_from_run_result_missing_manifest_raises(tmp_path: Path, monkeypatch):
    """run_dir exists but has no checkpoints.jsonl → RuntimeError naming the dir."""
    captured: dict = {}
    _stub_init(monkeypatch, captured)

    from evsys_sdk.inference.tinker import TinkerInference
    with pytest.raises(RuntimeError, match="no checkpoints.jsonl"):
        TinkerInference.from_run_result(_Res(str(tmp_path)), _RunCfg())


def test_from_run_result_no_sampler_path_raises(tmp_path: Path, monkeypatch):
    """Manifest present but no checkpoint has a sampler_path → RuntimeError."""
    _write_manifest(tmp_path, [
        {"label": "final", "step": 100},  # no sampler_path field
    ])
    captured: dict = {}
    _stub_init(monkeypatch, captured)

    from evsys_sdk.inference.tinker import TinkerInference
    with pytest.raises(RuntimeError, match="no usable sampler checkpoint"):
        TinkerInference.from_run_result(_Res(str(tmp_path)), _RunCfg())


# ---------------------------------------------------------------------------
# Registry plumbing
# ---------------------------------------------------------------------------


def test_default_factory_registered_for_tinker():
    """Importing inference.tinker registers a default factory keyed 'tinker'."""
    # Ensure the import side-effect fires
    from evsys_sdk.inference.tinker import TinkerInference  # noqa: F401

    fac = get_default_inference_factory("tinker")
    assert fac is not None and callable(fac)


def test_default_factory_returns_none_for_unregistered_kind():
    """Backends that haven't registered (e.g. mock) return None."""
    assert get_default_inference_factory("mock") is None
    assert get_default_inference_factory("__never_registered__") is None

"""Gated real-Modal smoke for the ``modal`` sandbox provider.

Skipped unless Modal credentials resolve (``~/.modal.toml`` or
MODAL_TOKEN_ID/MODAL_TOKEN_SECRET) and the `modal` package is installed. This
is the test the stub-based unit tests cannot be: it caught the double-`start()`
that orphaned a live, billed sandbox.

Run explicitly:  pytest tests/test_sandbox_modal_smoke.py -q
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

modal = pytest.importorskip("modal", reason="needs `pip install evsys-sdk[remote-modal]`")

from evsys_sdk.sandboxes import build_sandbox  # noqa: E402

APP = "evsys-agents-smoke"


def _has_credentials() -> bool:
    try:
        modal.App.lookup(APP, create_if_missing=True)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_credentials(), reason="no usable Modal credentials")


def _live_sandbox_ids() -> set[str]:
    return {sb.object_id for sb in modal.Sandbox.list()}


def test_copy_in_run_copy_out_against_real_modal():
    host_dir = Path(tempfile.mkdtemp())
    artifact = host_dir / "prompt.txt"
    artifact.write_text("seed")
    baseline = {"prompt.txt": "seed", "notes.md": "untouched"}

    before = _live_sandbox_ids()
    lines: list[str] = []
    with build_sandbox({"kind": "modal", "params": {"app_name": APP}},
                       envs={"EVSYS_TEST": "from-host"}, timeout_s=120) as sbx:
        created = _live_sandbox_ids() - before
        sbx.stage(baseline)
        assert sbx.read(sbx.path("prompt.txt")) == "seed"
        assert sbx.read(sbx.path("nope.txt")) is None

        code, _ = sbx.exec(
            "echo $EVSYS_TEST; printf 'IMPROVED' > prompt.txt; echo rewritten",
            timeout_s=60, on_line=lines.append)
        assert code == 0
        assert "from-host" in lines and "rewritten" in lines   # env + live streaming

        sbx.setup("echo provisioning ok", required=True)
        landed = sbx.collect(
            [("prompt.txt", artifact), ("notes.md", host_dir / "notes.md")], baseline)

    assert landed == ["prompt.txt"]
    assert artifact.read_text() == "IMPROVED"
    assert not (host_dir / "notes.md").exists()   # unchanged file never round-trips

    # Exactly ONE sandbox for the whole `with`, and it is gone afterwards.
    assert len(created) == 1
    assert not (created & _live_sandbox_ids()), "sandbox outlived the context manager"


def test_nonzero_exit_is_reported():
    with build_sandbox({"kind": "modal", "params": {"app_name": APP}},
                       timeout_s=120) as sbx:
        code, out = sbx.exec("echo boom >&2; exit 3", timeout_s=60)
    assert code == 3 and "boom" in out

"""Tests for `evsys report push` / report_upload.push_report."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evsys_sdk.cli import main as cli_main
from evsys_sdk.report_upload import ReportPushResult, push_report


@pytest.fixture()
def report_dir(tmp_path: Path) -> Path:
    root = tmp_path / "eval-report"
    root.mkdir()
    (root / "index.html").write_text("<html><body>hi</body></html>")
    assets = root / "assets"
    assets.mkdir()
    (assets / "app.css").write_text("body { color: #111; }")
    return root


def test_push_report_zips_and_posts(report_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVSYS_API_KEY", "sk_test")
    monkeypatch.setenv("EVSYS_PROJECT_ID", "proj-1")
    monkeypatch.setenv("EVSYS_API_URL", "http://example.test")

    fake_resp = MagicMock()
    fake_resp.status_code = 201
    fake_resp.json.return_value = {
        "report_id": "rep-1",
        "path": "experiments/run-1/eval",
        "content_hash": "abc",
        "n_files": 2,
        "size_bytes": 100,
        "entry_file": "index.html",
        "url": "http://localhost:3000/projects/proj-1/reports?report=rep-1",
    }
    fake_resp.text = json.dumps(fake_resp.json.return_value)

    with patch("evsys_sdk.report_upload.requests.post", return_value=fake_resp) as post:
        result = push_report(report_dir, path="experiments/run-1/eval")

    assert isinstance(result, ReportPushResult)
    assert result.report_id == "rep-1"
    assert result.n_files == 2
    assert "reports?report=rep-1" in result.url
    assert post.called
    kwargs = post.call_args.kwargs
    assert kwargs["data"]["project_id"] == "proj-1"
    assert kwargs["data"]["path"] == "experiments/run-1/eval"
    assert "file" in kwargs["files"]
    # zip payload non-empty
    assert len(kwargs["files"]["file"][1]) > 0


def test_push_requires_entry(report_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVSYS_API_KEY", "sk_test")
    monkeypatch.setenv("EVSYS_PROJECT_ID", "proj-1")
    with pytest.raises(ValueError, match="entry file"):
        push_report(report_dir, path="x/y", entry_file="missing.html")


def test_cli_report_push(report_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setenv("EVSYS_API_KEY", "sk_test")
    monkeypatch.setenv("EVSYS_PROJECT_ID", "proj-1")
    monkeypatch.setenv("EVSYS_API_URL", "http://example.test")

    fake = ReportPushResult(
        report_id="rep-9",
        path="a/b",
        content_hash="h",
        n_files=2,
        size_bytes=50,
        entry_file="index.html",
        url="http://localhost:3000/projects/p/reports?report=rep-9",
    )
    with patch("evsys_sdk.report_upload.push_report", return_value=fake) as push:
        code = cli_main([
            "report", "push", str(report_dir),
            "--path", "a/b",
        ])
    assert code == 0
    assert push.called
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["report_id"] == "rep-9"

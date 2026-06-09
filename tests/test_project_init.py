"""Tests for `evsys_sdk.project_init.init_project` and the
`evsys init-project` CLI subcommand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from evsys_sdk.cli import main as cli_main
from evsys_sdk.project_init import (
    GITKEEP_DIRS,
    SCAFFOLD_DIRS,
    init_project,
)


# ---------------------------------------------------------------------------
# init_project — golden tree
# ---------------------------------------------------------------------------


_EXPECTED_FILES = (
    "pyproject.toml",
    "README.md",
    ".gitignore",
    "data/README.md",
    "scripts/__init__.py",
    "scripts/verifiers.py",
    "scripts/metrics.py",
    "scripts/transforms.py",
)

_EXPECTED_GITKEEPS = tuple(f"{d}/.gitkeep" for d in GITKEEP_DIRS)


def test_init_project_creates_expected_tree(tmp_path: Path):
    path = init_project(tmp_path / "demo")

    assert path == (tmp_path / "demo").resolve()

    # Every scaffold dir exists.
    for sub in SCAFFOLD_DIRS:
        assert (path / sub).is_dir(), f"missing dir: {sub}"

    # Every templated file exists.
    for rel in _EXPECTED_FILES + _EXPECTED_GITKEEPS:
        assert (path / rel).is_file(), f"missing file: {rel}"


def test_init_project_pyproject_uses_name(tmp_path: Path):
    init_project(tmp_path / "my_proj")
    py = (tmp_path / "my_proj" / "pyproject.toml").read_text()
    assert 'name = "my_proj"' in py
    assert 'packages = ["scripts"]' in py


def test_init_project_uses_explicit_name(tmp_path: Path):
    init_project(tmp_path / "any_dir", name="explicit_name")
    py = (tmp_path / "any_dir" / "pyproject.toml").read_text()
    assert 'name = "explicit_name"' in py


def test_readme_references_layout(tmp_path: Path):
    init_project(tmp_path / "demo")
    readme = (tmp_path / "demo" / "README.md").read_text()
    for tok in ("data/", "scripts/", "experiments/", "evsys new-experiment",
                "evsys benchmark upload"):
        assert tok in readme


def test_gitignore_excludes_local_dirs(tmp_path: Path):
    init_project(tmp_path / "demo")
    gi = (tmp_path / "demo" / ".gitignore").read_text()
    assert ".evsys/" in gi
    assert "data/raw/" in gi


def test_scripts_init_imports_extension_modules(tmp_path: Path):
    init_project(tmp_path / "demo")
    init = (tmp_path / "demo" / "scripts" / "__init__.py").read_text()
    assert "from . import verifiers" in init
    assert "from . import metrics" in init
    assert "from . import transforms" in init


def test_scripts_package_is_importable(tmp_path: Path, monkeypatch):
    """Importing the scaffolded scripts package must not raise."""
    path = init_project(tmp_path / "demo")
    monkeypatch.syspath_prepend(str(path))
    monkeypatch.delitem(sys.modules, "scripts", raising=False)
    monkeypatch.delitem(sys.modules, "scripts.verifiers", raising=False)
    monkeypatch.delitem(sys.modules, "scripts.metrics", raising=False)
    monkeypatch.delitem(sys.modules, "scripts.transforms", raising=False)
    import scripts  # noqa: F401
    # The commented examples shouldn't accidentally execute.
    assert scripts.__doc__


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_init_into_empty_existing_dir(tmp_path: Path):
    target = tmp_path / "preexisting"
    target.mkdir()
    init_project(target)
    assert (target / "scripts" / "__init__.py").is_file()


def test_refuses_nonempty_dir_without_force(tmp_path: Path):
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "existing_file.txt").write_text("hi")
    with pytest.raises(FileExistsError, match="not empty"):
        init_project(target)


def test_force_overwrites_into_nonempty_dir(tmp_path: Path):
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "existing_file.txt").write_text("hi")
    init_project(target, force=True)
    # Existing file preserved (we don't overwrite without good reason)
    assert (target / "existing_file.txt").read_text() == "hi"
    # But scaffold files now exist
    assert (target / "scripts" / "__init__.py").is_file()


def test_force_does_not_clobber_existing_pyproject(tmp_path: Path):
    """A user's own pyproject.toml stays in place — force scaffolds *around* it."""
    target = tmp_path / "with_py"
    target.mkdir()
    (target / "pyproject.toml").write_text('[project]\nname = "user_owned"\n')
    init_project(target, force=True)
    assert 'user_owned' in (target / "pyproject.toml").read_text()


def test_refuses_if_target_is_a_file(tmp_path: Path):
    target = tmp_path / "file"
    target.write_text("not a dir")
    with pytest.raises(FileExistsError, match="is a file"):
        init_project(target)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_init_project_happy(tmp_path: Path, capsys):
    rc = cli_main(["init-project", str(tmp_path / "via_cli")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "scaffolded research project" in out
    assert (tmp_path / "via_cli" / "scripts" / "__init__.py").is_file()


def test_cli_init_project_refuses_nonempty(tmp_path: Path, capsys):
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "x.txt").touch()
    rc = cli_main(["init-project", str(target)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR" in err


def test_cli_init_project_force(tmp_path: Path):
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "x.txt").touch()
    rc = cli_main(["init-project", str(target), "--force"])
    assert rc == 0
    assert (target / "scripts" / "__init__.py").is_file()


def test_cli_init_project_with_name(tmp_path: Path):
    rc = cli_main(["init-project", str(tmp_path / "dirname"), "--name", "PRETTY"])
    assert rc == 0
    assert 'name = "PRETTY"' in (tmp_path / "dirname" / "pyproject.toml").read_text()

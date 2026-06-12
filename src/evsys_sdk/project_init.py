"""Scaffold a new EvolvingSystems research project on disk.

Emits the locked directory layout (see ``docs/DESIGN.md`` — researcher-project
layout) so every project the training-decider agent bootstraps has the same
predictable shape:

    <name>/
      pyproject.toml
      README.md
      .gitignore
      data/
        README.md
        raw/.gitkeep
        fetch/.gitkeep
        process/.gitkeep
        datasets/.gitkeep
        benchmark/.gitkeep
      src/
        __init__.py
        verifiers.py
        metrics.py
        transforms.py
      experiments/.gitkeep

Use via the CLI: ``evsys init-project <name>``. Programmatically: call
``init_project(path, name=...)``.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# File-tree spec — kept declarative so golden-file tests can assert against it.
# ---------------------------------------------------------------------------


SCAFFOLD_DIRS = (
    "data",
    "data/raw",
    "data/fetch",
    "data/process",
    "data/datasets",
    "data/benchmark",
    "data/validation",
    "src",
    "experiments",
)

GITKEEP_DIRS = (
    "data/raw",
    "data/fetch",
    "data/process",
    "data/datasets",
    "data/benchmark",
    "data/validation",
    "experiments",
)


def init_project(target: str | Path, *, name: str | None = None, force: bool = False) -> Path:
    """Create a new research project skeleton under ``target``.

    If ``target`` exists and is non-empty, refuse unless ``force=True``.
    Returns the absolute project path.
    """
    project_path = Path(target).expanduser().resolve()
    project_name = name or project_path.name

    if project_path.exists():
        if project_path.is_file():
            raise FileExistsError(f"{project_path} exists and is a file, not a directory")
        if any(project_path.iterdir()) and not force:
            raise FileExistsError(
                f"{project_path} is not empty (pass force=True to scaffold into it)"
            )
    else:
        project_path.mkdir(parents=True)

    for sub in SCAFFOLD_DIRS:
        (project_path / sub).mkdir(parents=True, exist_ok=True)
    for sub in GITKEEP_DIRS:
        (project_path / sub / ".gitkeep").touch(exist_ok=True)

    _write(project_path / "pyproject.toml", _pyproject_toml(project_name))
    _write(project_path / "README.md", _readme(project_name))
    _write(project_path / ".gitignore", _gitignore())
    _write(project_path / "data" / "README.md", _data_readme())
    _write(project_path / "src" / "__init__.py", _src_init(project_name))
    _write(project_path / "src" / "verifiers.py", _src_verifiers())
    _write(project_path / "src" / "metrics.py", _src_metrics())
    _write(project_path / "src" / "transforms.py", _src_transforms())

    return project_path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    # Never overwrite a file the user has already touched, even with force.
    # `force` is for "fill in gaps in a non-empty dir", not "clobber my work".
    if path.exists():
        return
    path.write_text(content)


def _pyproject_toml(name: str) -> str:
    return f'''\
[project]
name = "{name}"
version = "0.1.0"
description = "EvolvingSystems research project."
requires-python = ">=3.11"
dependencies = [
  "evsys-sdk",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
'''


def _readme(name: str) -> str:
    return f'''\
# {name}

A EvolvingSystems research project.

## Layout
```
data/         # raw → processed → versioned datasets; harbor-format benchmarks
src/          # project-specific SDK extensions (verifiers, metrics, transforms)
experiments/  # one date-prefixed dir per experiment (config.yaml + run.py)
```

## Common commands
```
evsys new-experiment <slug>             # scaffold experiments/<today>_<slug>/
evsys benchmark upload data/benchmark/<name>  # register a benchmark with the dashboard
python experiments/<dir>/run.py          # run an experiment
```

See `docs/DESIGN.md` in `evsys-sdk` for the layout rationale.
'''


def _gitignore() -> str:
    return '''\
__pycache__/
*.pyc
.venv/
.pytest_cache/

# EvolvingSystems local mirror + cached datasets, checkpoints, log_store output
.evsys/

# Untracked source dumps — usually too large for git
data/raw/
'''


def _data_readme() -> str:
    return '''\
# data/

Lineage flows top to bottom — never rewrite earlier stages.

  raw/        Source dumps, untouched. Gitignored (usually large).
  fetch/      Python scripts that populate raw/ from external sources.
  process/    Python scripts that turn raw/ into datasets/<name>/v<N>/.
  datasets/   Versioned training/test JSONL — chat_messages or harbor_task rows.
              Each dataset is data/datasets/<name>/v<N>/{train,test}.jsonl
              + metadata.yaml (source, parent version, row count, schema hash).
  benchmark/  Harbor-format eval suites. Each is data/benchmark/<name>/tasks.jsonl
              + optional images/ + raw/ + metadata.yaml. Register with
              `evsys benchmark upload data/benchmark/<name>`.
'''


def _src_init(name: str) -> str:
    return f'''\
"""Project-specific SDK extensions for {name}.

Importing this package registers all custom verifiers / metrics / transforms
with the SDK registries so YAML configs and the Experiment runner can find
them by name. Experiment `run.py` scripts do ``import src`` so the
decorators fire at startup.
"""
from . import verifiers  # noqa: F401
from . import metrics    # noqa: F401
from . import transforms # noqa: F401
'''


def _src_verifiers() -> str:
    return '''\
"""Project-specific verifiers.

Built-in verifier functions (exact_match, contains, regex_match,
tool_calls_match) live in the SDK and are referenced by name from harbor
task JSONL. Add a custom one below when those don't fit.

Two ways to register, depending on whether you need a Pydantic-config'd
Verifier class (registered via @register_verifier) or a plain in-process
verifier-fn (registered via @register_verifier_fn).
"""
# from typing import Any, ClassVar
# from pydantic import BaseModel
# from evsys_sdk.registry import register_verifier
# from evsys_sdk.protocols import VerificationResult
# from evsys_sdk.verifiers import register_verifier_fn
#
#
# @register_verifier_fn("my_match")
# def my_match(model_output: str, expected: Any, params: dict) -> float:
#     return 1.0 if model_output.strip() == str(expected).strip() else 0.0
'''


def _src_metrics() -> str:
    return '''\
"""Project-specific metric aggregators.

Built-in metrics (exact_match, pass_at_k, mean_reward, toolkit_match) live
in the SDK. Add custom ones here when those don't fit your benchmark.
"""
# from typing import Any, ClassVar
# from pydantic import BaseModel
# from evsys_sdk.registry import register_metric
#
#
# @register_metric("my_metric")
# class MyMetric:
#     name: ClassVar[str] = "my_metric"
#     Config: ClassVar[type] = BaseModel
#
#     def compute(self, *, predictions: list[dict], targets: list[dict]) -> float:
#         ...
'''


def _src_transforms() -> str:
    return '''\
"""Project-specific data transforms — convert raw rows to algorithm-ready rows.

A transform is registered with `@register_transform("name")` and chains in
`data.transforms` in a YAML config.
"""
# from typing import ClassVar, Iterable
# from pydantic import BaseModel
# from evsys_sdk.registry import register_transform
#
#
# @register_transform("my_transform")
# class MyTransform:
#     name: ClassVar[str] = "my_transform"
#     Config: ClassVar[type] = BaseModel
#
#     def __call__(self, rows: Iterable[dict]) -> Iterable[dict]:
#         for r in rows:
#             yield r
'''


__all__ = ["GITKEEP_DIRS", "SCAFFOLD_DIRS", "init_project"]

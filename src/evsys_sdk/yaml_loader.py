"""YAML <-> ExperimentConfig.

The YAML is canonical — Python builders just emit the same dict tree.

Validation has two layers:
  1. Top-level structure validated by ExperimentConfig (Pydantic, strict).
  2. Each `kind:` block has its `params:` validated against the registered
     extension's .Config model.

Step 2 happens lazily inside the runner (on actual use) so unknown extensions
discovered via entry points don't fail validation prematurely. But you can
call ``validate_yaml(path, *, deep=True)`` to force step 2 up front.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig, MatrixSpec, RunConfig
from .registry import _all_registries
from .sweep import expand_runs


def _read_yaml(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}")
    return data


def load_yaml(source: str | Path | dict[str, Any]) -> ExperimentConfig:
    """Parse and validate a YAML experiment file."""
    data = _read_yaml(source)
    cfg = ExperimentConfig.model_validate(data)
    if cfg.matrix is not None:
        cfg = _expand_matrix(cfg)
    return cfg


def dump_yaml(cfg: ExperimentConfig, *, path: str | Path | None = None) -> str:
    """Serialize an ExperimentConfig back to YAML."""
    data = cfg.model_dump(exclude_none=True, mode="json")
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    if path is not None:
        Path(path).write_text(text)
    return text


def validate_yaml(
    source: str | Path | dict[str, Any], *, deep: bool = False
) -> list[str]:
    """Return a list of validation errors. Empty list = valid.

    If ``deep`` is True, also validate every kind/params block against its
    registered .Config model.
    """
    errors: list[str] = []
    try:
        cfg = load_yaml(source)
    except Exception as e:
        errors.append(f"top-level: {e}")
        return errors

    if not deep:
        return errors

    runs: list[RunConfig] = []
    if cfg.run is not None:
        runs.append(cfg.run)
    if cfg.runs is not None:
        runs.extend(cfg.runs)

    regs = _all_registries()
    for r in runs:
        prefix = f"run '{r.name}'"
        # Algorithm
        try:
            cls = regs["algorithm"].get(r.algorithm.kind)
            if hasattr(cls, "Config"):
                cls.Config.model_validate(r.algorithm.params)
        except Exception as e:
            errors.append(f"{prefix} algorithm: {e}")
        # Backend (optional Config)
        try:
            cls = regs["backend"].get(r.backend.kind)
            if hasattr(cls, "Config"):
                cls.Config.model_validate(r.backend.params)
        except Exception as e:
            errors.append(f"{prefix} backend: {e}")
        # Transforms
        for i, t in enumerate(r.data.transforms):
            try:
                cls = regs["transform"].get(t.kind)
                if hasattr(cls, "Config"):
                    cls.Config.model_validate(t.params)
            except Exception as e:
                errors.append(f"{prefix} data.transforms[{i}]: {e}")
    return errors


# ---------------------------------------------------------------------------
# Matrix expansion — delegates to the canonical impl in sweep.expand_runs
# so the YAML matrix and programmatic Sweep share one expansion path.
# ---------------------------------------------------------------------------


def _expand_matrix(cfg: ExperimentConfig) -> ExperimentConfig:
    """Replace ``cfg.matrix`` with ``cfg.runs``, preserving everything else."""
    assert cfg.matrix is not None
    matrix: MatrixSpec = cfg.matrix
    runs = expand_runs(matrix.base_run, matrix.axes, matrix.name_template)
    return cfg.model_copy(update={"matrix": None, "runs": runs})

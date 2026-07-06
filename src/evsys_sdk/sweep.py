"""Sweep — declarative axes over a single `RunConfig`.

A ``Sweep`` is the OOP way to describe what YAML's ``matrix:`` block describes:
one ``base_run`` template + per-axis lists of values, expanded by cartesian
product into N concrete ``RunConfig`` rows.

This is what an experiment script reaches for instead of writing a per-axis
``for`` loop. Sweep + Experiment together let ``run.py`` stay declarative:

    sweep = Sweep(base_run, {"algorithm.params.lora_rank": [1, 4, 16]})
    for run in sweep.expand():
        ...

Or, more commonly, from a YAML matrix block already parsed into
``MatrixSpec`` — round-trip with ``Sweep.from_matrix`` / ``.to_matrix``.

The expansion is the single source of truth for both the YAML ``matrix:``
loader and programmatic builds (the legacy ``_expand_matrix`` delegates here).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from .config import MatrixSpec, RunConfig

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class Sweep:
    """Declarative axes over a single ``RunConfig``.

    Args:
      base_run: The template every arm starts from.
      axes: ``{dotted.key: [values, ...]}`` — at least one entry; each list
            must be non-empty.
      name_template: Optional ``"{base}__rank{algorithm.params.lora_rank}"``
            style template. When absent, a deterministic default suffix is
            appended to ``base_run.name``.
    """

    base_run: RunConfig
    axes: dict[str, list[Any]] = field(default_factory=dict)
    name_template: str | None = None

    def __post_init__(self) -> None:
        for axis, vals in self.axes.items():
            if not isinstance(vals, list) or len(vals) == 0:
                raise ValueError(
                    f"Sweep axis {axis!r} must be a non-empty list of values"
                )

    # -- expansion --------------------------------------------------------

    def expand(self) -> list[RunConfig]:
        """Cartesian product over axes → one ``RunConfig`` per cell."""
        return expand_runs(self.base_run, self.axes, self.name_template)

    def __len__(self) -> int:
        if not self.axes:
            return 1
        n = 1
        for vals in self.axes.values():
            n *= len(vals)
        return n

    # -- matrix bridge ----------------------------------------------------

    def to_matrix(self) -> MatrixSpec:
        """Repackage as a ``MatrixSpec`` for YAML serialization."""
        return MatrixSpec(
            base_run=self.base_run,
            axes=dict(self.axes),
            name_template=self.name_template,
        )

    @classmethod
    def from_matrix(cls, matrix: MatrixSpec) -> Sweep:
        return cls(
            base_run=matrix.base_run,
            axes=dict(matrix.axes),
            name_template=matrix.name_template,
        )


def expand_runs(
    base_run: RunConfig,
    axes: dict[str, list[Any]],
    name_template: str | None = None,
) -> list[RunConfig]:
    """Cartesian product of ``axes`` over ``base_run``.

    Single source of truth for matrix expansion — both ``Sweep.expand`` and
    the YAML loader's ``_expand_matrix`` route through here. Empty ``axes``
    returns ``[base_run]`` unchanged.
    """
    if not axes:
        return [base_run.model_copy(deep=True)]

    axis_names = list(axes.keys())
    axis_values = [axes[a] for a in axis_names]

    runs: list[RunConfig] = []
    for combo in product(*axis_values):
        new_dict = base_run.model_dump()
        binding: dict[str, str] = {"base": base_run.name}
        for axis, value in zip(axis_names, combo):
            _set_dotted(new_dict, axis, value)
            binding[axis] = _format_value_for_name(value)

        new_run = RunConfig.model_validate(new_dict)
        new_run.name = _render_name(name_template, binding, axis_names, combo, base_run.name)
        runs.append(new_run)
    return runs


# ---------------------------------------------------------------------------
# Internals — kept here so yaml_loader can import them as the canonical impl.
# ---------------------------------------------------------------------------


_TEMPLATE_RE = re.compile(r"\{([^{}]+)\}")


def _render_name(
    template: str | None,
    binding: dict[str, str],
    axis_names: list[str],
    combo: tuple[Any, ...],
    base_name: str,
) -> str:
    if template is not None:
        return _render_name_template(template, binding)
    suffix = "_".join(
        f"{axis.split('.')[-1]}{_format_value_for_name(value)}"
        for axis, value in zip(axis_names, combo)
    )
    return f"{base_name}__{suffix}" if suffix else base_name


def _render_name_template(template: str, binding: dict[str, str]) -> str:
    """Substitute ``{dotted.key}`` placeholders without going through str.format.

    ``str.format`` treats dots as attribute access; we just want literal
    dotted keys.
    """

    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in binding:
            raise ValueError(f"name_template references unknown key '{key}'")
        return binding[key]

    return _TEMPLATE_RE.sub(repl, template)


def _set_dotted(obj: dict, path: str, value: Any) -> None:
    """Walk a dotted path into a nested-dict tree and set the leaf value.

    Always called on a ``RunConfig.model_dump()`` result, so every node is a
    dict; intermediate keys missing on params-style dicts are auto-created
    (with ``setdefault``), and a typo in a strict path is caught later by
    ``RunConfig.model_validate``.
    """
    parts = path.split(".")
    for p in parts[:-1]:
        obj = obj.setdefault(p, {})
    obj[parts[-1]] = value


def _format_value_for_name(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}".replace("+", "").replace(".", "p")
    return str(v).replace("/", "_")


__all__ = ["Sweep", "expand_runs"]

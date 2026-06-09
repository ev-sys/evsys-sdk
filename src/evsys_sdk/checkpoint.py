"""Checkpoint — parse the `checkpoints.jsonl` manifests algorithms write.

Tinker SFT/RL (and any other algorithm that follows the same convention)
appends one JSON row per saved checkpoint to ``<output_dir>/checkpoints.jsonl``:

    {"name": "final", "batch": 1520, "epoch": 10,
     "state_path": "tinker://...", "sampler_path": "tinker://..."}

Researcher scripts repeatedly hand-roll a few lines to find this manifest,
parse it, and pick the right row to evaluate against. This module gives them:

  * ``Checkpoint`` — one row, typed.
  * ``read_manifest(path)`` — full ordered list.
  * ``find_manifest(run_dir)`` — locate the manifest under a run directory.
  * ``Checkpoint.pick_final(checkpoints)`` — pick the "evaluate me" row
    (prefer ``name == "final"``, else the last row that exposes a path).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MANIFEST_NAME = "checkpoints.jsonl"


@dataclass(frozen=True)
class Checkpoint:
    """One row of a `checkpoints.jsonl` manifest."""

    label: str
    """The `name` field — e.g. `"final"`, `"epoch-3"`, or a step string."""
    step: int | None = None
    """Training step / batch index, if recorded (``batch`` in tinker)."""
    epoch: int | None = None
    weights_path: str | None = None
    """Training-state checkpoint URI (``state_path`` in tinker)."""
    sampler_path: str | None = None
    """Inference-ready sampler URI; what you pass to a sampling client."""
    raw: dict = field(default_factory=dict)
    """Untouched manifest row, for fields not modeled above."""

    @property
    def has_path(self) -> bool:
        return bool(self.weights_path or self.sampler_path)

    @classmethod
    def from_manifest_row(cls, row: dict) -> Checkpoint:
        return cls(
            label=str(row.get("name", "?")),
            step=_as_int(row.get("batch")),
            epoch=_as_int(row.get("epoch")),
            weights_path=_as_str(row.get("state_path")),
            sampler_path=_as_str(row.get("sampler_path")),
            raw=dict(row),
        )

    @staticmethod
    def pick_final(checkpoints: list["Checkpoint"]) -> Checkpoint | None:
        """Pick the one to evaluate against.

        Strategy: prefer an explicit ``name == "final"`` row that exposes a
        path, else the last row that exposes a path, else None.
        """
        candidates = [c for c in checkpoints if c.has_path]
        if not candidates:
            return None
        for c in candidates:
            if c.label == "final":
                return c
        return candidates[-1]


def read_manifest(path: str | Path) -> list[Checkpoint]:
    """Parse a checkpoints.jsonl into ordered Checkpoint rows.

    Blank lines are skipped; malformed JSON raises ``ValueError`` (don't
    silently lose checkpoint pointers — the eval step depends on them).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint manifest not found: {p}")
    out: list[Checkpoint] = []
    for lineno, line in enumerate(p.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}:{lineno}: malformed jsonl: {e}") from e
        out.append(Checkpoint.from_manifest_row(row))
    return out


def find_manifest(run_dir: str | Path) -> Path | None:
    """Locate `checkpoints.jsonl` under ``run_dir`` (recursive).

    Algorithms sometimes nest the manifest under a sub-directory
    (e.g. ``<run_dir>/<sub>/checkpoints.jsonl``); search shallowest-first
    and return the first match. Returns ``None`` if no manifest exists.
    """
    base = Path(run_dir)
    if not base.is_dir():
        return None
    # Sort by depth so the shallowest match wins.
    matches = sorted(base.rglob(MANIFEST_NAME), key=lambda p: len(p.parts))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _as_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v)
    return s if s else None


__all__ = [
    "Checkpoint",
    "MANIFEST_NAME",
    "find_manifest",
    "read_manifest",
]

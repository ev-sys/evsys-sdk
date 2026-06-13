"""CheckpointManager — write the `checkpoints.jsonl` manifest that
:mod:`evsys_sdk.checkpoint` already knows how to read.

The reader half lives in ``evsys_sdk/checkpoint.py``
(:func:`~evsys_sdk.checkpoint.read_manifest`,
:meth:`~evsys_sdk.checkpoint.Checkpoint.pick_final`). This is the writer
counterpart used by :class:`~evsys_sdk.training.loop.TrainingLoop`.

The format is intentionally identical to the manifest tinker_cookbook used
to emit, so downstream consumers (e.g.
:meth:`~evsys_sdk.inference.tinker.TinkerInference.from_run_result`) keep
working unchanged when an algorithm is ported to the native loop.

Schema per row (one JSON object per line)::

    {"name": "step_500", "batch": 499, "epoch": 3,
     "state_path": "tinker://.../weights/step_500",
     "sampler_path": "tinker://.../sampler_weights/step_500"}

``state_path`` is the full training state (weights + optimizer) used for
resume. ``sampler_path`` is the inference-ready weights snapshot used by
the eval client.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..checkpoint import MANIFEST_NAME, Checkpoint, read_manifest

logger = logging.getLogger(__name__)


@dataclass
class ManifestRow:
    name: str
    batch: int | None = None
    epoch: int | None = None
    state_path: str | None = None
    sampler_path: str | None = None

    def to_json(self) -> str:
        d: dict[str, Any] = {"name": self.name}
        if self.batch is not None:
            d["batch"] = self.batch
        if self.epoch is not None:
            d["epoch"] = self.epoch
        if self.state_path:
            d["state_path"] = self.state_path
        if self.sampler_path:
            d["sampler_path"] = self.sampler_path
        return json.dumps(d, ensure_ascii=False)


class CheckpointManager:
    """Decide WHEN to save and WRITE the manifest row when we do.

    The decision policy is intentionally tiny — `should_save(step)` returns
    True every ``save_every`` steps (after the optimizer step at that index).
    The actual save call is dispatched by the loop, which has the live
    `Backend` handle; the manager only records the resulting paths.

    Final-step save is unconditional: the loop calls ``save_final(...)`` after
    its for-loop completes, so even when ``save_every`` doesn't land exactly
    on the last step the final sampler is always recorded — that's the URI
    downstream eval consumes.
    """

    def __init__(
        self,
        *,
        log_path: Path,
        save_every: int,
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.save_every = max(0, int(save_every))
        self.manifest_path = self.log_path / MANIFEST_NAME
        self._rows: list[ManifestRow] = []

    # --- decision policy ----------------------------------------------------

    def should_save(self, step: int) -> bool:
        """Save after the optimizer step at index ``step`` (zero-based).

        The convention matches tinker_cookbook: ``(step + 1) % save_every == 0``.
        Disabled when ``save_every == 0``.
        """
        if self.save_every <= 0:
            return False
        return (step + 1) % self.save_every == 0

    # --- write surface ------------------------------------------------------

    def record(self, row: ManifestRow) -> None:
        """Append one row to the manifest on disk and remember it."""
        with self.manifest_path.open("a") as f:
            f.write(row.to_json() + "\n")
        self._rows.append(row)

    @property
    def rows(self) -> list[ManifestRow]:
        return list(self._rows)

    # --- resume helpers -----------------------------------------------------

    def find_resume(self) -> Checkpoint | None:
        """Find the most-recent recorded checkpoint to resume training from.

        Reads the existing `checkpoints.jsonl` (if any) under ``log_path`` and
        picks the last row that has a ``state_path``. The loop hands the
        state_path back to the backend to recreate the training client with
        optimizer state intact.

        Returns ``None`` when no resumable checkpoint is on disk — caller
        should start fresh.
        """
        if not self.manifest_path.is_file():
            return None
        try:
            ckpts = read_manifest(self.manifest_path)
        except Exception:
            logger.exception("CheckpointManager.find_resume: failed to parse %s",
                             self.manifest_path)
            return None
        for c in reversed(ckpts):
            if c.weights_path:
                return c
        return None


__all__ = ["CheckpointManager", "ManifestRow"]

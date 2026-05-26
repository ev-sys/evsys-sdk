"""TinkerSFT — supervised fine-tuning via tinker_cookbook.

This is a thin orchestrator: it builds a SupervisedDataset from the rows the
runner already loaded+transformed, then drives tinker_cookbook.supervised.train.

Data contract:
    Each input row must have a `messages` key (list of {role, content})
    suitable for chat-templating. A chat transform (e.g. `jsonl_to_chat`)
    produces this shape.

Checkpoint fractions are converted to absolute step numbers using
tinker_cookbook's save_every; we set save_every to floor(total/lcm) and rely
on the checkpoints folder snapshots.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..protocols import RunContext, RunResult
from ..registry import register_algorithm

# raise ImportError at module load time if tinker_cookbook isn't installed
import chz  # noqa: E402
import tinker  # noqa: E402
from tinker_cookbook.supervised import train as sft_train  # noqa: E402
from tinker_cookbook.supervised.types import (  # noqa: E402
    SupervisedDataset,
    SupervisedDatasetBuilder,
)
from tinker_cookbook.supervised.common import datum_from_model_input_weights  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

# A module-level cache so the chz-frozen builder can fetch its rows by id.
# chz dataclasses are immutable and reject mutable fields like list[dict];
# we keep rows in this dict keyed by builder UUID.
_ROW_CACHE: dict[str, list[dict[str, Any]]] = {}

logger = logging.getLogger(__name__)


class TinkerSFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = 1e-4
    num_epochs: int = 1
    batch_size: int = 4
    max_steps: int | None = None
    lora_rank: int = 8
    max_seq_len: int = 2048
    save_every: int = 0
    """If 0, computed from save_at_fractions."""
    save_at_fractions: list[float] = Field(default_factory=lambda: [1.0])
    """Save checkpoints at these fractions of total steps (most recent wins)."""
    eval_every: int = 0
    wandb_project: str | None = None
    wandb_name: str | None = None
    renderer_name: str | None = None
    """Override model.renderer_name; defaults to it if None."""


def _row_to_datum(row: dict[str, Any], tokenizer, max_seq_len: int):
    """Convert a {messages} row into a tinker Datum, training only on assistant tokens."""
    messages = row.get("messages") or []
    if not messages:
        raise ValueError("row has empty messages")

    # Build the full chat-templated string + identify assistant span(s).
    # We render in two passes to compute the prefix-only token count for masking.
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # If the row has only system+user (rl-style), nothing to learn — skip.
    has_assistant = any(m.get("role") == "assistant" for m in messages)
    if not has_assistant:
        return None

    weights = [0.0] * len(full_ids)

    # Find assistant turns and mark their token spans for training.
    # Strategy: render up to (and not including) each assistant message, diff
    # the lengths to derive the assistant span.
    cursor = 0
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        prefix_messages = messages[:i]
        prefix_text = tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True
        )
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        # Render through this assistant message.
        through_text = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=False, add_generation_prompt=False
        )
        through_ids = tokenizer.encode(through_text, add_special_tokens=False)
        start = max(cursor, len(prefix_ids))
        end = min(len(weights), len(through_ids))
        for j in range(start, end):
            weights[j] = 1.0
        cursor = end

    # Truncate to max_seq_len.
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        weights = weights[:max_seq_len]

    if sum(weights) == 0:
        return None

    import torch

    model_input = tinker.ModelInput.from_ints(full_ids)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    return datum_from_model_input_weights(model_input, weight_tensor, max_length=max_seq_len)


class _InMemorySupervisedDataset(SupervisedDataset):
    """SupervisedDataset implementation backed by an in-memory list of Datums."""

    def __init__(self, data: list[tinker.Datum], batch_size: int) -> None:
        self._data = data
        self.batch_size = batch_size

    def __len__(self) -> int:
        return max(0, len(self._data) // self.batch_size)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self.batch_size
        end = start + self.batch_size
        return self._data[start:end]

    def set_epoch(self, seed: int = 0) -> None:
        rng = __import__("random").Random(seed)
        # In-place deterministic shuffle.
        rng.shuffle(self._data)


@chz.chz
class _RowsBuilder(SupervisedDatasetBuilder):
    """Build a _InMemorySupervisedDataset from rows stashed in _ROW_CACHE.

    chz frozen-dataclass semantics: only primitive fields. The actual rows
    live in `_ROW_CACHE` keyed by `cache_key`.
    """

    cache_key: str
    model_name: str
    max_seq_len: int
    batch_size: int

    def __call__(self):
        rows = _ROW_CACHE[self.cache_key]
        tokenizer = get_tokenizer(self.model_name)
        data = []
        for r in rows:
            datum = _row_to_datum(r, tokenizer, self.max_seq_len)
            if datum is not None:
                data.append(datum)
        if not data:
            raise ValueError("All rows produced empty (no assistant tokens). Check your transform.")
        ds = _InMemorySupervisedDataset(data=data, batch_size=self.batch_size)
        return ds, None


@register_algorithm("tinker_sft")
class TinkerSFT:
    name: ClassVar[str] = "tinker_sft"
    Config: ClassVar[type] = TinkerSFTConfig

    def __init__(self, **kwargs) -> None:
        self.cfg = TinkerSFTConfig.model_validate(kwargs)

    def _resolve_save_every(self, total_steps: int) -> int:
        if self.cfg.save_every:
            return self.cfg.save_every
        # Strategy: save often enough that for each desired fraction there's a
        # checkpoint within `tolerance_steps` of it. We pick save_every so the
        # max snap-to-grid error is <= 5% of total_steps. This avoids the
        # pathological save_every=1 when fractions don't share a clean GCD.
        marks = sorted({max(1, int(round(f * total_steps))) for f in self.cfg.save_at_fractions})
        if not marks:
            return total_steps
        # If marks are exactly evenly spaced, gcd is the right choice and is large.
        gcd = marks[0]
        for m in marks[1:]:
            gcd = math.gcd(gcd, m)
        # If gcd is "too small" (< 5% of total), use total_steps/10 instead and
        # let the consumer pick the closest checkpoint to each desired fraction.
        min_acceptable = max(1, total_steps // 20)
        if gcd >= min_acceptable:
            return gcd
        return max(1, total_steps // 10)

    def train(self, ctx: RunContext) -> RunResult:
        if ctx.backend.name != "tinker":
            raise RuntimeError(
                f"TinkerSFT requires backend=tinker (got '{ctx.backend.name}'). "
                "Use mock_sft or local_sft for other backends."
            )
        rows = ctx.extras.get("train_rows")
        if not rows:
            raise RuntimeError("TinkerSFT.train: ctx.extras['train_rows'] missing/empty")

        handles = ctx.extras.get("backend_handles", {})
        model_name = handles.get("model_name") or ctx.extras.get("model_name")
        if not model_name:
            raise RuntimeError("model_name not set in backend handles")

        tokenizer = get_tokenizer(model_name)
        n = len(rows)
        steps_per_epoch = max(1, n // self.cfg.batch_size)
        total_steps = (
            self.cfg.max_steps
            if self.cfg.max_steps is not None
            else steps_per_epoch * self.cfg.num_epochs
        )
        save_every = self._resolve_save_every(total_steps)

        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        log_path = str(out)

        ctx.log_store.log_hyperparams(
            {
                "algorithm": self.name,
                **self.cfg.model_dump(),
                "model_name": model_name,
                "n_train_rows": n,
                "total_steps": total_steps,
                "save_every": save_every,
            }
        )

        import uuid as _uuid
        cache_key = f"sft_{ctx.run_id}_{_uuid.uuid4().hex}"
        _ROW_CACHE[cache_key] = list(rows)
        builder = _RowsBuilder(
            cache_key=cache_key,
            model_name=model_name,
            max_seq_len=self.cfg.max_seq_len,
            batch_size=self.cfg.batch_size,
        )

        renderer = self.cfg.renderer_name or handles.get("renderer_name")

        config = sft_train.Config(
            log_path=log_path,
            model_name=model_name,
            load_checkpoint_path=handles.get("load_checkpoint_path"),
            renderer_name=renderer,
            dataset_builder=builder,
            learning_rate=self.cfg.learning_rate,
            num_epochs=self.cfg.num_epochs,
            lora_rank=self.cfg.lora_rank,
            save_every=save_every,
            eval_every=self.cfg.eval_every,
            max_steps=self.cfg.max_steps,
            wandb_project=self.cfg.wandb_project,
            wandb_name=self.cfg.wandb_name,
        )
        try:
            asyncio.run(sft_train.main(config))
        except Exception as e:
            logger.exception("TinkerSFT.train failed")
            return RunResult(run_id=ctx.run_id, status="failed", error=str(e))

        # Tinker writes a `checkpoints.jsonl` manifest with rows like:
        #   {"name": "<step_or_'final'>", "batch": N, "epoch": M, "state_path": "tinker://..."}
        artifacts: dict[str, str] = {"run_dir": str(out)}
        ckpt_manifest = out / "checkpoints.jsonl"
        if ckpt_manifest.exists():
            import json as _json
            for line in ckpt_manifest.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                except Exception:
                    continue
                step_label = str(entry.get("name", "?"))
                state_path = entry.get("state_path") or entry.get("sampler_path")
                if not state_path:
                    continue
                key = f"checkpoint-{step_label}"
                artifacts[key] = state_path
        for k, v in artifacts.items():
            ctx.log_store.log_artifact(k, v, kind="checkpoint")

        return RunResult(
            run_id=ctx.run_id,
            status="completed",
            metrics={"total_steps": float(total_steps)},
            artifacts=artifacts,
        )

"""EmbeddingSFT — fine-tune a sentence-transformers bi-encoder on (anchor,
positive) pairs for doc-based tool retrieval.

This is the training side of the doc-based tool-discovery approach. It consumes
rows produced by the ``composio_doc_pairs`` transform — each carrying an
``anchor`` (user query) and a ``positive`` (tool documentation) — and trains a
bi-encoder so that queries land near their correct tool's doc in vector space.

Uses MultipleNegativesRankingLoss: for each (anchor, positive) pair in a batch,
every other positive in the batch acts as an in-batch negative. This is the
standard, label-free contrastive objective for retrieval and needs no explicit
hard negatives.

Requires the optional ``embedding`` extra:
    pip install trajectory-labs[embedding]

Pre-conditions:
  * ctx.extras['train_rows'] contains rows with 'anchor' and 'positive'.
The backend is irrelevant (sentence-transformers manages its own device), so
this runs with backend.kind = 'mock' or 'local'.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from ..protocols import RunContext, RunResult
from ..registry import register_algorithm

logger = logging.getLogger(__name__)


class EmbeddingSFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    """HF sentence-transformers model to start from."""
    num_epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 10
    max_seq_len: int = 256
    anchor_field: str = "anchor"
    positive_field: str = "positive"
    seed: int = 42


@register_algorithm("embedding_sft")
class EmbeddingSFT:
    name: ClassVar[str] = "embedding_sft"
    Config: ClassVar[type] = EmbeddingSFTConfig

    def __init__(self, **kwargs) -> None:
        self.cfg = EmbeddingSFTConfig.model_validate(kwargs)

    def train(self, ctx: RunContext) -> RunResult:
        try:
            from sentence_transformers import (
                SentenceTransformer,
                SentenceTransformerTrainer,
                SentenceTransformerTrainingArguments,
                losses,
            )
            from datasets import Dataset
        except ImportError as e:
            return RunResult(
                run_id=ctx.run_id,
                status="failed",
                error=(
                    "embedding_sft requires the 'embedding' extra: "
                    "pip install trajectory-labs[embedding]"
                ),
            )

        rows = ctx.extras.get("train_rows") or []
        pairs = [
            {
                "anchor": r[self.cfg.anchor_field],
                "positive": r[self.cfg.positive_field],
            }
            for r in rows
            if r.get(self.cfg.anchor_field) and r.get(self.cfg.positive_field)
        ]
        if not pairs:
            return RunResult(
                run_id=ctx.run_id,
                status="failed",
                error="embedding_sft: no rows with both anchor and positive fields",
            )

        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ctx.log_store.log_hyperparams({"algorithm": self.name, **self.cfg.model_dump()})

        model = SentenceTransformer(self.cfg.base_model)
        model.max_seq_length = self.cfg.max_seq_len
        train_ds = Dataset.from_list(pairs)
        loss = losses.MultipleNegativesRankingLoss(model)

        args = SentenceTransformerTrainingArguments(
            output_dir=str(out),
            num_train_epochs=self.cfg.num_epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            learning_rate=self.cfg.learning_rate,
            warmup_steps=self.cfg.warmup_steps,
            seed=self.cfg.seed,
            report_to=[],
            logging_steps=10,
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            loss=loss,
        )

        try:
            trainer.train()
        except Exception as e:
            logger.exception("EmbeddingSFT.train failed")
            return RunResult(run_id=ctx.run_id, status="failed", error=str(e))

        final = out / "final"
        model.save_pretrained(str(final))
        ctx.log_store.log_artifact("final_checkpoint", str(final), kind="checkpoint")

        log_history = getattr(trainer.state, "log_history", [])
        loss_entries = [e for e in log_history if "loss" in e]
        final_loss = float(loss_entries[-1]["loss"]) if loss_entries else 0.0
        for entry in log_history:
            step = int(entry.get("step", 0) or 0)
            metrics = {
                k: float(v)
                for k, v in entry.items()
                if isinstance(v, (int, float)) and k != "step"
            }
            if metrics:
                ctx.log_store.log_metrics(metrics, step=step)

        return RunResult(
            run_id=ctx.run_id,
            status="completed",
            metrics={"train/final_loss": final_loss, "n_pairs": float(len(pairs))},
            artifacts={"final_checkpoint": str(final)},
        )

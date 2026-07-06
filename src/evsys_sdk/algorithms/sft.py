"""SFT — supervised fine-tuning on the SDK training loop.

All the composer
plumbing (backend, step/save cadence, evaluators, loop wiring, artifacts)
lives in :class:`~evsys_sdk.algorithms.base.BaseAlgorithm`. SFT only has to
say *how a batch is built*:

* :meth:`setup` — standardize rows → typed :class:`ChatMessagesRow`, tokenize
  to a static list of :class:`tinker.Datum` with assistant-span loss masks.
* :meth:`build_batch` — slice ``batch_size`` datums per step (wrapping the
  dataset so ``num_steps`` can exceed one epoch).
* :meth:`step_metrics` — ``train_mean_nll`` from the per-position logprobs.

Researchers wanting a one-line tweak (focal loss, custom loss, extra metrics)
subclass ``SFT`` and override ``build_batch`` / ``step_metrics`` — no SDK
change needed.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, cast

from ..data_types import ChatMessagesRow, TargetFormat, parse_rows
from ..protocols import RunContext
from ..registry import register_algorithm
from ..training.batch_utils import coerce_floats, extract_weights
from ..training.loop import TrainingBatch
from ..training.sft_data import sft_tokenize
from ..training.tinker_backend import TinkerBackend
from .base import BaseAlgorithm, BaseAlgorithmConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class SFTConfig(BaseAlgorithmConfig):
    """Config for :class:`SFT`. Inherits the shared training/save/eval knobs
    from :class:`BaseAlgorithmConfig`; adds SFT-only fields."""

    max_seq_len: int = 2048

    # Which assistant turns the loss is computed on. Lives here (algorithm
    # config), NOT on the dataset — ChatMessagesRow carries only the
    # conversation; the algorithm decides what to supervise.
    supervise: Literal["all_assistant", "last_assistant"] = "all_assistant"


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


@register_algorithm("sft")
class SFT(BaseAlgorithm):
    name: ClassVar[str] = "sft"
    Config: ClassVar[type] = SFTConfig

    def _check_inputs(self, ctx: RunContext) -> None:
        if not ctx.extras.get("train_rows"):
            raise RuntimeError("SFT.train: ctx.extras['train_rows'] missing/empty")

    async def setup(self, ctx: RunContext, backend: TinkerBackend) -> None:
        rows = ctx.extras["train_rows"]

        # Standardize raw rows → typed ChatMessagesRow (strict), then tokenize
        # → Datum with assistant-span loss masks. The supervise decision is
        # the algorithm's, not the dataset's.
        chat_rows = cast("list[ChatMessagesRow]", parse_rows(rows, TargetFormat.CHAT_MESSAGES))
        self._datums = sft_tokenize(
            chat_rows, backend.get_tokenizer(),
            max_seq_len=self.cfg.max_seq_len,
            enable_thinking=self.cfg.enable_thinking,
            supervise=self.cfg.supervise,
        )
        if not self._datums:
            raise RuntimeError("SFT.setup: tokenization produced no datums")
        self._n_rows = len(rows)
        self._steps_per_epoch = max(1, self._n_rows // self.cfg.batch_size)

    async def build_batch(self, step_idx: int) -> TrainingBatch:
        """Slice ``batch_size`` Datums for ``step_idx``, wrapping the dataset
        when the slice straddles the end."""
        n = len(self._datums)
        start = (step_idx * self.cfg.batch_size) % n
        end = start + self.cfg.batch_size
        if end <= n:
            data = self._datums[start:end]
        else:
            data = self._datums[start:] + self._datums[: end - n]
        return TrainingBatch(data=data, loss_fn="cross_entropy")

    def step_metrics(
        self, step_idx: int, batch: TrainingBatch, fb_result: Any,
    ) -> dict[str, float]:
        """``train_mean_nll`` from the per-position logprobs of each Datum,
        weighted by the loss mask.

        Tinker's cross_entropy loss returns ``loss_fn_outputs[i]["logprobs"]``:
        a per-position vector of log-probabilities of the target token (a
        "perfect" prediction has logprob 0; otherwise negative). The mean NLL
        is ``-sum(logprob * weight) / sum(weight)`` over the loss-mask
        positions, averaged across the batch."""
        outputs = getattr(fb_result, "loss_fn_outputs", None)
        if not outputs:
            return {}

        total_logprob = 0.0
        total_weight = 0.0
        for datum, out in zip(batch.data, outputs):
            logprobs = coerce_floats(out.get("logprobs") if isinstance(out, dict)
                                     else getattr(out, "logprobs", None))
            if logprobs is None:
                continue
            weights = coerce_floats(extract_weights(datum))
            if weights is None or len(weights) == 0:
                continue
            # Truncate to min length so a token-count mismatch between the
            # per-position logprobs and the per-position mask doesn't blow up.
            k = min(len(logprobs), len(weights))
            for j in range(k):
                total_logprob += logprobs[j] * weights[j]
                total_weight += weights[j]

        if total_weight <= 0:
            return {}
        return {"train_mean_nll": -float(total_logprob) / float(total_weight)}

    def _hyperparams_extra(self) -> dict[str, Any]:
        return {"n_train_rows": self._n_rows, "n_train_datums": len(self._datums)}


__all__ = ["SFT", "SFTConfig"]

"""ValidationEvaluator — in-loop harbor validation scored with the metrics.py registry.

tinker_cookbook's training loops (SFT + RL) run any ``SamplingClientEvaluator``
every ``eval_every`` steps, handing it a fresh ``tinker.SamplingClient``. This
evaluator bridges *our* world into that hook: it generates on the harbor
validation tasks and scores the completions with the project's registered
metrics (``metrics.py`` via ``get_metric``) — the same predict→metric shape
``runner._run_eval`` uses post-training, but now mid-run.

Shared by ``tinker_sft`` and ``tinker_rl`` (Option A: one definition, both
import it). Both loops build evaluators once
(``[b() for b in evaluator_builders]``), so a single instance is reused across
every eval — we keep an internal call counter to log each eval at the right
training step (call ``k`` → step ``k * eval_for_every``).

Recording: validation metrics are keyed ``val/<metric>`` and written to the
SDK's ``log_store`` (``run_dir/logs/metrics.jsonl`` in the nested
``{step, metrics}`` shape). ``Experiment.forward_step_metrics`` later forwards
them to the dashboard tagged ``split="val"``. (tinker_cookbook's own
``metrics.jsonl`` is flat and at a different path, so we don't rely on it.)
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from ..config import MetricSpec
from ..data_types import HarborTask
from ..registry import get_metric

# tinker_cookbook is optional (only the in-loop evaluator needs it). Guarding the
# import keeps ``compute_validation_metrics`` — the pure, unit-testable scoring
# core — importable in environments without tinker installed.
try:
    from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
except ImportError:  # pragma: no cover - exercised only where tinker is absent
    SamplingClientEvaluator = object  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"<answer>\s*([\w]+)\s*</answer>")


def _extract_answer(text: str) -> str:
    """Pull ``<answer>X</answer>`` if present, else the stripped raw text.

    Mirrors ``runner._run_eval`` so the same metrics.py metrics score
    validation the same way they score the post-training eval.
    """
    m = _ANSWER_RE.search(text or "")
    return m.group(1) if m else (text or "").strip()


def _target_from_task(task: HarborTask) -> dict[str, Any]:
    """Gold target for a harbor task, in the {answer, toolkit} shape metrics expect."""
    expected = getattr(task.verifier, "expected", None)
    md = task.metadata or {}
    return {
        "answer": expected if expected is not None else md.get("answer", ""),
        "toolkit": md.get("toolkit", ""),
    }


def compute_validation_metrics(
    tasks: list[HarborTask],
    completions: list[str],
    metric_specs: list[MetricSpec],
) -> dict[str, float]:
    """Score raw completions against harbor tasks via the metrics.py registry.

    Pure (no I/O) so it's unit-testable without a sampling client. Returns
    ``{"val/<metric_kind>": value}``; a metric that raises is logged and skipped.
    """
    predictions = [{"answer": _extract_answer(c), "raw": c} for c in completions]
    targets = [_target_from_task(t) for t in tasks]
    out: dict[str, float] = {}
    for ms in metric_specs:
        try:
            cls = get_metric(ms.kind)
            inst = cls(**(ms.params or {}))
            out[f"val/{ms.kind}"] = float(inst.compute(predictions=predictions, targets=targets))
        except Exception:
            logger.exception("validation metric %s failed", ms.kind)
    return out


class ValidationEvaluator(SamplingClientEvaluator):
    """Score a harbor validation set with metrics.py, in-loop during training."""

    def __init__(
        self,
        *,
        tasks: list[HarborTask],
        metric_specs: list[MetricSpec],
        tokenizer: Any,
        eval_for_every: int,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        log_store: Any | None = None,
    ) -> None:
        self.tasks = list(tasks)
        self.metric_specs = list(metric_specs)
        self.tokenizer = tokenizer
        self.eval_for_every = int(eval_for_every)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.stop = list(stop or [])
        self.log_store = log_store
        self._calls = 0  # eval ordinal → training step via eval_for_every

    async def __call__(self, sampling_client: Any) -> dict[str, float]:
        import tinker

        params = tinker.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=list(self.stop),
        )

        async def _gen(task: HarborTask) -> str:
            ids = self.tokenizer.encode(task.instruction)
            model_input = tinker.ModelInput.from_ints(ids)
            try:
                r = await sampling_client.sample_async(
                    prompt=model_input, num_samples=1, sampling_params=params
                )
                tokens = r.sequences[0].tokens
            except Exception:
                logger.exception("validation generate failed for task %s", task.task_id)
                return ""
            return self.tokenizer.decode(list(tokens)) if tokens else ""

        completions = await asyncio.gather(*[_gen(t) for t in self.tasks])
        metrics = compute_validation_metrics(self.tasks, list(completions), self.metric_specs)

        step = self._calls * self.eval_for_every
        self._calls += 1
        if metrics and self.log_store is not None:
            try:
                self.log_store.log_metrics(metrics, step=step)
            except Exception:
                logger.exception("validation log_metrics failed at step %s", step)
        return metrics


def build_validation_evaluator_builders(ctx: Any, tokenizer: Any) -> tuple[list, int | None]:
    """Translate ``ctx.extras['validation']`` into tinker_cookbook config inputs.

    Returns ``(evaluator_builders, eval_every)`` to pass straight into
    ``sft_train.Config`` / ``rl_train.Config``. When no validation is
    configured, returns ``([], None)`` so the caller falls back to its own
    ``cfg.eval_every``. Shared by ``tinker_sft`` and ``tinker_rl`` (Option A).
    """
    val = (getattr(ctx, "extras", None) or {}).get("validation")
    if not val:
        return [], None

    def _build() -> ValidationEvaluator:
        gen = val.get("gen") or {}
        return ValidationEvaluator(
            tasks=val["tasks"],
            metric_specs=val["metric_specs"],
            tokenizer=tokenizer,
            eval_for_every=val["eval_for_every"],
            max_tokens=int(gen.get("max_tokens", 256)),
            temperature=float(gen.get("temperature", 0.0)),
            log_store=getattr(ctx, "log_store", None),
        )

    return [_build], int(val["eval_for_every"])


__all__ = [
    "ValidationEvaluator",
    "compute_validation_metrics",
    "build_validation_evaluator_builders",
]

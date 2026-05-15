"""Built-in metrics: exact_match, mean_reward, pass@k, toolkit_match."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import register_metric


class _NoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


@register_metric("exact_match")
class ExactMatch:
    """Exact-string match between prediction['answer'] and target['answer']."""

    name: ClassVar[str] = "exact_match"
    Config: ClassVar[type] = _NoConfig

    def compute(
        self,
        *,
        predictions: list[dict[str, Any]],
        targets: list[dict[str, Any]],
    ) -> float:
        if not predictions:
            return 0.0
        if len(predictions) != len(targets):
            raise ValueError("predictions and targets must have the same length")
        n_correct = sum(
            1 for p, t in zip(predictions, targets) if p.get("answer") == t.get("answer")
        )
        return n_correct / len(predictions)


@register_metric("toolkit_match")
class ToolkitMatch:
    """Predicted answer's toolkit prefix matches target's toolkit."""

    name: ClassVar[str] = "toolkit_match"
    Config: ClassVar[type] = _NoConfig

    def compute(
        self,
        *,
        predictions: list[dict[str, Any]],
        targets: list[dict[str, Any]],
    ) -> float:
        if not predictions:
            return 0.0
        n = 0
        for p, t in zip(predictions, targets):
            ans = p.get("answer", "") or ""
            tk = t.get("toolkit", "") or ""
            if tk and ans.startswith(tk + "_"):
                n += 1
        return n / len(predictions)


@register_metric("mean_reward")
class MeanReward:
    """Mean of prediction['reward']."""

    name: ClassVar[str] = "mean_reward"
    Config: ClassVar[type] = _NoConfig

    def compute(
        self,
        *,
        predictions: list[dict[str, Any]],
        targets: list[dict[str, Any]],
    ) -> float:
        if not predictions:
            return 0.0
        return sum(float(p.get("reward", 0.0)) for p in predictions) / len(predictions)


class PassAtKConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    k: int = 1


@register_metric("pass_at_k")
class PassAtK:
    """Pass@k: prediction['samples'] is a list[str]; target['answer'] must appear in first k."""

    name: ClassVar[str] = "pass_at_k"
    Config: ClassVar[type] = PassAtKConfig

    def __init__(self, *, k: int = 1) -> None:
        self.k = k

    def compute(
        self,
        *,
        predictions: list[dict[str, Any]],
        targets: list[dict[str, Any]],
    ) -> float:
        if not predictions:
            return 0.0
        n = 0
        for p, t in zip(predictions, targets):
            samples = p.get("samples") or [p.get("answer")]
            if t.get("answer") in samples[: self.k]:
                n += 1
        return n / len(predictions)

"""Built-in deterministic trigger fns.

Each is a class carrying ``name`` + a Pydantic ``Config`` (``extra="forbid"``) and
an ``evaluate(state) -> TriggerDecision`` method — cheap, pure-Python, no LLM.
They read the accumulated :class:`~evsys_sdk.triggers.state.TriggerState` (a
rolling window + aggregates + counters) and decide whether the batch is worth
escalating to the trigger agent. A researcher registers their own the same way::

    @register_trigger("my_gate")
    class MyGate:
        name = "my_gate"
        class Config(BaseModel):
            model_config = ConfigDict(extra="forbid")
            ...
        def evaluate(self, state): ...
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from ..protocols import TriggerDecision
from ..registry import register_trigger
from .state import TriggerState


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


@register_trigger("failure_rate")
class FailureRate:
    """Escalate when a large fraction of the recent window failed — either a hard
    ``status == "error"`` or a reward below ``reward_below``."""

    name = "failure_rate"

    class Config(_Params):
        threshold: float = 0.4
        """Escalate once the failing fraction reaches this."""
        reward_below: float = 0.5
        """A reward strictly under this counts as a failure."""
        min_traces: int = 10
        """Don't judge until the window holds at least this many traces."""

    def __init__(self, **params: Any) -> None:
        self.cfg = self.Config(**params)

    def evaluate(self, state: TriggerState) -> TriggerDecision:
        w = state.window
        if len(w) < self.cfg.min_traces:
            return TriggerDecision(False, f"only {len(w)} traces (< {self.cfg.min_traces})")
        bad = [
            s for s in w
            if s.get("status") == "error"
            or (isinstance(s.get("reward"), (int, float)) and s["reward"] < self.cfg.reward_below)
        ]
        rate = len(bad) / len(w)
        signal = {"failure_rate": rate, "window": len(w), "failed": len(bad)}
        if rate >= self.cfg.threshold:
            return TriggerDecision(
                True,
                f"failure rate {rate:.0%} ≥ {self.cfg.threshold:.0%} over {len(w)} traces",
                signal,
                [s["trace_id"] for s in bad],
            )
        return TriggerDecision(False, f"failure rate {rate:.0%} < {self.cfg.threshold:.0%}", signal)


@register_trigger("feedback_drop")
class FeedbackDrop:
    """Escalate when the average reward over the window falls below ``floor``."""

    name = "feedback_drop"

    class Config(_Params):
        floor: float = 0.5
        min_traces: int = 10

    def __init__(self, **params: Any) -> None:
        self.cfg = self.Config(**params)

    def evaluate(self, state: TriggerState) -> TriggerDecision:
        w = state.window
        rewards = [s["reward"] for s in w if isinstance(s.get("reward"), (int, float))]
        if len(rewards) < self.cfg.min_traces:
            return TriggerDecision(False, f"only {len(rewards)} scored traces (< {self.cfg.min_traces})")
        avg = sum(rewards) / len(rewards)
        signal = {"avg_reward": avg, "scored": len(rewards)}
        if avg < self.cfg.floor:
            return TriggerDecision(
                True,
                f"avg reward {avg:.2f} < floor {self.cfg.floor:.2f} over {len(rewards)} traces",
                signal,
                [s["trace_id"] for s in w],
            )
        return TriggerDecision(False, f"avg reward {avg:.2f} ≥ floor {self.cfg.floor:.2f}", signal)


@register_trigger("volume")
class Volume:
    """Escalate purely on throughput — every ``every`` traces since the last
    escalation, regardless of quality (a periodic 'take a look' gate)."""

    name = "volume"

    class Config(_Params):
        every: int = 50

    def __init__(self, **params: Any) -> None:
        self.cfg = self.Config(**params)

    def evaluate(self, state: TriggerState) -> TriggerDecision:
        n = int(state.counters.get("since_last_escalation", 0))
        signal = {"since_last_escalation": n, "every": self.cfg.every}
        if n >= self.cfg.every:
            return TriggerDecision(
                True,
                f"{n} new traces since last escalation (≥ {self.cfg.every})",
                signal,
                [s["trace_id"] for s in state.window],
            )
        return TriggerDecision(False, f"{n} new traces (< {self.cfg.every})", signal)


@register_trigger("novelty")
class Novelty:
    """Escalate on input drift — a high fraction of *distinct* input signatures in
    the window (the agent is fielding many new kinds of task)."""

    name = "novelty"

    class Config(_Params):
        threshold: float = 0.8
        min_traces: int = 10

    def __init__(self, **params: Any) -> None:
        self.cfg = self.Config(**params)

    def evaluate(self, state: TriggerState) -> TriggerDecision:
        sigs = [s.get("input_sig") for s in state.window if s.get("input_sig")]
        if len(sigs) < self.cfg.min_traces:
            return TriggerDecision(False, f"only {len(sigs)} traces with input (< {self.cfg.min_traces})")
        frac = len(set(sigs)) / len(sigs)
        signal = {"distinct_fraction": frac, "n": len(sigs)}
        if frac >= self.cfg.threshold:
            return TriggerDecision(
                True,
                f"{frac:.0%} distinct inputs (≥ {self.cfg.threshold:.0%}) — input drift",
                signal,
                [s["trace_id"] for s in state.window],
            )
        return TriggerDecision(False, f"{frac:.0%} distinct inputs (< {self.cfg.threshold:.0%})", signal)


__all__ = ["FailureRate", "FeedbackDrop", "Novelty", "Volume"]

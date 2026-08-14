"""The function — a deterministic function the system runs, as an extension point.

The SDK already runs two families of these, each behind its own registry:

  * **trigger gate fns** (``@register_trigger`` — ``evaluate(state) ->
    TriggerDecision``): the cheap, always-on gate over ingested traces;
  * **verifier fns** (``verifiers.fns`` — ``(model_output, expected, params)
    -> float``): the scoring functions eval tasks reference by name.

Both are the same *concept*: pure params in, a value out, no LLM in the loop
— the deterministic counterpart of :class:`~evsys_sdk.agents.base.EvsysAgent`.
This module gives that concept one name (:class:`EvsysFunction`), one registry
(``@register_function``), and ADAPTERS over the existing registries. It is a
base class + adapters, deliberately not a rewrite: the ``trigger`` and
``verifier_fn`` registries keep working untouched, and nothing re-registers
their entries under new names (a project's gate registered as ``my_gate``
stays ``my_gate`` everywhere).

Like agents, a function declares WHERE it runs via an ``environment`` field.
Every existing function is in-process, so the default is ``"local"`` — the
field exists so a future sandboxed verifier (the ``E2BVerifier`` payload
shape already implies one) has a first-class place to say so.

Register a new function like any other extension::

    from evsys_sdk import EvsysFunction, register_function

    @register_function("latency_budget")
    class LatencyBudget(EvsysFunction):
        class Config(EvsysFunction.Config):
            max_ms: float = 500.0

        def run(self, trace) -> bool:
            return trace["latency_ms"] <= self.cfg.max_ms
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from ..registry import get_trigger, register_function


class EvsysFunction:
    """One deterministic function: params + a ``run`` + where it executes.

    Construct with keyword params (validated against this class's ``Config``)
    or through the registry with :func:`~evsys_sdk.functions.build_function`.
    """

    name: ClassVar[str] = ""

    #: Where the function executes. ``"local"`` = in this process — true of
    #: every built-in today. A subclass that dispatches elsewhere overrides
    #: this (or passes ``environment=`` at construction).
    environment: ClassVar[str] = "local"

    class Config(BaseModel):
        model_config = {"extra": "forbid"}

    def __init__(self, *, environment: str | None = None, **params: Any) -> None:
        # Validate params against the subclass's Config (loud on a typo).
        self.cfg = self.Config(**params)
        if environment is not None:
            self.environment = environment

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the function. Subclasses define their own argument shape
        (a gate fn takes trigger state; a verifier fn takes output/expected)."""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)


# ---------------------------------------------------------------------------
# Adapters over the two existing function families
# ---------------------------------------------------------------------------


@register_function("trigger_fn")
class TriggerFunction(EvsysFunction):
    """A registered ``@register_trigger`` gate fn, as an :class:`EvsysFunction`.

    ``run(state)`` delegates to the wrapped fn's ``evaluate(state)`` — the
    Trigger protocol is untouched; this only gives gate fns the common shape.
    """

    class Config(BaseModel):
        model_config = {"extra": "forbid"}

        kind: str
        """Registry key of the wrapped ``@register_trigger`` fn."""
        params: dict[str, Any] = {}
        """Passed through to the wrapped fn's constructor (its own Config)."""

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._fn = get_trigger(self.cfg.kind)(**(self.cfg.params or {}))

    def run(self, state: Any) -> Any:  # -> TriggerDecision
        return self._fn.evaluate(state)


@register_function("verifier_fn")
class VerifierFunction(EvsysFunction):
    """A named in-process verifier fn (``verifiers.fns``), as an
    :class:`EvsysFunction`.

    ``run(model_output, expected, params)`` keeps the exact ``VerifierFn``
    call shape; per-call ``params`` default to the ones given at construction.
    """

    class Config(BaseModel):
        model_config = {"extra": "forbid"}

        fn_name: str
        """Name in the ``verifiers.fns`` registry (e.g. ``exact_match``)."""
        params: dict[str, Any] = {}
        """Default fn params, overridable per call."""

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        from ..verifiers import fns

        self._fn = fns.get(self.cfg.fn_name)

    def run(self, model_output: str, expected: Any = None,
            params: dict | None = None) -> float:
        return self._fn(model_output, expected,
                        params if params is not None else dict(self.cfg.params))


__all__ = ["EvsysFunction", "TriggerFunction", "VerifierFunction"]

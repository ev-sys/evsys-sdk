"""Factory + hook resolution for the trigger gate.

``build_trigger`` materializes a live :class:`~evsys_sdk.protocols.Trigger` from a
:class:`~evsys_sdk.triggers.state.TriggerPolicy` (mirrors
``trace_sources.runtime.build_trace_sources``); ``resolve_hook`` turns the
``trigger:`` section of a ``SystemConfig`` into the
:data:`~evsys_sdk.trace_sources.base.TraceHook` that Layer-1 ingestion fires per
trace. With no ``trigger:`` configured it returns ``None`` — the Layer-1 no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..registry import get_trigger
from .state import LocalTriggerStore, TriggerPolicy

if TYPE_CHECKING:
    from ..protocols import Trigger
    from ..trace_sources.base import TraceHook


def build_trigger(policy: TriggerPolicy) -> Trigger:
    """Resolve ``policy.kind`` in the trigger registry and construct the fn,
    validating ``policy.params`` against its ``Config`` (a typo fails loudly)."""
    cls = get_trigger(policy.kind)
    return cls(**(policy.params or {}))


def resolve_hook(trigger_cfg: Any, *, store: LocalTriggerStore | None = None) -> TraceHook | None:
    """Build the deterministic gate as a per-trace hook.

    ``trigger_cfg`` is the ``TriggerConfig`` (the policy *seed*). It seeds
    ``policy.json`` on first run but never clobbers an existing (agent-edited)
    one; thereafter the persisted policy is authoritative and read live. Returns
    ``None`` when ``trigger_cfg`` is ``None`` so ingestion falls back to the
    Layer-1 no-op hook.
    """
    if trigger_cfg is None:
        return None
    # Imported here to keep the module import-cycle-free (driver → runtime).
    from .driver import TriggerDriver

    state_dir = getattr(trigger_cfg, "state_dir", None) or ".evsys/triggers"
    st = store or LocalTriggerStore(state_dir)
    seed = TriggerPolicy(
        kind=getattr(trigger_cfg, "kind", ""),
        params=dict(getattr(trigger_cfg, "params", None) or {}),
        every_n=int(getattr(trigger_cfg, "every_n", 20)),
        window=int(getattr(trigger_cfg, "window", 100)),
        signals=list(getattr(trigger_cfg, "signals", None) or TriggerPolicy().signals),
    )
    return TriggerDriver(st, seed_policy=seed, agent_cfg=getattr(trigger_cfg, "agent", None))


__all__ = ["build_trigger", "resolve_hook"]

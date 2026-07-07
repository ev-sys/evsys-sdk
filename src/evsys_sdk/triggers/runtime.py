"""Factory + hook resolution for the trigger gate.

``build_trigger`` materializes a live :class:`~evsys_sdk.protocols.Trigger` from a
:class:`~evsys_sdk.triggers.state.TriggerPolicy` (mirrors
``trace_sources.runtime.build_trace_sources``); ``resolve_hook`` turns the
``trigger:`` section of a ``SystemConfig`` into the
:data:`~evsys_sdk.trace_sources.base.TraceHook` that Layer-1 ingestion fires per
trace. With no ``trigger:`` configured it returns ``None`` — the Layer-1 no-op.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
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


def import_trigger_code(import_path: str) -> None:
    """Import the researcher's trigger module for its ``@register_trigger``
    side effect — a ``.py`` file path or a dotted module name.

    Registration happens at import time, so the daemon process must import the
    fn's module before the gate resolves ``kind``; this is how ``system.yaml``
    alone (``trigger.import_path``) gets user code loaded. Fails loudly at
    startup — a broken path should stop the daemon, not surface as a per-cycle
    error event.
    """
    p = Path(import_path)
    if p.suffix == ".py":
        if not p.exists():
            raise FileNotFoundError(f"trigger.import_path file not found: {import_path}")
        name = f"_evsys_trigger_{p.stem}"
        if name in sys.modules:  # already imported (e.g. watch restarts the hook)
            return
        spec = importlib.util.spec_from_file_location(name, p)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    else:
        importlib.import_module(import_path)


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

    # Load the researcher's @register_trigger code first — without this the
    # daemon process has no way to know the fn `kind` names (CLI included).
    import_path = getattr(trigger_cfg, "import_path", None)
    if import_path:
        import_trigger_code(import_path)

    state_dir = getattr(trigger_cfg, "state_dir", None) or ".evsys/triggers"
    st = store or LocalTriggerStore(state_dir)
    seed = TriggerPolicy(
        kind=getattr(trigger_cfg, "kind", ""),
        params=dict(getattr(trigger_cfg, "params", None) or {}),
        every_n=int(getattr(trigger_cfg, "every_n", 20)),
        window=int(getattr(trigger_cfg, "window", 100)),
    )
    return TriggerDriver(st, seed_policy=seed)


__all__ = ["build_trigger", "import_trigger_code", "resolve_hook"]

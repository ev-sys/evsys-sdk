"""Sandbox providers — *where* an agent runs, as a pluggable extension point.

The trigger + autoresearch agents don't care whether they execute in an E2B
microVM, a scratch dir on this host, or a provider a user wrote this morning:
they need files in, a command run, and files out. :class:`BaseSandbox` is that
contract; providers implement five methods and register a name.

Importing this package registers the built-ins (side-effect imports below), so
``@register_sandbox`` fires on ``import evsys_sdk``. Vendor SDKs stay lazy —
``e2b`` is imported only when an E2B sandbox actually starts.
"""

from __future__ import annotations

# Side-effect imports: register the built-in @register_sandbox providers.
from . import e2b as _e2b  # noqa: F401
from . import local as _local  # noqa: F401
from . import modal as _modal  # noqa: F401
from .base import DEFAULT_WORKDIR, BaseSandbox, SandboxSetupError
from .runtime import available_sandboxes, build_sandbox, resolve_envs

__all__ = [
    "DEFAULT_WORKDIR",
    "BaseSandbox",
    "SandboxSetupError",
    "available_sandboxes",
    "build_sandbox",
    "resolve_envs",
]

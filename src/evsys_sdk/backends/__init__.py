"""Built-in backends. Importing this module registers them."""

from .mock import MockBackend  # noqa: F401

# Tinker is optional; load only if installed.
try:
    from .tinker import TinkerBackend  # noqa: F401

    # SkyRL serves the same protocol from your own hardware; it uses the very
    # same client, so it lives or dies with the tinker package too.
    from .skyrl import SkyRLBackend  # noqa: F401
except ImportError:
    pass

# Local TRL is optional too.
try:
    from .local import LocalBackend  # noqa: F401
except ImportError:
    pass

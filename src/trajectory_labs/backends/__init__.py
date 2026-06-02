"""Built-in backends. Importing this module registers them."""

from .mock import MockBackend  # noqa: F401
from .modal import ModalBackend  # noqa: F401  (modal imported lazily at dispatch)

# Tinker is optional; load only if installed.
try:
    from .tinker import TinkerBackend  # noqa: F401
except ImportError:
    pass

# Local TRL is optional too.
try:
    from .local import LocalBackend  # noqa: F401
except ImportError:
    pass

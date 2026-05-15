"""Built-in inference clients."""

from .mock import MockInference  # noqa: F401

try:
    from .tinker import TinkerInference  # noqa: F401
except ImportError:
    pass

try:
    from .local import LocalInference  # noqa: F401
except ImportError:
    pass

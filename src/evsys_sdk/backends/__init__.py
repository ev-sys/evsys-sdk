"""Built-in backends. Importing this module registers them."""

from .mock import MockBackend  # noqa: F401

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

# Fireworks registers unconditionally — the module imports fireworks-ai lazily
# (inside prepare()), so the "fireworks" kind is always selectable; the dep is
# only required at run time.
from .fireworks import FireworksBackend  # noqa: F401

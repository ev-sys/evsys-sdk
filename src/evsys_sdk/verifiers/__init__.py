"""Built-in verifiers.

Two layers:
  * ``@register_verifier`` *classes* (runtime ``Verifier`` Protocol).
  * in-process verifier *functions* in ``fns`` — the SDK-local registry that
    ``InProcessVerifier(fn_name=…)`` resolves against (D10: SDK is the single
    source of truth; the remote stores only ``fn_name`` + ``expected``/``params``).
"""

from . import fns  # noqa: F401  in-process verifier-fn registry (D10)
from .fns import get as get_verifier_fn  # noqa: F401
from .fns import list_fns as list_verifier_fns  # noqa: F401
from .fns import register as register_verifier_fn  # noqa: F401
from .format_only import FormatOnlyVerifier  # noqa: F401

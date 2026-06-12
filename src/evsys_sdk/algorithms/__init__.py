"""Built-in algorithms.

Each algorithm targets a specific (recipe-kind, backend-kind) pair via the
registry: the algorithm's `name` is what users put in `algorithm.kind:` of
the YAML; routing to a specific backend happens inside the algorithm's
.train() (it inspects ctx.backend.name).

Some algorithms are backend-agnostic (Mock); most route to a single backend.
"""

from .mock_sft import MockSFT  # noqa: F401
from .mock_rl import MockRL  # noqa: F401
from .combo import ComboAlgorithm  # noqa: F401
from .gepa_prompt import GEPAPromptAlgorithm  # noqa: F401

# Tinker recipes — optional.
try:
    from .native_sft import NativeSFT  # noqa: F401  — native loop, no cookbook
except ImportError:
    pass

try:
    from .tinker_sft import TinkerSFT  # noqa: F401  — deprecated, removed after one release
except ImportError:
    pass

try:
    from .tinker_rl import TinkerRL  # noqa: F401
except ImportError:
    pass

try:
    from .native_sdft import NativeSDFT  # noqa: F401  — native loop
except ImportError:
    pass

try:
    from .tinker_sdft import TinkerSDFT  # noqa: F401  — deprecated
except ImportError:
    pass

# Local TRL — optional.
try:
    from .local_sft import LocalSFT  # noqa: F401
except ImportError:
    pass

try:
    from .local_rl import LocalRL  # noqa: F401
except ImportError:
    pass

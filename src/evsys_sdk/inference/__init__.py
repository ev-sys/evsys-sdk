"""Built-in inference clients."""

from .chat_templated import ChatTemplatedInference  # noqa: F401
from .mock import MockInference  # noqa: F401

try:
    from .tinker import TinkerInference  # noqa: F401
except ImportError:
    pass

try:
    from .local import LocalInference  # noqa: F401
except ImportError:
    pass

# Frontier-API clients — optional. Each constructor raises ImportError if the
# vendor SDK isn't installed; the module imports as a no-op in that case.
try:
    from .claude import ClaudeInference  # noqa: F401
except ImportError:
    pass

try:
    from .gemini import GeminiInference  # noqa: F401
except ImportError:
    pass

try:
    from .openai import OpenAIInference  # noqa: F401
except ImportError:
    pass

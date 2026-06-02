"""Built-in inference clients."""

from .mock import MockInference  # noqa: F401

# Embedding retrieval — hashing backend is numpy-only (always available);
# the sentence_transformers backend is gated at construction time.
try:
    from .embedding import EmbeddingRetrieval  # noqa: F401
except ImportError:
    pass

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

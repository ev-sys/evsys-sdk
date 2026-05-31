"""Built-in transforms. Importing this module registers them.

Built-ins are deliberately generic (`identity`, `jsonl_to_chat`). Domain-specific
transforms are written in your own project via ``@register_transform`` — see the
SDK reference's "writing extensions" section."""

from .identity import IdentityTransform  # noqa: F401
from .jsonl_to_chat import JSONLToChatTransform  # noqa: F401
from .composio import ComposioSFTNoToolsTransform, ComposioDocPairsTransform  # noqa: F401

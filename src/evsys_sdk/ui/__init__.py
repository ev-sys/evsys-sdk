"""Local observability UI for the continual-learning loop (``evsys ui``)."""

from .server import collect_state, serve

__all__ = ["collect_state", "serve"]

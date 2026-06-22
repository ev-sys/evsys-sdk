"""A no-op :class:`~evsys_sdk.protocols.LogStore`.

Used on the training-loop path so per-step / eval metrics flow *only* through
the callbacks (``on_step_end`` / ``on_eval`` → ``local_logger``), making
``local_logger`` the single local writer with no duplicate ``metrics.jsonl``.
Not registered — it is an internal plumbing detail, not a user-selectable kind.
"""

from __future__ import annotations

from typing import Any, ClassVar


class NullLogStore:
    """Swallows every call. Satisfies the ``LogStore`` protocol."""

    name: ClassVar[str] = "null"

    def log_scalar(self, key: str, value: float, step: int) -> None:
        pass

    def log_metrics(
        self, metrics: dict[str, float], step: int, *, split: str = "train"
    ) -> None:
        pass

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        pass

    def log_artifact(self, name: str, path: str, *, kind: str = "file") -> None:
        pass

    def close(self) -> None:
        pass

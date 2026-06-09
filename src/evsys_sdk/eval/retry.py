"""Retry decorator + report for connection-style transient errors.

Wraps a callable; retries on a configurable set of exceptions with exponential
backoff. Exhausted failures are pushed onto a RetryReport so the eval harness
can surface them in the final report rather than aborting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# Default transient errors: anything that looks like a connection / timeout /
# transient HTTP error. Imported lazily so the SDK doesn't hard-depend on httpx.
# (Callers can pass extra exception types via `retriable_exceptions`.)
def _default_retriable_exceptions() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [ConnectionError, TimeoutError]
    try:
        import httpx

        types += [
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.LocalProtocolError,
        ]
    except ImportError:
        pass
    return tuple(types)


@dataclass
class RetryFailure:
    """One exhausted-retry event."""

    context: str
    """Free-form label for the call site (e.g. 'eval:task42:sample0')."""
    exception_type: str
    exception_message: str
    attempts: int


@dataclass
class RetryReport:
    """Collects retry-exhausted failures during an eval pass."""

    failures: list[RetryFailure] = field(default_factory=list)

    def record(self, context: str, exc: BaseException, attempts: int) -> None:
        self.failures.append(
            RetryFailure(
                context=context,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                attempts=attempts,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_failures": len(self.failures),
            "by_exception_type": self._by_type(),
            "failures": [
                {
                    "context": f.context,
                    "exception_type": f.exception_type,
                    "exception_message": f.exception_message,
                    "attempts": f.attempts,
                }
                for f in self.failures
            ],
        }

    def _by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.failures:
            counts[f.exception_type] = counts.get(f.exception_type, 0) + 1
        return counts


def call_with_retry(
    fn: Callable[..., T],
    *args: Any,
    max_attempts: int = 5,
    backoff_base: float = 1.0,
    backoff_cap: float = 16.0,
    retriable: tuple[type[BaseException], ...] | None = None,
    report: RetryReport | None = None,
    context: str = "",
    **kwargs: Any,
) -> T | None:
    """Call ``fn(*args, **kwargs)`` retrying on transient errors.

    On exhaustion: records to ``report`` (if supplied) and returns ``None``.
    Non-transient exceptions propagate immediately — eval bugs shouldn't be
    silently retried.
    """
    if retriable is None:
        retriable = _default_retriable_exceptions()

    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except retriable as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
            time.sleep(delay)
    if report is not None and last_exc is not None:
        report.record(context, last_exc, max_attempts)
    return None

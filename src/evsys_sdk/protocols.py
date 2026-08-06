"""Protocol contracts for every extension point.

We use ``typing.Protocol`` (PEP 544) instead of ABCs so extensions don't have
to subclass anything from the library — any class that implements the methods
satisfies the protocol. This is what makes third-party algorithms / verifiers
/ metrics trivial to add.

A few protocols are runtime-checkable so the registry can give better error
messages, but most are duck-typed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Run context + result (passed to every algorithm.train call)
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    """Everything an algorithm needs to execute one training run.

    Held loosely on purpose: an algorithm only consumes the fields it needs.
    Backends construct this from the parsed ExperimentConfig.
    """

    run_id: str
    """Stable id for this run; used for output paths and log keys."""
    output_dir: str
    """Local filesystem dir where the run can write checkpoints, logs, etc."""
    config: Any
    """The full parsed ExperimentConfig (kept generic to avoid circular import)."""
    data_store: DataStore
    """Datastore handle (read inputs, write outputs)."""
    backend: Backend
    """Backend handle (Tinker / Local / Mock)."""
    extras: dict[str, Any] = field(default_factory=dict)
    """Free-form bag for backend-specific bits (e.g. tinker training_client)."""


@dataclass
class RunResult:
    """What an algorithm returns from .train(ctx)."""

    run_id: str
    status: str
    """One of: 'completed', 'failed', 'cancelled'."""
    metrics: dict[str, float] = field(default_factory=dict)
    """Final aggregated metrics for the run."""
    artifacts: dict[str, str] = field(default_factory=dict)
    """Named artifact paths (e.g. {'final_checkpoint': 's3://...', ...})."""
    error: str | None = None
    """If status != 'completed', a short human-readable reason."""
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Algorithm — the training recipe (SFT, RL, DPO, custom...)
# ---------------------------------------------------------------------------


@runtime_checkable
class Algorithm(Protocol):
    """A training recipe.

    Implementations must declare:
      * ``name`` (class var) — registry key, also used as the YAML ``kind``.
      * ``Config`` (class var) — Pydantic model for the recipe's parameters.

    And implement:
      * ``train(ctx: RunContext) -> RunResult``
    """

    name: ClassVar[str]
    Config: ClassVar[type]

    def train(self, ctx: RunContext) -> RunResult: ...


# ---------------------------------------------------------------------------
# Verifier — produces a reward / boolean for a single (prompt, completion).
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    reward: float
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Verifier(Protocol):
    name: ClassVar[str]
    Config: ClassVar[type]

    def verify(
        self,
        *,
        prompt: str,
        completion: str,
        target: dict[str, Any],
    ) -> VerificationResult: ...


# ---------------------------------------------------------------------------
# Metric — reduces per-task rollout rewards to a scalar.
# ---------------------------------------------------------------------------


@runtime_checkable
class Metric(Protocol):
    name: ClassVar[str]

    def compute(self, task_rewards: Sequence[Sequence[float]]) -> float: ...


# ---------------------------------------------------------------------------
# DataStore — abstract input/output. Subclassed for local FS and Supabase.
# ---------------------------------------------------------------------------


@runtime_checkable
class DataStore(Protocol):
    """Abstract data sink+source.

    Implementations:
      * ``LocalDataStore`` — filesystem JSONL/parquet/json, no network.
      * ``SupabaseDataStore`` — REST against PostgREST tables.
      * ``InMemoryDataStore`` — for tests.
    """

    name: ClassVar[str]

    def read_jsonl(self, path: str) -> list[dict[str, Any]]: ...

    def write_jsonl(self, path: str, rows: Iterable[dict[str, Any]]) -> None: ...

    def read_json(self, path: str) -> Any: ...

    def write_json(self, path: str, value: Any) -> None: ...

    def exists(self, path: str) -> bool: ...

    def list(self, prefix: str) -> list[str]: ...


# ---------------------------------------------------------------------------
# Backend — Tinker / Local / Mock. Owns model resources.
# ---------------------------------------------------------------------------


@runtime_checkable
class Backend(Protocol):
    """Compute backend.

    A backend's job is to materialise a model + training resources and hand
    them to whichever algorithm runs. The Algorithm protocol is intentionally
    NOT parameterised by Backend: instead, the registry routes (recipe.kind,
    backend.kind) to a specific Algorithm implementation.
    """

    name: ClassVar[str]

    def prepare(self, *, model: dict[str, Any], run_dir: str) -> dict[str, Any]:
        """Return a dict of backend-specific handles attached to ctx.extras.

        e.g. for TinkerBackend: {'service_client': ..., 'training_client': ...,
        'tokenizer': ..., 'model_name': ...}.
        """
        ...

    def teardown(self, handles: dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# InferenceClient — synchronous text generation.
# ---------------------------------------------------------------------------


@runtime_checkable
class InferenceClient(Protocol):
    """Generate text given a prompt. Used by evaluators and RL rollouts."""

    name: ClassVar[str]

    def generate(
        self,
        *,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Transform — convert raw rows to typed rows.
# ---------------------------------------------------------------------------


@runtime_checkable
class Transform(Protocol):
    """Convert raw input rows to algorithm-ready typed rows.

    e.g. a transform takes raw rows like {query, answer, ...} and emits
    {messages: [...], ...} ready for the algorithm.
    """

    name: ClassVar[str]
    Config: ClassVar[type]

    def __call__(self, rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]: ...

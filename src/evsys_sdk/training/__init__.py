"""evsys_sdk.training — native training-loop package.

Three concerns, three modules:

* :mod:`~evsys_sdk.training.backend` — `Backend` Protocol, `SamplingClient`
  Protocol, in-memory `MockBackend` for tests. The only seam that talks to
  tinker (or a stand-in).
* :mod:`~evsys_sdk.training.loop` — `TrainingLoop` driver, `StepBuilder`
  Protocol, `TrainingBatch` dataclass, `Evaluator` Protocol, `LoopArtifacts`.
* :mod:`~evsys_sdk.training.checkpoints` — `CheckpointManager` (writes the
  manifest :class:`evsys_sdk.checkpoint.Checkpoint` reads).

Concrete algorithm wrappers (`native_sft`, `native_sdft`, `native_rl`) live
under :mod:`evsys_sdk.algorithms` and compose these three pieces; researchers
who want a custom algorithm can subclass `StepBuilder` and re-register.

See ``docs/DESIGN.md`` for the architecture overview, and the
"Writing a new algorithm" section in ``skills/using-the-sdk/SKILL.md`` for
a template.
"""

from __future__ import annotations

from .backend import (
    Backend,
    ForwardBackwardResult,
    LossCallable,
    MockBackend,
    MockSamplingClient,
    OptimStepResult,
    SamplingClient,
)
from .checkpoints import CheckpointManager, ManifestRow
from .loop import (
    Evaluator,
    LoopArtifacts,
    StepBuilder,
    TrainingBatch,
    TrainingLoop,
)
from .sft_data import row_to_datum, sft_tokenize
from .step_builder import SFTStepBuilder
from .templates import (
    Message,
    apply_template,
    messages_to_model_input,
    text_to_model_input,
)

# TinkerBackend is optional — it imports `tinker_cookbook` lazily for the
# tokenizer helper. Most tests don't need it; surface it conditionally so
# `import evsys_sdk.training` doesn't pull the cookbook for a MockBackend run.
try:  # pragma: no cover  — import-guard for environments without tinker
    from .tinker_backend import TinkerBackend, TinkerSamplingClient
except ImportError as _e:  # pragma: no cover
    TinkerBackend = None  # type: ignore[assignment]
    TinkerSamplingClient = None  # type: ignore[assignment]

__all__ = [
    "Backend",
    "CheckpointManager",
    "Evaluator",
    "ForwardBackwardResult",
    "LoopArtifacts",
    "LossCallable",
    "ManifestRow",
    "Message",
    "MockBackend",
    "MockSamplingClient",
    "OptimStepResult",
    "SFTStepBuilder",
    "SamplingClient",
    "StepBuilder",
    "TinkerBackend",
    "TinkerSamplingClient",
    "TrainingBatch",
    "TrainingLoop",
    "apply_template",
    "messages_to_model_input",
    "row_to_datum",
    "sft_tokenize",
    "text_to_model_input",
]

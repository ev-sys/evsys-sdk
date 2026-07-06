"""Pydantic config models — the canonical YAML schema.

The full experiment is described by ExperimentConfig, which is what gets
serialized to and from YAML. Each block uses a `kind:` discriminator that
selects the matching registry entry; the registered class's `.Config` model
validates the rest of that block.

`extra='forbid'` is set everywhere so an evolutionary algorithm can't drop a
key by typoing it — the YAML loader will reject unknown fields loudly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    """Base model for everything in this file — strict by default."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)


# ---------------------------------------------------------------------------
# Per-block "spec" shapes — generic envelopes around a `kind` + free params.
# The runner looks up `kind` in the appropriate registry and validates
# `params` against the registered class's .Config Pydantic model.
# ---------------------------------------------------------------------------


class AlgorithmConfig(_Strict):
    """Training algorithm (recipe) — one of the registered @register_algorithm."""

    kind: str
    """Registry key, e.g. 'sft', 'rl_grpo', 'dpo'."""
    params: dict[str, Any] = Field(default_factory=dict)
    """Algorithm-specific parameters; validated against <Algorithm>.Config."""


class VerifierSpec(_Strict):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class TransformSpec(_Strict):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class CallbackSpec(_Strict):
    """A training-loop callback to attach, by registry name + params. e.g.
    ``{kind: early_stopping, params: {metric: pass_rate, patience: 3}}``."""

    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class TraceSourceSpec(_Strict):
    """A trace-ingestion source, by registry name + params. e.g.
    ``{kind: langgraph, params: {project_name: my-agent}, pull_every: 60s}``."""

    kind: str
    """Registry key of the @register_trace_source adapter, e.g. 'langgraph'."""
    params: dict[str, Any] = Field(default_factory=dict)
    """Adapter-specific parameters; validated against <TraceSource>.Config."""
    pull_every: str = "60s"
    """Poll interval for the ``--watch`` daemon (duration string, e.g. '60s', '5m')."""
    since: str | None = None
    """ISO-8601 start time; overrides the stored cursor for the first pull."""
    window: str | None = None
    """Lookback window (e.g. '24h') used when there is no cursor yet."""
    state_dir: str = ".evsys/traces"
    """Local dir where pulled traces + the cursor land."""


class TracesConfig(_Strict):
    """The ``traces`` section of :class:`SystemConfig` — trace ingestion sources."""

    trace_sources: list[TraceSourceSpec] = Field(default_factory=list)


class SystemConfig(_Strict):
    """The continual-learning **system** config (one ``system.yaml``) — the loop
    around individual experiments.

    Distinct from :class:`ExperimentConfig`, which describes ONE training
    experiment (the atom the autoresearch agent runs). This describes the system
    that decides *when* to run experiments and *what* to do with the results.
    Sections are added as layers land::

        traces:            # Layer 1 (this) — pull production traces in
          trace_sources: [...]
        # trigger: ...     # Layer 2 (future) — decide "worth learning from?"
        # deployment: ...  # Layer 3 (future) — gate + ship the winner
    """

    traces: TracesConfig = Field(default_factory=TracesConfig)


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


class DataStoreSpec(_Strict):
    kind: str = "local"
    """e.g. 'local', 'supabase', 'in_memory'."""
    params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data, Model, Backend, Eval
# ---------------------------------------------------------------------------


class DataConfig(_Strict):
    """Where the training data comes from + how to transform it.

    The preferred way to reference a dataset is by ``dataset_id`` (or
    ``dataset_name``): the SDK pulls it from the dashboard into the local
    ``.evsys/`` workspace and trains from that cache — so stored
    experiment scripts are portable and don't depend on local file layout.
    ``dataset_id`` / ``dataset_name`` take precedence over ``source_kind`` /
    ``path``, which remain as an offline / dev fallback.
    """

    source_kind: Literal["jsonl", "json", "in_memory", "hf_dataset"] = "jsonl"
    """Which built-in source loader to use (ignored when dataset_id/name set)."""
    dataset_id: str | None = None
    """Dashboard dataset id — pulled into .evsys/ and trained from locally."""
    dataset_name: str | None = None
    """Dashboard dataset name — resolved to the latest version's id, then pulled."""
    path: str | None = None
    """For 'jsonl' / 'json': absolute or store-relative path."""
    rows: list[dict[str, Any]] | None = None
    """For 'in_memory'."""
    hf_dataset: str | None = None
    """For 'hf_dataset': HuggingFace dataset id."""
    hf_split: str = "train"
    transforms: list[TransformSpec] = Field(default_factory=list)
    """Applied in order to the raw rows."""


class ModelConfig(_Strict):
    """Which model to fine-tune."""

    name: str
    """HuggingFace identifier, e.g. 'Qwen/Qwen3-4B'."""
    load_checkpoint_path: str | None = None
    """Resume from a previous checkpoint *with* optimizer state (full resume)."""
    init_from_checkpoint: str | None = None
    """Initialise weights from a previous checkpoint but start a *fresh*
    optimizer (weights-only). Used to chain continual-learning stages."""
    renderer_name: str | None = None
    """Tinker chat renderer hint, e.g. 'qwen3'."""


class BackendConfig(_Strict):
    """Compute backend."""

    kind: Literal["mock", "local", "tinker"] = "tinker"
    params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# RunConfig — one training run.
# ---------------------------------------------------------------------------


class RunConfig(_Strict):
    """A single training run (one cell in a campaign matrix)."""

    name: str
    """Human-readable name; unique within the experiment."""
    data: DataConfig
    model: ModelConfig
    algorithm: AlgorithmConfig
    backend: BackendConfig = Field(default_factory=BackendConfig)
    seed: int = 42
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ExperimentConfig — the top-level YAML object.
# ---------------------------------------------------------------------------


class ExperimentConfig(_Strict):
    """Top-level experiment.

    Either ``run`` (one run) or ``runs`` (multiple — a campaign) must be set.

    The ``matrix`` shorthand expands into ``runs`` automatically: each axis
    maps to a list of values, and the cartesian product is taken. Each cell
    inherits everything from ``base_run`` and overrides the matrixed fields.
    """

    version: int = 1
    """Schema version. Bumped when breaking changes are made."""
    name: str
    description: str = ""
    output_dir: str = "./outputs"
    """Where local artifacts (checkpoints, logs) are written."""
    log_rollouts: bool = False
    """When true, on-policy training rollouts (RL/SDFT) are logged via the
    ``on_rollout`` hook. A ``--dry`` run turns this on (and caps steps)."""
    data_store: DataStoreSpec = Field(default_factory=DataStoreSpec)

    # Logger callbacks ({kind, params}) built ONCE per experiment and shared
    # across all arms + their training loops. Each subscribes to the full
    # lifecycle (on_experiment_start → on_run_start → on_step_end /
    # on_benchmark_eval → on_run_end) and persists to its backend. e.g.
    #   callbacks: [{kind: wandb_logger}, {kind: tensorboard_logger}]
    callbacks: list[CallbackSpec] = Field(default_factory=list)

    # Exactly one of these three.
    run: RunConfig | None = None
    runs: list[RunConfig] | None = None
    matrix: MatrixSpec | None = None

    # -- Continual learning ---------------------------------------------------
    # Optional modifier (not a fourth mode): when set, the single ``run`` is
    # trained once per dataset in ``continual.datasets``, in order, where each
    # stage starts from the previous stage's weights (fresh optimizer). All
    # stages live in one experiment and each is scored on all benchmarks.
    continual: ContinualConfig | None = None

    # -- Run groups (variance studies) ---------------------------------------
    # When ``n_repeats > 1``, each primary RunConfig (from ``run`` / ``runs`` /
    # ``matrix``) becomes a *group*: it's replicated N times with seeds
    # ``[base_seed, base_seed+1, ...]`` (or ``[primary.seed, primary.seed+1, ...]``
    # when ``base_seed`` is None). Replicates share a dashboard ``group_id`` so
    # mean/stddev across seeds is straightforward. ``n_repeats: 1`` (default)
    # preserves the original behavior — no groups created.
    n_repeats: int = 1
    """How many seeded replicates per primary RunConfig. 1 = no grouping."""
    base_seed: int | None = None
    """Starting seed for auto-generated replicates. None → use each primary's own ``seed``."""

    parent_experiment_id: str | None = None
    """For evolutionary lineage."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Free-form (e.g. budget, hypothesis, client tag)."""

    def model_post_init(self, __context: Any) -> None:
        n = sum(x is not None for x in (self.run, self.runs, self.matrix))
        if n != 1:
            raise ValueError(
                "ExperimentConfig must have exactly one of: run, runs, matrix "
                f"(found {n})."
            )
        if self.n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1 (got {self.n_repeats}).")
        if self.continual is not None and self.run is None:
            raise ValueError(
                "continual requires a single `run` as the base (not runs/matrix)."
            )


class MatrixSpec(_Strict):
    """Cartesian-product expansion of a base run.

    Example:
        matrix:
          base_run:
            ...                # full RunConfig template
          axes:
            algorithm.params.lora_rank: [1, 8]
            algorithm.params.learning_rate: [1e-4, 5e-5]
          name_template: "{base}__rank{algorithm.params.lora_rank}__lr{algorithm.params.learning_rate}"
    """

    base_run: RunConfig
    axes: dict[str, list[Any]] = Field(default_factory=dict)
    name_template: str | None = None
    """Optional template for run names. Uses {key} for axis values + {base}."""


class ContinualConfig(_Strict):
    """Continual-learning stages over a single base ``run``.

    Each entry in ``datasets`` becomes one training stage: the base ``run`` is
    copied with its ``data`` replaced by that entry, trained in order, and each
    stage starts from the previous stage's weights (fresh optimizer). All stages
    run inside one experiment and are scored on all configured benchmarks.

    Example:
        run:
          data: {...}            # ignored; the per-stage data below is used
          model: {...}
          algorithm: {kind: sft, ...}
        continual:
          datasets:
            - {dataset_name: corpus_a, transforms: [...]}
            - {dataset_name: corpus_b, transforms: [...]}
            - {dataset_name: corpus_c, transforms: [...]}
    """

    datasets: list[DataConfig] = Field(min_length=1)
    name_template: str | None = None
    """Optional stage-name template; uses {base} and {i}. Default '{base}_stage{i}'."""


# pydantic v2 forward-ref resolution
ExperimentConfig.model_rebuild()

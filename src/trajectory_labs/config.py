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


class MetricSpec(_Strict):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class TransformSpec(_Strict):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class InferenceSpec(_Strict):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


class DataStoreSpec(_Strict):
    kind: str = "local"
    """e.g. 'local', 'supabase', 'in_memory'."""
    params: dict[str, Any] = Field(default_factory=dict)


class LogStoreSpec(_Strict):
    kind: str = "jsonl"
    """e.g. 'jsonl', 'tensorboard', 'multiplex', 'supabase'."""
    params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data, Model, Backend, Eval
# ---------------------------------------------------------------------------


class DataConfig(_Strict):
    """Where the training data comes from + how to transform it.

    The preferred way to reference a dataset is by ``dataset_id`` (or
    ``dataset_name``): the SDK pulls it from the dashboard into the local
    ``.trajectory/`` workspace and trains from that cache — so stored
    experiment scripts are portable and don't depend on local file layout.
    ``dataset_id`` / ``dataset_name`` take precedence over ``source_kind`` /
    ``path``, which remain as an offline / dev fallback.
    """

    source_kind: Literal["jsonl", "json", "in_memory", "hf_dataset"] = "jsonl"
    """Which built-in source loader to use (ignored when dataset_id/name set)."""
    dataset_id: str | None = None
    """Dashboard dataset id — pulled into .trajectory/ and trained from locally."""
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
    """Resume from a previous Tinker / local checkpoint."""
    renderer_name: str | None = None
    """Tinker chat renderer hint, e.g. 'qwen3'."""


class BackendConfig(_Strict):
    """Compute backend."""

    kind: Literal["mock", "local", "tinker"] = "tinker"
    params: dict[str, Any] = Field(default_factory=dict)


class EvalConfig(_Strict):
    """Optional eval pass after training."""

    enabled: bool = True
    metrics: list[MetricSpec] = Field(default_factory=list)
    inference: InferenceSpec | None = None
    """How to query the trained model for eval. Defaults to a backend-native client."""
    eval_data: DataConfig | None = None
    """Eval set; if absent, training data's eval split is used."""
    n_samples: int | None = None
    """Cap on how many examples to eval (None = all)."""


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
    eval: EvalConfig = Field(default_factory=EvalConfig)
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
    data_store: DataStoreSpec = Field(default_factory=DataStoreSpec)
    log_store: LogStoreSpec = Field(default_factory=LogStoreSpec)

    # Exactly one of these three.
    run: RunConfig | None = None
    runs: list[RunConfig] | None = None
    matrix: MatrixSpec | None = None

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


# pydantic v2 forward-ref resolution
ExperimentConfig.model_rebuild()

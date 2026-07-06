"""evsys_sdk — declarative, modular LLM experiment framework.

Most researcher code only needs the OOP orchestration surface:

    from evsys_sdk import Experiment
    Experiment.from_yaml("config.yaml").run()

For everything else:

    from evsys_sdk import (
        # OOP orchestration
        Experiment, ExperimentResult, ArmResult, Sweep,
        Benchmark, BenchmarkScore, Checkpoint,
        # Config models
        ExperimentConfig, RunConfig, AlgorithmConfig, DataConfig, ModelConfig,
        BackendConfig, VerifierSpec,
        # YAML
        load_yaml, dump_yaml, validate_yaml,
        # Registries (decorators for extensions)
        register_algorithm, register_verifier, register_metric,
        register_backend, register_data_store, register_log_store,
        register_inference, register_transform,
        get_algorithm, get_verifier, get_metric,
        # Imperative runner (kept for advanced use; Experiment is the default)
        run_experiment,
    )

Built-in extensions live in subpackages and self-register on import.
External packages can extend any registry via Python entry points
(group: ``evsys_sdk.<plural>`` — see docs/cookbook.md).
"""

from . import _entry_points  # noqa: F401  loads third-party extensions

# Trigger registration of built-in extensions.
from . import algorithms as _algorithms  # noqa: F401
from . import backends as _backends  # noqa: F401
from . import data_stores as _data_stores  # noqa: F401
from . import inference as _inference  # noqa: F401
from . import log_stores as _log_stores  # noqa: F401
from . import metrics as _metrics  # noqa: F401
from . import transforms as _transforms  # noqa: F401
from . import verifiers as _verifiers  # noqa: F401
from .benchmark import Benchmark, BenchmarkScore, BenchmarkTaskResult
from .benchmark_run import run_benchmark
from .checkpoint import Checkpoint, find_manifest, read_manifest
from .config import (
    AlgorithmConfig,
    BackendConfig,
    CallbackSpec,
    DataConfig,
    DataStoreSpec,
    ExperimentConfig,
    LogStoreSpec,
    ModelConfig,
    RunConfig,
    TransformSpec,
    VerifierSpec,
)
from .dashboard_client import (
    DashboardClient,
    DashboardClientError,
    EvsysAuthError,
    ExperimentRun,
)
from .data_types import (
    ChatMessagesRow,
    E2BVerifier,
    HarborTask,
    InProcessVerifier,
    LLMJudgeVerifier,
    PromptExample,
    TargetFormat,
    VerifierPayload,
    block_to_image_src,
    chat_messages_row_from_dict,
    detect_format,
    from_dict,
    harbor_task_from_dict,
    has_images,
    image_base64_block,
    image_url_block,
    iter_jsonl,
    parse_rows,
    prompt_example_from_dict,
    text_block,
    to_dict,
)
from .experiment import ArmResult, EvalResult, Experiment, ExperimentResult
from .logger import configure_logger, get_logger, set_level
from .protocols import (
    Algorithm,
    Backend,
    DataStore,
    InferenceClient,
    LogStore,
    Metric,
    RunContext,
    RunResult,
    Transform,
    Verifier,
)
from .registry import (
    get_algorithm,
    get_backend,
    get_callback,
    get_data_store,
    get_inference,
    get_log_store,
    get_metric,
    get_transform,
    get_verifier,
    list_algorithms,
    list_backends,
    list_callbacks,
    list_data_stores,
    list_inferences,
    list_log_stores,
    list_metrics,
    list_transforms,
    list_verifiers,
    register_algorithm,
    register_backend,
    register_callback,
    register_data_store,
    register_inference,
    register_log_store,
    register_metric,
    register_transform,
    register_verifier,
)
from .runner import run_experiment
from .step_metrics import forward_step_metrics
from .store import EvsysStore, EvsysStoreError
from .sweep import Sweep, expand_runs
from .workspace import MaterializedDataset, Workspace
from .yaml_loader import dump_yaml, load_yaml, validate_yaml

__version__ = "0.1.0"

__all__ = [
    # Protocols
    "Algorithm",
    # Config
    "AlgorithmConfig",
    # OOP orchestration (researcher-project layout)
    "ArmResult",
    "Backend",
    "BackendConfig",
    "Benchmark",
    "BenchmarkScore",
    "BenchmarkTaskResult",
    "CallbackSpec",
    "ChatMessagesRow",
    "Checkpoint",
    # Dashboard client (push runs to the EvolvingSystems dashboard)
    "DashboardClient",
    "DashboardClientError",
    "DataConfig",
    "DataStore",
    "DataStoreSpec",
    "E2BVerifier",
    "EvalResult",
    "EvsysAuthError",
    # Backend-routed data-access (project → … → runs → evals/metrics)
    "EvsysStore",
    "EvsysStoreError",
    "Experiment",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRun",
    "HarborTask",
    "InProcessVerifier",
    "InferenceClient",
    "LLMJudgeVerifier",
    "LogStore",
    "LogStoreSpec",
    "MaterializedDataset",
    "Metric",
    "ModelConfig",
    "PromptExample",
    "RunConfig",
    "RunContext",
    "RunResult",
    "Sweep",
    # Harbor data shapes (data interchange with internal stack + dashboards)
    "TargetFormat",
    "Transform",
    "TransformSpec",
    "Verifier",
    "VerifierPayload",
    "VerifierSpec",
    # Local cache for remote datasets/benchmarks
    "Workspace",
    "__version__",
    "block_to_image_src",
    "chat_messages_row_from_dict",
    # Logging
    "configure_logger",
    "detect_format",
    # YAML
    "dump_yaml",
    "expand_runs",
    "find_manifest",
    "forward_step_metrics",
    "from_dict",
    # Registry
    "get_algorithm",
    "get_backend",
    "get_callback",
    "get_data_store",
    "get_inference",
    "get_log_store",
    "get_logger",
    "get_metric",
    "get_transform",
    "get_verifier",
    "harbor_task_from_dict",
    "has_images",
    "image_base64_block",
    "image_url_block",
    "iter_jsonl",
    "list_algorithms",
    "list_backends",
    "list_callbacks",
    "list_data_stores",
    "list_inferences",
    "list_log_stores",
    "list_metrics",
    "list_transforms",
    "list_verifiers",
    "load_yaml",
    "parse_rows",
    "prompt_example_from_dict",
    "read_manifest",
    "register_algorithm",
    "register_backend",
    "register_callback",
    "register_data_store",
    "register_inference",
    "register_log_store",
    "register_metric",
    "register_transform",
    "register_verifier",
    "run_benchmark",
    # Runner
    "run_experiment",
    "set_level",
    "text_block",
    "to_dict",
    "validate_yaml",
]

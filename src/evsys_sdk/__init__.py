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

from .config import (
    AgentSpec,
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
    get_agent,
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
    list_agents,
    register_agent,
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
from .benchmark import Benchmark, BenchmarkScore, BenchmarkTaskResult
from .benchmark_run import run_benchmark
from .checkpoint import Checkpoint, find_manifest, read_manifest
from .experiment import ArmResult, EvalResult, Experiment, ExperimentResult
from .runner import run_experiment
from .step_metrics import forward_step_metrics
from .sweep import Sweep, expand_runs
from .yaml_loader import dump_yaml, load_yaml, validate_yaml
from .dashboard_client import (
    DashboardClient,
    DashboardClientError,
    ExperimentRun,
    EvsysAuthError,
)
from .logger import configure_logger, get_logger, set_level
from .store import EvsysStore, EvsysStoreError
from .workspace import MaterializedDataset, Workspace
from .data_types import (
    TargetFormat,
    ChatMessagesRow,
    HarborTask,
    PromptExample,
    InProcessVerifier,
    E2BVerifier,
    LLMJudgeVerifier,
    VerifierPayload,
    text_block,
    image_url_block,
    image_base64_block,
    block_to_image_src,
    has_images,
    detect_format,
    harbor_task_from_dict,
    chat_messages_row_from_dict,
    prompt_example_from_dict,
    from_dict,
    parse_rows,
    to_dict,
    iter_jsonl,
)

# Trigger registration of built-in extensions.
from . import algorithms as _algorithms  # noqa: F401
from . import backends as _backends  # noqa: F401
from . import data_stores as _data_stores  # noqa: F401
from . import inference as _inference  # noqa: F401
from . import log_stores as _log_stores  # noqa: F401
from . import metrics as _metrics  # noqa: F401
from . import transforms as _transforms  # noqa: F401
from . import verifiers as _verifiers  # noqa: F401
from . import _entry_points  # noqa: F401  loads third-party extensions

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Config
    "AgentSpec",
    "AlgorithmConfig",
    "BackendConfig",
    "DataConfig",
    "DataStoreSpec",
    "CallbackSpec",
    "ExperimentConfig",
    "LogStoreSpec",
    "ModelConfig",
    "RunConfig",
    "TransformSpec",
    "VerifierSpec",
    # Protocols
    "Algorithm",
    "Backend",
    "DataStore",
    "InferenceClient",
    "LogStore",
    "Metric",
    "RunContext",
    "RunResult",
    "Transform",
    "Verifier",
    # Registry
    "get_agent",
    "get_algorithm",
    "get_backend",
    "get_callback",
    "get_data_store",
    "get_inference",
    "get_log_store",
    "get_metric",
    "get_transform",
    "get_verifier",
    "list_agents",
    "list_algorithms",
    "list_backends",
    "list_callbacks",
    "list_data_stores",
    "list_inferences",
    "list_log_stores",
    "list_metrics",
    "list_transforms",
    "list_verifiers",
    "register_agent",
    "register_algorithm",
    "register_backend",
    "register_callback",
    "register_data_store",
    "register_inference",
    "register_log_store",
    "register_metric",
    "register_transform",
    "register_verifier",
    # Runner
    "run_experiment",
    # YAML
    "dump_yaml",
    "load_yaml",
    "validate_yaml",
    # Harbor data shapes (data interchange with internal stack + dashboards)
    "TargetFormat",
    "ChatMessagesRow",
    "HarborTask",
    "PromptExample",
    "InProcessVerifier",
    "E2BVerifier",
    "LLMJudgeVerifier",
    "VerifierPayload",
    "text_block",
    "image_url_block",
    "image_base64_block",
    "block_to_image_src",
    "has_images",
    "detect_format",
    "harbor_task_from_dict",
    "chat_messages_row_from_dict",
    "prompt_example_from_dict",
    "from_dict",
    "parse_rows",
    "to_dict",
    "iter_jsonl",
    # Dashboard client (push runs to the EvolvingSystems dashboard)
    "DashboardClient",
    "DashboardClientError",
    "EvsysAuthError",
    "ExperimentRun",
    # Logging
    "configure_logger",
    "get_logger",
    "set_level",
    # Backend-routed data-access (project → … → runs → evals/metrics)
    "EvsysStore",
    "EvsysStoreError",
    # Local cache for remote datasets/benchmarks
    "Workspace",
    "MaterializedDataset",
    # OOP orchestration (researcher-project layout)
    "ArmResult",
    "Benchmark",
    "run_benchmark",
    "BenchmarkScore",
    "BenchmarkTaskResult",
    "Checkpoint",
    "EvalResult",
    "Experiment",
    "ExperimentResult",
    "Sweep",
    "expand_runs",
    "find_manifest",
    "forward_step_metrics",
    "read_manifest",
]

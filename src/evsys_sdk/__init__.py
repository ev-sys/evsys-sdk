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
        load_yaml, dump_yaml, validate_yaml, apply_dry_run,
        # Registries (decorators for extensions)
        register_algorithm, register_verifier, register_metric,
        register_backend, register_data_store,
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
    FunctionSpec,
    ModelConfig,
    RemoteAgentConfig,
    RunConfig,
    SandboxSpec,
    SystemConfig,
    TracesConfig,
    TraceSourceSpec,
    TransformSpec,
    TriggerAgentConfig,
    TriggerConfig,
    VerifierSpec,
)
from .protocols import (
    Algorithm,
    Backend,
    DataStore,
    InferenceClient,
    Metric,
    RunContext,
    RunResult,
    Transform,
    Trigger,
    TriggerDecision,
    Verifier,
)
from .registry import (
    get_agent,
    get_algorithm,
    get_backend,
    get_callback,
    get_data_store,
    get_function,
    get_inference,
    get_metric,
    get_sandbox,
    get_context_source,
    get_trace_source,
    get_transform,
    get_trigger,
    get_verifier,
    list_agents,
    list_algorithms,
    list_backends,
    list_callbacks,
    list_data_stores,
    list_functions,
    list_inferences,
    list_metrics,
    list_sandboxes,
    list_context_sources,
    list_computes,
    list_trace_sources,
    list_transforms,
    list_triggers,
    list_verifiers,
    register_agent,
    register_algorithm,
    register_backend,
    register_callback,
    register_data_store,
    register_function,
    register_inference,
    register_metric,
    register_sandbox,
    register_context_source,
    register_trace_source,
    register_transform,
    register_trigger,
    register_verifier,
)
from .benchmark import Benchmark, BenchmarkScore, BenchmarkTaskResult
from .benchmark_run import run_benchmark
from .checkpoint import Checkpoint, find_manifest, read_manifest
from .experiment import ArmResult, EvalResult, Experiment, ExperimentResult
from .runner import run_experiment
from .sweep import Sweep, expand_runs
from .yaml_loader import apply_dry_run, dump_yaml, load_yaml, validate_yaml
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
from .context_types import ContextItem
from .trace_types import Trace, trace_from_dict, iter_traces_jsonl

# Trigger registration of built-in extensions.
from . import agents as _agents  # noqa: F401
from .agents import AutoresearchAgent, EvsysAgent, TriggerAgent, build_agent
from . import algorithms as _algorithms  # noqa: F401
from . import sandboxes as _sandboxes  # noqa: F401
from .sandboxes import BaseSandbox
from . import functions as _functions  # noqa: F401
from .functions import EvsysFunction, TriggerFunction, VerifierFunction, build_function
from . import trace_sources as _trace_sources  # noqa: F401
from . import triggers as _triggers  # noqa: F401
from . import context_sources as _context_sources  # noqa: F401
from . import backends as _backends  # noqa: F401
# Compute targets — where a training service runs (SkyPilot, …).
from . import compute as _compute  # noqa: F401
from .compute import BaseCompute, build_compute
from . import data_stores as _data_stores  # noqa: F401
from . import inference as _inference  # noqa: F401
from . import metrics as _metrics  # noqa: F401
from . import transforms as _transforms  # noqa: F401
from . import verifiers as _verifiers  # noqa: F401
from . import _entry_points  # noqa: F401  loads third-party extensions

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Config
    "AlgorithmConfig",
    "BackendConfig",
    "DataConfig",
    "DataStoreSpec",
    "CallbackSpec",
    "ExperimentConfig",
    "ModelConfig",
    "RunConfig",
    "TransformSpec",
    "VerifierSpec",
    # Protocols
    "Algorithm",
    "Backend",
    "DataStore",
    "InferenceClient",
    "Metric",
    "RunContext",
    "RunResult",
    "Transform",
    "Verifier",
    # Registry
    "get_algorithm",
    "get_backend",
    "get_callback",
    "get_data_store",
    "get_inference",
    "get_metric",
    "get_transform",
    "get_verifier",
    "list_algorithms",
    "list_backends",
    "list_computes",
    "BaseCompute",
    "build_compute",
    "list_callbacks",
    "list_data_stores",
    "list_inferences",
    "list_metrics",
    "list_transforms",
    "list_verifiers",
    "register_algorithm",
    "register_backend",
    "register_callback",
    "register_data_store",
    "register_inference",
    "register_metric",
    "register_transform",
    "register_verifier",
    "register_trace_source",
    "get_trace_source",
    "list_trace_sources",
    "register_trigger",
    "get_trigger",
    "list_triggers",
    "register_sandbox",
    "get_sandbox",
    "list_sandboxes",
    "BaseSandbox",
    "SandboxSpec",
    # Agents (LLM agents the SDK spawns)
    "register_agent",
    "get_agent",
    "list_agents",
    "EvsysAgent",
    "TriggerAgent",
    "AutoresearchAgent",
    "build_agent",
    "AgentSpec",
    # Functions (deterministic fns the system runs)
    "register_function",
    "get_function",
    "list_functions",
    "EvsysFunction",
    "TriggerFunction",
    "VerifierFunction",
    "build_function",
    "FunctionSpec",
    "register_context_source",
    "get_context_source",
    "list_context_sources",
    # Trace ingestion + system config
    "Trace",
    "trace_from_dict",
    "ContextItem",
    "iter_traces_jsonl",
    "SystemConfig",
    "TracesConfig",
    "TraceSourceSpec",
    "TriggerConfig",
    "TriggerAgentConfig",
    "RemoteAgentConfig",
    # Trigger (Layer 2 — the cheap gate)
    "Trigger",
    "TriggerDecision",
    # Runner
    "run_experiment",
    # YAML
    "dump_yaml",
    "load_yaml",
    "apply_dry_run",
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
    "read_manifest",
]

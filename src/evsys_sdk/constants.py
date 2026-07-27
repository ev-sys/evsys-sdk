"""Central constants for the evsys_sdk SDK.

Single source of truth for env-var names, defaults, HTTP endpoint paths,
status strings, and logging config. Change endpoints / defaults here — not
scattered across the codebase.

Distinct from ``config.py``, which holds the user-facing Pydantic *experiment*
config models (ExperimentConfig, RunConfig, ...). This module is SDK app-level
plumbing.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Environment variable names (all SDK env vars start with EVSYS_)
# ---------------------------------------------------------------------------

EVSYS_API_URL_ENV = "EVSYS_API_URL"
EVSYS_API_KEY_ENV = "EVSYS_API_KEY"
EVSYS_PROJECT_ID_ENV = "EVSYS_PROJECT_ID"
EVSYS_LOG_DIR_ENV = "EVSYS_LOG_DIR"
EVSYS_OFFLINE_ENV = "EVSYS_OFFLINE"
EVSYS_LOGGING_LEVEL_ENV = "EVSYS_LOGGING_LEVEL"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_LOG_DIR = "./evsys_sdk"
DEFAULT_TIMEOUT_S = 30.0

# ---------------------------------------------------------------------------
# HTTP endpoint paths
# ---------------------------------------------------------------------------

# Prefix that sits between base_url and the per-resource paths below.
API_PREFIX = "/api/dashboard/api"

# SDK write routes (relative to API_PREFIX). Runs schema (D18): generations → runs.
EP_CREATE_EXPERIMENT = "/sdk/experiments/"
EP_UPDATE_EXPERIMENT = "/sdk/experiments/{experiment_id}/"
EP_CREATE_RUN = "/sdk/runs/"
EP_UPDATE_RUN = "/sdk/runs/{run_id}/"
EP_LOG_METRICS = "/sdk/runs/{run_id}/metrics/"          # long-format run_metrics (D15)
EP_LOG_EVAL = "/sdk/runs/{run_id}/evals/"               # evals (D13)
EP_ADD_CHECKPOINT = "/sdk/runs/{run_id}/checkpoints/"   # run_checkpoints (D7)
EP_LOG_PREDICTIONS = "/sdk/runs/{run_id}/predictions/"

# ---------------------------------------------------------------------------
# JSON response / field keys
# ---------------------------------------------------------------------------

KEY_EXPERIMENT = "experiment"
KEY_RUN = "run"
KEY_RAW = "_raw"

FIELD_PROJECT_ID = "project_id"
FIELD_BEST_SCORE = "best_score"
FIELD_ERROR_MESSAGE = "error_message"
FIELD_CONCLUSION = "conclusion"

# ---------------------------------------------------------------------------
# Status strings
# ---------------------------------------------------------------------------

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

# ---------------------------------------------------------------------------
# HTTP headers
# ---------------------------------------------------------------------------

HEADER_AUTHORIZATION = "Authorization"
HEADER_CONTENT_TYPE = "Content-Type"
CONTENT_TYPE_JSON = "application/json"


def bearer(api_key: str) -> str:
    return f"Bearer {api_key}"


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

DEFAULT_LOGGING_LEVEL = "INFO"
SUPPORTED_LOGGING_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOGGER_NAME = "evsys_sdk"

# ---------------------------------------------------------------------------
# Local mirror layout (always-on, wandb-offline style)
# ---------------------------------------------------------------------------

LOCAL_EXPERIMENT_FILE = "experiment.json"
LOCAL_GENERATION_FILE = "generation.json"
LOCAL_METRICS_FILE = "metrics.jsonl"
LOCAL_EVALS_FILE = "evals.jsonl"
LOCAL_PREDICTIONS_FILE = "predictions.jsonl"
LOCAL_CHECKPOINTS_FILE = "checkpoints.jsonl"


def truthy_env(value: str | None) -> bool:
    """Interpret an env var string as a boolean."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}

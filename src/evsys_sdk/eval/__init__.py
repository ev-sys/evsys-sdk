"""Generic eval infra for tool-detection / model-output scoring.

Public surface:

    from evsys_sdk.eval import (
        AliasMatcher, ModelEvalConfig,
        evaluate_model, EvalArtifacts, EvalSummary,
        RetryReport, call_with_retry,
        score_rows, format_summary_markdown,
    )

Every inference call is wrapped in ``call_with_retry`` (configurable attempts +
backoff); exhausted failures surface via ``RetryReport`` instead of aborting.
This module is domain-agnostic — project-specific eval harnesses (e.g. an API
search eval) build on this infra in their own repos.
"""

from .matcher import AliasMatcher
from .model_eval import (
    ModelEvalConfig,
    ModelEvalResult,
    extract_predicted_slug,
    qwen_chat_prompt,
    run_model_eval,
)
from .report import (
    EvalSummary,
    format_summary_markdown,
    model_query_found,
    score_rows,
)
from .retry import RetryFailure, RetryReport, call_with_retry
from .runner import EvalArtifacts, evaluate_model, load_eval_dataset

__all__ = [
    "AliasMatcher",
    "EvalArtifacts",
    "EvalSummary",
    "ModelEvalConfig",
    "ModelEvalResult",
    "RetryFailure",
    "RetryReport",
    "call_with_retry",
    "evaluate_model",
    "extract_predicted_slug",
    "format_summary_markdown",
    "load_eval_dataset",
    "model_query_found",
    "qwen_chat_prompt",
    "run_model_eval",
    "score_rows",
]

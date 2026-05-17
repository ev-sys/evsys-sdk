"""Eval harness for tool-detection tasks.

Public surface:

    from trajectory_experiments.eval import (
        AliasMatcher,
        ComposioSearchConfig, ModelEvalConfig,
        evaluate_composio_search, evaluate_model,
        EvalArtifacts, EvalSummary,
        RetryReport, call_with_retry,
        format_summary_markdown,
    )

The whole module wraps every Composio API / inference call in
``call_with_retry`` (5 attempts, exponential backoff) and surfaces
exhausted failures via ``RetryReport`` instead of aborting the run.
"""

from .composio_search import ComposioSearchConfig, ComposioSearchEvalResult, run_composio_search_eval
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
    composio_query_found,
    composio_query_found_primary,
    format_summary_markdown,
    model_query_found,
    score_rows,
)
from .retry import RetryFailure, RetryReport, call_with_retry
from .runner import EvalArtifacts, evaluate_composio_search, evaluate_model, load_eval_dataset

__all__ = [
    "AliasMatcher",
    "ComposioSearchConfig",
    "ComposioSearchEvalResult",
    "EvalArtifacts",
    "EvalSummary",
    "ModelEvalConfig",
    "ModelEvalResult",
    "RetryFailure",
    "RetryReport",
    "call_with_retry",
    "composio_query_found",
    "composio_query_found_primary",
    "evaluate_composio_search",
    "evaluate_model",
    "extract_predicted_slug",
    "format_summary_markdown",
    "load_eval_dataset",
    "model_query_found",
    "qwen_chat_prompt",
    "run_composio_search_eval",
    "run_model_eval",
    "score_rows",
]

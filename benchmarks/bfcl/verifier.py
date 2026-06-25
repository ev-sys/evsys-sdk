"""``bfcl_match`` — the in-process verifier fn that scores a BFCL completion.

This is a faithful, ``inspect_ai``-free port of the BFCL AST / abstention
scorer (Berkeley Function-Call Leaderboard, gorilla commit
``dac44e7ac9db5ff26a01ab0c1ec5de5a1e703b7a``). It registers into the SDK's
in-process verifier-fn registry — the same one ``InProcessVerifier(fn_name=…)``
resolves against at harbor rollout time — so a benchmark task can carry
``verifier = {kind: in_process, fn_name: "bfcl_match", expected: {...}}`` and be
scored through :meth:`Benchmark.score_via_harbor` with no SDK edit.

The harbor host-side ``EvsysVerifier`` calls us as ``fn(completion, expected,
params) -> float`` where ``completion`` is the decoded model text. We:

  1. parse hermes ``<tool_call>{"name": …, "arguments": {…}}</tool_call>`` blocks
     out of the completion (tolerant of minor format noise);
  2. dispatch on ``expected["category"]``'s matching function:
       * ``simple`` / ``parallel`` / ``multiple`` → type-aware AST match
         (order-independent for parallel) against the ground-truth possible
         answers, returning 1.0/0.0;
       * ``irrelevance`` → 1.0 iff NO call was emitted (correct abstention);
       * ``relevance``   → 1.0 iff a call WAS emitted.

``expected`` is the per-task spec the dataset builder writes::

    {
      "category":     "live_multiple",        # raw BFCL category name
      "ground_truth": [{"func": {"p": [v1, v2]}}, ...],  # AST possible answers
      "tools":        [<FunctionDoc>, ...],   # tool schemas (for type rules)
      "language":     "python" | "java" | "js",
    }

The matching logic (``tool_call_matches_possible_answers``, ``_match_parallel``,
``_match_multiple``, ``_standardize_string``, the int/float type rules, and
``normalize_function_name``) is ported verbatim in behaviour from
``inspect_evals/bfcl/score/scorer.py`` + ``utils/{tool_parsing,function_parsing}.py``.
Importing this module fires the registration (see :data:`FN_NAME`).
"""

from __future__ import annotations

import json
import re
from typing import Any

# NB: this module imports NOTHING from ``evsys_sdk`` at module scope. The scorer
# (``parse_tool_calls`` + ``bfcl_match``) is pure stdlib, so the RAW harness
# (which must not import ``evsys_sdk``) can import it freely. The SDK-registry
# hook is deferred into ``_register_into_sdk`` and only fires if ``evsys_sdk`` is
# already importable — so importing this module never drags the SDK in.

# Name a HarborTask references in its verifier spec.
FN_NAME = "bfcl_match"


# ---------------------------------------------------------------------------
# Parsed-call shape
#
# A tool call parsed off the completion. Mirrors inspect's ``ToolCall`` to the
# two fields the scorer touches: ``function`` (the name) and ``arguments``
# (a name->value dict). Kept a plain tuple-free dataclass-lite dict-pair so the
# ported matchers read like the originals (``tc.function`` -> ``tc["function"]``).
# ---------------------------------------------------------------------------


def _mk_call(function: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"function": str(function), "arguments": dict(arguments)}


# ---------------------------------------------------------------------------
# Tool-call extraction from completion text
# ---------------------------------------------------------------------------

# Qwen / hermes function-call envelope. We tolerate optional whitespace and a
# missing closing tag at end-of-string (models occasionally truncate it).
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)


def parse_tool_calls(completion: str) -> list[dict[str, Any]]:
    """Extract every hermes ``<tool_call>`` JSON block from ``completion``.

    Each block is ``{"name": str, "arguments": {…}}`` (the Qwen3 / hermes
    function-call format). ``arguments`` may itself arrive as a JSON *string*
    (some templates double-encode it) — we decode that too. Blocks that don't
    parse as a JSON object carrying a ``name`` are skipped, not fatal: a noisy
    completion simply yields fewer calls (and the AST matchers then fail it on
    the call-count check, exactly like the official scorer).
    """
    calls: list[dict[str, Any]] = []
    for raw in _TOOL_CALL_RE.findall(completion or ""):
        obj = _loads_obj(raw)
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):  # double-encoded arguments
            args = _loads_obj(args)
        if not isinstance(args, dict):
            args = {}
        calls.append(_mk_call(name, args))
    return calls


def _loads_obj(s: str) -> Any:
    """Best-effort JSON-object parse: try as-is, else grab the first ``{...}``."""
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Type system — ports utils/tool_parsing.py::get_type + normalize_function_name
# ---------------------------------------------------------------------------

# JSON-schema types inspect's ToolParam accepts; BFCL types map onto these.
_JSON_TYPES = frozenset(
    {"string", "integer", "number", "boolean", "array", "object", "null"}
)
_BFCL_TYPE_MAP = {
    None: "null",
    "dict": "object",
    "float": "number",
    "tuple": "array",
    "any": "string",
}


def get_type(bfcl_type: str | None) -> str:
    """BFCL type string → JSON-schema type. Mirrors ``utils.get_type``."""
    json_type = _BFCL_TYPE_MAP.get(bfcl_type, bfcl_type)
    if json_type not in _JSON_TYPES:
        raise ValueError(f"Invalid type: {json_type}")
    return json_type


def normalize_function_name(name: str) -> str:
    """Some models emit ``a_b_c`` for ``a.b.c``; compare with dots→underscores."""
    return (name or "").replace(".", "_")


# ---------------------------------------------------------------------------
# Value standardisation — ports scorer.py::_standardize_string / _value_matches
# ---------------------------------------------------------------------------

# Per BFCL spec: case-insensitive, drop spaces + ``, . / - _ * ^``, and treat
# single quotes as double quotes (so nested-dict reprs compare equal).
_STD_STRIP_RE = re.compile(r"[ \,\.\/\-\_\*\^]")


def _standardize_string(s: str) -> str:
    return _STD_STRIP_RE.sub("", s).lower().replace("'", '"')


def _normalize_value(value: Any) -> Any:
    """Recursively standardise strings inside lists / dicts for comparison."""
    if isinstance(value, str):
        return _standardize_string(value)
    if isinstance(value, (tuple, list)):
        return [_normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    return value


def _value_matches(value: Any, possible_values: list[Any]) -> bool:
    """True iff ``value`` equals any of ``possible_values`` after normalisation.

    ``bool`` is a subclass of ``int`` in Python (``True == 1``); BFCL treats
    booleans and numbers as distinct types, so we reject a bool/non-bool pair.
    """
    normalized_value = _normalize_value(value)
    for possible in possible_values:
        normalized_possible = _normalize_value(possible)
        if isinstance(normalized_value, bool) != isinstance(normalized_possible, bool):
            continue
        if normalized_value == normalized_possible:
            return True
    return False


# ---------------------------------------------------------------------------
# int/float type rules — ports scorer.py::_apply_numeric_type_rules + arrays
# ---------------------------------------------------------------------------


def _apply_numeric_type_rules(
    value: Any, json_type: str, language: str, param: str, nested: bool = False,
) -> tuple[Any, dict[str, Any] | None]:
    """BFCL int/float rules. Returns ``(coerced_value, None)`` on success or
    ``(value, error_dict)`` on failure.

    * Python only: an ``int`` is accepted where a ``number`` (float) is expected
      (coerced to float).  Java / JS: a literal float is required, ``int`` fails.
    * A ``float`` supplied for an ``integer`` param is invalid in all languages.
    """
    prefix = (
        f"Nested type checking failed for {param!r}" if nested
        else f"Invalid type for {param}"
    )
    suffix = " elements" if nested else ""
    if isinstance(value, int) and not isinstance(value, bool) and json_type == "number":
        if language == "python":
            return float(value), None
        return value, {
            "valid": False,
            "error": [
                f"{prefix}: {language} requires a literal float{suffix}"
                f" (e.g. {float(value)!r}), got int {value!r}"
            ],
        }
    if json_type == "integer" and isinstance(value, float):
        return value, {
            "valid": False,
            "error": [f"{prefix}: expected integer{suffix}, got float {value!r}"],
        }
    return value, None


def _check_array_element_types(
    param: str, elements: list[Any], param_schema: dict[str, Any], language: str,
) -> tuple[list[Any], dict[str, Any] | None]:
    """One-level-deep element typecheck for array params (matches official depth)."""
    items_type_str = (param_schema.get("items") or {}).get("type")
    if items_type_str is None:
        return elements, None
    try:
        element_json_type = get_type(items_type_str)
    except ValueError:
        return elements, None

    coerced: list[Any] = []
    for element in elements:
        coerced_element, err = _apply_numeric_type_rules(
            element, element_json_type, language, param, nested=True
        )
        if err is not None:
            return [], err
        coerced.append(coerced_element)
    return coerced, None


# ---------------------------------------------------------------------------
# Single-call matcher — ports scorer.py::tool_call_matches_possible_answers
# ---------------------------------------------------------------------------


def tool_call_matches_possible_answers(
    actual: dict[str, Any],
    expected: dict[str, dict[str, list[Any]]],
    func_description: dict[str, Any] | None = None,
    language: str = "python",
) -> dict[str, Any]:
    """Does ``actual`` (a parsed call) satisfy one ground-truth entry?

    ``expected`` is ``{"func_name": {"param": [possible_val, ...]}}``. A param
    whose possible-values list contains ``""`` is optional. ``func_description``
    (the tool schema) supplies required-param + type info when present; its
    absence falls back to the ``""``-in-allowed-values convention.
    """
    func_name = next(iter(expected))
    expected_params = expected[func_name]

    param_schemas: dict[str, Any] = {}
    required_params: set[str] = set()
    if func_description:
        params_block = func_description.get("parameters", {}) or {}
        param_schemas = params_block.get("properties", {}) or {}
        required_params = set(params_block.get("required", []) or [])

    actual_args = actual["arguments"]

    # Function name.
    if normalize_function_name(actual["function"]) != normalize_function_name(func_name):
        return {"valid": False, "error": [
            f"Function name mismatch: expected {func_name}, got {actual['function']}"
        ]}

    # Required params present.
    for param in required_params:
        if param not in actual_args:
            return {"valid": False, "error": [f"Missing required parameter: {param}"]}

    # No unexpected params.
    for param in actual_args:
        if param not in expected_params:
            return {"valid": False, "error": [f"Unexpected parameter: {param}"]}

    # Each expected param.
    for param, allowed_values in expected_params.items():
        if param not in actual_args:
            # Optional iff allowed values include "" (and it's not required).
            if param not in required_params and "" in allowed_values:
                continue
            if not func_description and "" in allowed_values:
                continue
            return {"valid": False, "error": [f"Missing parameter: {param}"]}

        model_value = actual_args[param]

        # Type validation + coercion when the schema is known.
        if param in param_schemas:
            try:
                param_json_type = get_type(param_schemas[param].get("type"))
            except ValueError as e:
                raise ValueError(
                    f"Unknown type for parameter '{param}' in function '{func_name}': {e}"
                ) from e
            model_value, err = _apply_numeric_type_rules(
                model_value, param_json_type, language, param
            )
            if err is not None:
                return err
            if param_json_type == "array" and isinstance(model_value, tuple):
                model_value = list(model_value)
            if param_json_type == "array" and isinstance(model_value, list):
                coerced, err = _check_array_element_types(
                    param, model_value, param_schemas[param], language
                )
                if err is not None:
                    return err
                model_value = coerced

        if not _value_matches(model_value, allowed_values):
            return {"valid": False, "error": [
                f"Invalid value for {param}: {model_value!r}. Expected one of {allowed_values}"
            ]}

    return {"valid": True, "error": []}


# ---------------------------------------------------------------------------
# Multi-call matchers — port scorer.py::_match_parallel / _match_multiple
# ---------------------------------------------------------------------------


def _match_parallel(
    tool_calls: list[dict[str, Any]],
    possible_answers: list[dict[str, Any]],
    func_descriptions: list[dict[str, Any]] | None,
    language: str,
) -> dict[str, Any]:
    """All expected calls must be present; order is free (greedy bipartite match)."""
    if len(tool_calls) != len(possible_answers):
        return {"valid": False, "error": [
            f"Wrong number of function calls: expected {len(possible_answers)}, "
            f"got {len(tool_calls)}"
        ]}

    func_desc_lookup: dict[str, dict[str, Any]] = {}
    for fd in func_descriptions or []:
        func_desc_lookup[normalize_function_name(fd.get("name", ""))] = fd

    used_answers: set[int] = set()
    for tc in tool_calls:
        matched = False
        func_desc = func_desc_lookup.get(normalize_function_name(tc["function"]))
        for i, expected in enumerate(possible_answers):
            if i in used_answers:
                continue
            if tool_call_matches_possible_answers(tc, expected, func_desc, language)["valid"]:
                used_answers.add(i)
                matched = True
                break
        if not matched:
            return {"valid": False, "error": [f"No match found for: {tc['function']}"]}
    return {"valid": True, "error": []}


def _match_multiple(
    tool_calls: list[dict[str, Any]],
    possible_answers: list[dict[str, Any]],
    func_descriptions: list[dict[str, Any]] | None,
    language: str,
) -> dict[str, Any]:
    """Pick-the-right-tool: exactly one call, one answer."""
    if len(tool_calls) != 1 or len(possible_answers) != 1:
        return {"valid": False, "error": [
            f"Expected 1 function call and 1 answer, got {len(tool_calls)} "
            f"and {len(possible_answers)}"
        ]}
    func_desc = None
    if func_descriptions:
        expected_func_name = normalize_function_name(next(iter(possible_answers[0])))
        for fd in func_descriptions:
            if normalize_function_name(fd.get("name", "")) == expected_func_name:
                func_desc = fd
                break
    return tool_call_matches_possible_answers(
        tool_calls[0], possible_answers[0], func_desc, language
    )


# ---------------------------------------------------------------------------
# Category dispatch — ports task_categories.py::matching_function
# ---------------------------------------------------------------------------


def matching_function(category: str) -> str:
    """Scorer name for a raw BFCL category (mirrors ``CategoryConfig``)."""
    if "irrelevance" in category:
        return "irrelevance"
    if "relevance" in category:
        return "relevance"
    if "parallel" in category:
        return "parallel"
    if "multiple" in category:
        return "multiple"
    return "simple"


# ---------------------------------------------------------------------------
# The registered verifier fn
# ---------------------------------------------------------------------------


def bfcl_match(completion: str, expected: Any, params: dict) -> float:
    """1.0 iff the completion's tool calls satisfy the BFCL ground truth.

    Dispatches by ``expected['category']``. ``params`` is unused (the spec is
    fully self-describing) but kept for the ``fn(completion, expected, params)``
    contract the harbor ``EvsysVerifier`` calls us with.
    """
    if not isinstance(expected, dict):
        return 0.0
    category = str(expected.get("category", ""))
    tool_calls = parse_tool_calls(completion)
    fn = matching_function(category)

    # Abstention categories: only call *presence* matters.
    if fn == "irrelevance":
        return 1.0 if not tool_calls else 0.0
    if fn == "relevance":
        return 1.0 if tool_calls else 0.0

    possible_answers = list(expected.get("ground_truth") or [])
    tools = list(expected.get("tools") or [])
    language = str(expected.get("language") or "python")

    try:
        if fn == "parallel":
            result = _match_parallel(tool_calls, possible_answers, tools, language)
        elif fn == "multiple":
            result = _match_multiple(tool_calls, possible_answers, tools, language)
        else:  # simple — exactly one call expected
            if len(tool_calls) != 1 or len(possible_answers) != 1:
                return 0.0
            func_desc = tools[0] if tools else None
            result = tool_call_matches_possible_answers(
                tool_calls[0], possible_answers[0], func_desc, language
            )
    except ValueError:
        # A malformed schema (unknown type) is a dataset bug, not a model error;
        # score 0.0 so one bad task can't crash a whole rollout sweep.
        return 0.0
    return 1.0 if result["valid"] else 0.0


def _register_into_sdk() -> None:
    """Register ``bfcl_match`` into the SDK in-process verifier-fn registry —
    BUT only if ``evsys_sdk`` is already loaded.

    Gating on ``"evsys_sdk" in sys.modules`` (rather than "is it importable") is
    what keeps the RAW harness SDK-free: the SDK harness / self_test import
    ``evsys_sdk`` BEFORE ``import verifier``, so the registry is present and we
    register. The RAW harness never imports ``evsys_sdk`` at all, so it's absent
    here and we skip — importing this module for the pure scorer never drags the
    SDK in. Re-registration is harmless if some path imported the SDK later and
    calls this again."""
    import sys

    if "evsys_sdk" not in sys.modules:
        return
    try:
        from evsys_sdk.verifiers import register_verifier_fn
    except Exception:
        return
    register_verifier_fn(FN_NAME, bfcl_match)


# Fire the registration on import (mirrors experiments/stv/stv_verifier_fns.py).
# No-op unless evsys_sdk is already imported (SDK harness), so the RAW harness
# importing this module for the pure scorer stays SDK-free.
_register_into_sdk()


__all__ = [
    "FN_NAME",
    "bfcl_match",
    "parse_tool_calls",
    "matching_function",
    "tool_call_matches_possible_answers",
    "normalize_function_name",
    "get_type",
]

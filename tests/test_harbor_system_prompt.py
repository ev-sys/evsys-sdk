"""Per-task ``HarborTask.system_prompt`` — serialization + adapter wiring.

The field lets one harbor job mix tasks that each need a distinct system message
(e.g. function-calling tasks whose tool schemas live in the system turn). harbor
passes the agent only the per-task instruction string, so the system prompt rides
inside ``instruction.md`` behind a sentinel; ``BasicLoopAgent`` splits it back
into ``(system, user)``. These are unit tests — no real harbor job is run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evsys_sdk.data_types import (
    HarborTask,
    InProcessVerifier,
    harbor_task_from_dict,
)

pytest.importorskip("harbor")  # harbor_engine imports harbor at module top
pytest.importorskip("tinker")
pytest.importorskip("torch")

from evsys_sdk.training import harbor_engine as he


def _task(system_prompt=None):
    return HarborTask(
        task_id="t0",
        instruction="the user query",
        verifier=InProcessVerifier(fn_name="exact_match", expected="42"),
        metadata={},
        system_prompt=system_prompt,
    )


# --- serialization ---------------------------------------------------------


def test_system_prompt_defaults_to_none():
    assert _task().system_prompt is None
    # plain dict (no system_prompt key) → None, backward compatible
    d = {
        "task_id": "t0",
        "instruction": "i",
        "verifier": {"kind": "in_process", "fn_name": "f", "expected": 1},
    }
    assert harbor_task_from_dict(d).system_prompt is None


def test_system_prompt_round_trips_through_from_dict():
    d = {
        "task_id": "t0",
        "instruction": "i",
        "verifier": {"kind": "in_process", "fn_name": "f", "expected": 1},
        "system_prompt": "you are a tool-calling assistant",
    }
    assert harbor_task_from_dict(d).system_prompt == "you are a tool-calling assistant"


# --- split helper (round-trips the sentinel) -------------------------------


def test_split_system_instruction_no_sentinel_is_identity():
    assert he.split_system_instruction("plain instruction") == (None, "plain instruction")


def test_split_system_instruction_recovers_system_and_user():
    sys, user = "SYS BLOCK", "USER QUERY"
    packed = f"{sys}{he._SYSTEM_PROMPT_SENTINEL}\n{user}"
    assert he.split_system_instruction(packed) == (sys, user)


# --- adapter writes the per-task system prompt into instruction.md ----------


def test_adapter_packs_system_prompt_into_instruction(tmp_path: Path):
    cfgs = he.HarborTaskAdapter([_task(system_prompt="SYS BLOCK")]).to_harbor(tmp_path)
    written = (Path(cfgs[0].path) / "instruction.md").read_text()
    # the agent's split recovers exactly the (system, user) it was built from
    assert he.split_system_instruction(written) == ("SYS BLOCK", "the user query")


def test_adapter_writes_instruction_verbatim_when_no_system_prompt(tmp_path: Path):
    cfgs = he.HarborTaskAdapter([_task(system_prompt=None)]).to_harbor(tmp_path)
    written = (Path(cfgs[0].path) / "instruction.md").read_text()
    assert written == "the user query"
    assert he._SYSTEM_PROMPT_SENTINEL not in written

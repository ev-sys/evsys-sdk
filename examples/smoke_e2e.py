"""End-to-end Tinker smoke test for the dev -> main release gate.

Runs SFT, RL, and SDFT for one step each on the hosted ``tinker`` backend and
asserts every arm reaches ``status == "completed"``. Configs are inline so the
test is self-contained (no dependency on the example files).

Requires ``TINKER_API_KEY``. Used by ``.github/workflows/smoke.yml``; runnable
locally with ``python examples/smoke_e2e.py``.
"""

from __future__ import annotations

import os
import sys
import tempfile

from evsys_sdk import (
    AlgorithmConfig,
    BackendConfig,
    DataConfig,
    Experiment,
    ExperimentConfig,
    ModelConfig,
    RunConfig,
    TransformSpec,
)

MODEL = "Qwen/Qwen3.5-4B"
COMMON = dict(max_steps=1, batch_size=1, lora_rank=1)


def sft_cfg(out: str) -> ExperimentConfig:
    return ExperimentConfig(
        name="smoke_sft", output_dir=out,
        run=RunConfig(
            name="sft",
            data=DataConfig(
                source_kind="in_memory",
                rows=[{"query": "save a contact", "tool_slug": "OUTLOOK_CREATE_CONTACT"}],
                transforms=[TransformSpec(kind="jsonl_to_chat", params={
                    "user_template": "Query: {query}",
                    "assistant_template": "<answer>{tool_slug}</answer>"})],
            ),
            model=ModelConfig(name=MODEL),
            backend=BackendConfig(kind="tinker"),
            algorithm=AlgorithmConfig(kind="sft", params={**COMMON}),
        ),
    )


def rl_cfg(out: str) -> ExperimentConfig:
    return ExperimentConfig(
        name="smoke_rl", output_dir=out,
        run=RunConfig(
            name="rl",
            data=DataConfig(source_kind="in_memory", rows=[
                {"task_id": "t0", "instruction": "What is 2 + 2? Put the answer in <answer></answer>.",
                 "verifier": {"kind": "in_process", "fn_name": "contains", "expected": "<answer>4</answer>"}},
            ]),
            model=ModelConfig(name=MODEL),
            backend=BackendConfig(kind="tinker"),
            algorithm=AlgorithmConfig(kind="rl", params={
                **COMMON, "num_samples": 2, "max_tokens": 64, "user_template": "{prompt}"}),
        ),
    )


def sdft_cfg(out: str) -> ExperimentConfig:
    return ExperimentConfig(
        name="smoke_sdft", output_dir=out,
        run=RunConfig(
            name="sdft",
            data=DataConfig(source_kind="in_memory", rows=[
                {"inputs": {"question": "What is the capital of France?"}, "expected": "Paris"},
            ]),
            model=ModelConfig(name=MODEL),
            backend=BackendConfig(kind="tinker"),
            algorithm=AlgorithmConfig(kind="sdft", params={
                **COMMON, "topk": 20, "max_tokens": 64, "user_template": "{question}"}),
        ),
    )


def main() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY not set — cannot run the Tinker smoke.")

    builders = {"sft": sft_cfg, "rl": rl_cfg, "sdft": sdft_cfg}
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        for name, build in builders.items():
            print(f"\n===== {name.upper()} =====", flush=True)
            try:
                result = Experiment(build(f"{tmp}/{name}")).run()
                arm = result.best_arm or (result.arms[0] if result.arms else None)
                status = result.status
                print(f"{name}: status={status} arm={(arm and arm.run_result.status)}")
                if status != "completed":
                    failures.append(f"{name} (status={status})")
            except Exception as exc:  # noqa: BLE001
                print(f"{name}: ERROR {exc!r}")
                failures.append(f"{name} (error)")

    if failures:
        sys.exit(f"\nSMOKE FAILED: {', '.join(failures)}")
    print("\nALL SMOKE PASSED (sft, rl, sdft)")


if __name__ == "__main__":
    main()

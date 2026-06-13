"""Tests for `evsys_sdk.training.evaluators`.

Two surfaces:
  1. `BenchmarkEvaluator.evaluate(sampler)` — wraps an async SamplingClient
     into a sync InferenceClient and runs `Benchmark.score(client)` in a
     thread. Tests use a fake sampler that returns canned completion ids;
     a fake tokenizer decodes them. The benchmark verifier reads the
     decoded string.
  2. `build_in_loop_evaluators(metadata)` — filters benchmark specs by
     `run_every > 0`, materializes each, returns one evaluator per entry.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from evsys_sdk.benchmark import Benchmark
from evsys_sdk.training.evaluators import (
    BenchmarkEvaluator,
    _AsyncToSyncSampler,
    build_in_loop_evaluators,
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """char-code encode / 1-char-per-id decode."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


@dataclass
class _CannedSampler:
    """Returns the decoded `canned` string for each sample_async call."""
    canned: str
    calls: list = None  # populated lazily

    async def sample_async(self, *, prompt, params, num_samples=1, **kw):
        if self.calls is None:
            self.calls = []
        self.calls.append({"prompt": prompt, "params": params})
        @dataclass
        class _Seq:
            tokens: list[int]
        @dataclass
        class _Resp:
            sequences: list
        return _Resp(sequences=[_Seq(tokens=[ord(c) for c in self.canned])])


def _make_bench(tmp_path, name, expected):
    """Tiny harbor benchmark with one task expecting `expected`."""
    root = tmp_path / name
    root.mkdir()
    (root / "tasks.jsonl").write_text(json.dumps({
        "task_id": "t1", "instruction": "Q",
        "verifier": {"kind": "in_process", "fn_name": "exact_match",
                     "expected": expected},
        "metadata": {},
    }) + "\n")
    (root / "metadata.yaml").write_text(yaml.safe_dump({"name": name}))
    return Benchmark.from_dir(root)


# ---------------------------------------------------------------------------
# _AsyncToSyncSampler — the sync adapter Benchmark.score consumes
# ---------------------------------------------------------------------------


def test_async_to_sync_sampler_routes_through_loop():
    """The sync .generate() call should reach the async sampler via the
    event loop and return the decoded completion."""
    sampler = _CannedSampler(canned="HELLO")
    tok = _StubTokenizer()

    async def _run():
        loop = asyncio.get_running_loop()
        client = _AsyncToSyncSampler(sampler, tok, loop)
        # Call sync .generate from a worker thread (Benchmark.score does this).
        out = await asyncio.to_thread(client.generate, prompt="Q",
                                      max_tokens=4, temperature=0.0)
        return out

    assert asyncio.run(_run()) == "HELLO"
    assert len(sampler.calls) == 1


# ---------------------------------------------------------------------------
# BenchmarkEvaluator.evaluate
# ---------------------------------------------------------------------------


def test_benchmark_evaluator_scores_benchmark(tmp_path: Path):
    bench = _make_bench(tmp_path, "val", "MATCH")
    ev = BenchmarkEvaluator(
        name="val", benchmark=bench, tokenizer=_StubTokenizer(),
        run_every=10,
    )
    sampler = _CannedSampler(canned="MATCH")
    metrics = asyncio.run(ev.evaluate(sampler))
    assert metrics["pass_rate"] == 1.0
    assert metrics["n_tasks"] == 1.0


def test_benchmark_evaluator_scores_mismatch(tmp_path: Path):
    bench = _make_bench(tmp_path, "val", "MATCH")
    ev = BenchmarkEvaluator(
        name="val", benchmark=bench, tokenizer=_StubTokenizer(),
    )
    sampler = _CannedSampler(canned="NOPE")
    metrics = asyncio.run(ev.evaluate(sampler))
    assert metrics["pass_rate"] == 0.0


# ---------------------------------------------------------------------------
# build_in_loop_evaluators
# ---------------------------------------------------------------------------


def test_returns_empty_when_no_metadata():
    assert build_in_loop_evaluators(None, tokenizer=_StubTokenizer()) == []
    assert build_in_loop_evaluators({}, tokenizer=_StubTokenizer()) == []
    assert build_in_loop_evaluators({"hypothesis": "x"}, tokenizer=_StubTokenizer()) == []


def test_skips_entries_without_run_every(tmp_path: Path):
    """An entry without run_every is post-training-only and shouldn't
    produce an in-loop evaluator."""
    bench_dir = tmp_path / "b"; bench_dir.mkdir()
    (bench_dir / "tasks.jsonl").write_text(json.dumps({
        "task_id": "t", "instruction": "q",
        "verifier": {"kind": "in_process", "fn_name": "exact_match", "expected": "A"},
        "metadata": {},
    }) + "\n")
    (bench_dir / "metadata.yaml").write_text(yaml.safe_dump({"name": "b"}))
    meta = {"benchmark": [
        {"name": "post", "path": str(bench_dir)},                   # no run_every
        {"name": "inloop", "path": str(bench_dir), "run_every": 50},
    ]}
    evals = build_in_loop_evaluators(meta, tokenizer=_StubTokenizer())
    assert [e.name for e in evals] == ["inloop"]
    assert evals[0].run_every == 50


def test_accepts_single_dict_form(tmp_path: Path):
    """Single-dict benchmark with run_every set still yields one evaluator."""
    bench_dir = tmp_path / "b"; bench_dir.mkdir()
    (bench_dir / "tasks.jsonl").write_text(json.dumps({
        "task_id": "t", "instruction": "q",
        "verifier": {"kind": "in_process", "fn_name": "exact_match", "expected": "A"},
        "metadata": {},
    }) + "\n")
    (bench_dir / "metadata.yaml").write_text(yaml.safe_dump({"name": "b"}))
    meta = {"benchmark": {"name": "v", "path": str(bench_dir), "run_every": 30}}
    evals = build_in_loop_evaluators(meta, tokenizer=_StubTokenizer())
    assert len(evals) == 1
    assert evals[0].run_every == 30


def test_passes_chat_template_and_scoring_knobs(tmp_path: Path):
    bench_dir = tmp_path / "b"; bench_dir.mkdir()
    (bench_dir / "tasks.jsonl").write_text(json.dumps({
        "task_id": "t", "instruction": "q",
        "verifier": {"kind": "in_process", "fn_name": "exact_match", "expected": "A"},
        "metadata": {},
    }) + "\n")
    (bench_dir / "metadata.yaml").write_text(yaml.safe_dump({"name": "b"}))
    meta = {"benchmark": [{
        "name": "v", "path": str(bench_dir), "run_every": 10,
        "max_tokens": 128, "temperature": 0.0, "breakdown_keys": ["toolkit"],
        "limit": 50,
        "chat_template": {"system_prompt": "S", "user_template": "Q: {prompt}"},
    }]}
    [ev] = build_in_loop_evaluators(meta, tokenizer=_StubTokenizer())
    assert ev.max_tokens == 128
    assert ev.breakdown_keys == ["toolkit"]
    assert ev.limit == 50
    assert ev.chat_template == {"system_prompt": "S", "user_template": "Q: {prompt}"}


# ---------------------------------------------------------------------------
# TrainingLoop integration — per-evaluator run_every
# ---------------------------------------------------------------------------


def test_loop_respects_per_evaluator_run_every(tmp_path: Path):
    """An evaluator with `run_every: 3` should fire at step 2, 5, 8;
    one with `run_every: 5` should fire at 4, 9. The loop checks each
    independently."""
    import tinker
    from evsys_sdk.training import MockBackend, TrainingLoop, TrainingBatch

    fired: list[tuple[str, int]] = []

    @dataclass
    class _Ev:
        name: str
        run_every: int = 0
        async def evaluate(self, sampler):
            fired.append((self.name, _Ev.last_step))
            return {"pass_rate": 1.0}

    @dataclass
    class _SB:
        async def build_batch(self, step_idx):
            _Ev.last_step = step_idx
            return TrainingBatch(data=[tinker.Datum(
                model_input=tinker.ModelInput.from_ints([1]),
                loss_fn_inputs={},
            )], loss_fn="cross_entropy")
        def step_metrics(self, *a, **k): return {}

    class _LS:
        def log_metrics(self, *a, **k): pass

    loop = TrainingLoop(
        backend=MockBackend(), step_builder=_SB(),
        log_store=_LS(), output_dir=tmp_path,
        adam_params=tinker.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8),
        save_every=100, eval_every=0,
        evaluators=[_Ev(name="fast", run_every=3), _Ev(name="slow", run_every=5)],
    )
    asyncio.run(loop.run(num_steps=10))
    fast_steps = [s for n, s in fired if n == "fast"]
    slow_steps = [s for n, s in fired if n == "slow"]
    # (step+1)%3==0 → steps 2, 5, 8
    assert fast_steps == [2, 5, 8]
    # (step+1)%5==0 → steps 4, 9
    assert slow_steps == [4, 9]


def test_loop_fallback_to_eval_every_when_evaluator_has_no_run_every(tmp_path: Path):
    """An evaluator without `run_every` (or with 0) inherits the loop's
    `eval_every`. Preserves PR #18 behaviour."""
    import tinker
    from evsys_sdk.training import MockBackend, TrainingLoop, TrainingBatch

    fired = []

    @dataclass
    class _Ev:
        name: str = "legacy"
        async def evaluate(self, sampler):
            fired.append(1)
            return {"pass_rate": 0.5}

    @dataclass
    class _SB:
        async def build_batch(self, step_idx):
            return TrainingBatch(data=[tinker.Datum(
                model_input=tinker.ModelInput.from_ints([1]),
                loss_fn_inputs={},
            )], loss_fn="cross_entropy")
        def step_metrics(self, *a, **k): return {}

    class _LS:
        def log_metrics(self, *a, **k): pass

    loop = TrainingLoop(
        backend=MockBackend(), step_builder=_SB(),
        log_store=_LS(), output_dir=tmp_path,
        adam_params=tinker.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8),
        save_every=100, eval_every=2,    # → fires at steps 1, 3, 5
        evaluators=[_Ev()],
    )
    asyncio.run(loop.run(num_steps=6))
    # 3 fires (steps 1, 3, 5)
    assert len(fired) == 3

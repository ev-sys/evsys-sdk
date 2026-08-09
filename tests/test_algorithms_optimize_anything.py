"""optimize_anything algorithm — engine dispatch, evaluator contract, omni
explore→continue composition. The gepa engine API is faked via the `_load_oa`
seam so the suite has no gepa/network dependency."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from evsys_sdk.algorithms.optimize_anything import OptimizeAnythingAlgorithm
from evsys_sdk.protocols import RunContext
from evsys_sdk.registry import get_algorithm

import evsys_sdk.algorithms.optimize_anything as oa_mod


@dataclass
class _MockBackend:
    name: str = "mock"

    def prepare(self, *, model, run_dir): return {}
    def teardown(self, handles): return None


@dataclass
class _MockDataStore:
    name: str = "in_memory"
    def read_jsonl(self, path): return []
    def write_jsonl(self, path, rows): pass
    def read_json(self, path): return None
    def write_json(self, path, value): pass
    def exists(self, path): return False
    def list(self, prefix): return []


def _make_ctx(tmp_path: Path, extras: dict | None = None) -> RunContext:
    return RunContext(
        run_id="test_run",
        output_dir=str(tmp_path),
        config=None,
        data_store=_MockDataStore(),                # type: ignore[arg-type]
        backend=_MockBackend(),                     # type: ignore[arg-type]
        extras=extras or {},
    )


@dataclass
class _FakeOA:
    """Stands in for gepa.optimize_anything: records calls, drives the evaluator."""

    calls: list[dict] = field(default_factory=list)
    best: str = "OPTIMIZED PROMPT"

    def OptimizeAnythingConfig(self, **kw):  # noqa: N802 — mirrors the real name
        return SimpleNamespace(**kw)

    def optimize_anything(self, seed, *, evaluator, dataset, objective, background, config):
        # Exercise the evaluator exactly like a real engine would.
        scores = [evaluator(seed, ex)[0] for ex in dataset]
        self.calls.append({"fn": "optimize_anything", "seed": seed, "engine": config.engine,
                           "max_evals": config.max_evals, "n_rows": len(dataset),
                           "rows": dataset})
        return SimpleNamespace(best_candidate=self.best, best_score=max([*scores, 0.5]))

    def optimize_best_of(self, seed, *, evaluator, dataset, objective, background, configs):
        self.calls.append({"fn": "optimize_best_of", "seed": seed,
                           "engines": [c.engine for c in configs],
                           "budgets": [c.max_evals for c in configs]})
        return SimpleNamespace(best_candidate="EXPLORED SEED", best_score=0.3)


EXAMPLES = [
    {"inputs": {"q": "a"}, "expected": "alpha"},
    {"inputs": {"q": "b"}, "expected": "beta"},
]


class TestConfigValidation:
    def test_registered(self):
        assert get_algorithm("optimize_anything") is OptimizeAnythingAlgorithm

    def test_unknown_engine_rejected(self):
        with pytest.raises(ValueError, match="unknown optimize_anything engine"):
            OptimizeAnythingAlgorithm(seed_prompt="x", engine="magic")

    def test_unbounded_run_rejected(self):
        with pytest.raises(ValueError, match="bounded"):
            OptimizeAnythingAlgorithm(seed_prompt="x", max_evals=None)

    def test_exactly_one_seed(self):
        with pytest.raises(ValueError, match="exactly one"):
            OptimizeAnythingAlgorithm(seed_prompt="x", seed_path="y.txt")
        with pytest.raises(ValueError, match="exactly one"):
            OptimizeAnythingAlgorithm()


class TestTrain:
    def _patch(self, monkeypatch) -> _FakeOA:
        fake = _FakeOA()
        monkeypatch.setattr(oa_mod, "_load_oa", lambda: fake)
        return fake

    def test_requires_examples(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        algo = OptimizeAnythingAlgorithm(seed_prompt="seed")
        result = algo.train(_make_ctx(tmp_path))
        assert result.status == "failed" and "prompt_examples" in (result.error or "")

    def test_single_engine_run(self, tmp_path, monkeypatch):
        fake = self._patch(monkeypatch)
        algo = OptimizeAnythingAlgorithm(seed_prompt="seed", engine="gepa", max_evals=7)
        result = algo.train(_make_ctx(tmp_path, extras={"prompt_examples": EXAMPLES}))
        assert result.status == "completed"
        assert result.extras["best_prompt"] == "OPTIMIZED PROMPT"
        (call,) = fake.calls
        assert call["engine"] == "gepa" and call["max_evals"] == 7 and call["n_rows"] == 2
        # artifact contract shared with gepa_prompt
        prompts = json.loads(Path(result.artifacts["final_prompt"]).read_text())
        assert prompts["system_prompt"] == "OPTIMIZED PROMPT"
        assert result.metrics["oa/eval_calls"] == 2.0  # fake engine evaluated each row once

    def test_examples_per_eval_samples_evenly(self, tmp_path, monkeypatch):
        """The cap must spread across the dataset, not take the head slice."""
        fake = self._patch(monkeypatch)
        rows = [{"inputs": {"q": str(i)}, "expected": str(i)} for i in range(100)]
        algo = OptimizeAnythingAlgorithm(seed_prompt="s", examples_per_eval=4)
        algo.train(_make_ctx(tmp_path, extras={"prompt_examples": rows}))
        assert fake.calls[0]["n_rows"] == 4
        picked = [ex["inputs"]["q"] for ex in fake.calls[0]["rows"]]
        assert picked == ["0", "25", "50", "75"]

    def test_seed_path_reads_file(self, tmp_path, monkeypatch):
        fake = self._patch(monkeypatch)
        p = tmp_path / "prompt.txt"
        p.write_text("from file\n")
        algo = OptimizeAnythingAlgorithm(seed_path=str(p))
        result = algo.train(_make_ctx(tmp_path, extras={"prompt_examples": EXAMPLES}))
        assert result.status == "completed"
        assert fake.calls[0]["seed"] == "from file"

    def test_engine_config_only_reaches_its_engine(self, tmp_path, monkeypatch):
        """best_of_n/autoresearch hard-reject gepa's reflection keys — the
        passthrough must only go to phases running the main engine."""
        fake = self._patch(monkeypatch)
        seen: dict[str, dict] = {}

        real_best_of = fake.optimize_best_of

        def capture(seed, *, configs, **kw):
            for c in configs:
                seen[c.engine] = c.engine_config
            return real_best_of(seed, configs=configs, **kw)

        fake.optimize_best_of = capture
        algo = OptimizeAnythingAlgorithm(
            seed_prompt="seed", engine="gepa",
            explore_engines=["gepa", "best_of_n"],
            engine_config={"reflection": {"reflection_lm": "anthropic/x"}},
        )
        algo.train(_make_ctx(tmp_path, extras={"prompt_examples": EXAMPLES}))
        assert seen["gepa"] == {"reflection": {"reflection_lm": "anthropic/x"}}
        assert seen["best_of_n"] == {}

    def test_omni_explore_then_continue(self, tmp_path, monkeypatch):
        fake = self._patch(monkeypatch)
        algo = OptimizeAnythingAlgorithm(
            seed_prompt="seed", engine="gepa",
            explore_engines=["gepa", "best_of_n"], explore_max_evals=3, max_evals=9,
        )
        result = algo.train(_make_ctx(tmp_path, extras={"prompt_examples": EXAMPLES}))
        assert result.status == "completed"
        explore, main = fake.calls
        assert explore["fn"] == "optimize_best_of"
        assert explore["engines"] == ["gepa", "best_of_n"] and explore["budgets"] == [3, 3]
        # phase 2 continues from the explore winner, not the original seed
        assert main["seed"] == "EXPLORED SEED"
        assert [p["phase"] for p in result.extras["phases"]] == ["explore", "main"]
        summary = json.loads((tmp_path / "oa_summary.json").read_text())
        assert summary["engine"] == "gepa" and len(summary["phases"]) == 2

    def test_score_fn_tuple_feedback(self, tmp_path, monkeypatch):
        """(score, feedback) score fns flow feedback into the engine's info dict."""
        infos: list[dict] = []
        fake = self._patch(monkeypatch)

        real_oa = fake.optimize_anything

        def capture(seed, *, evaluator, dataset, **kw):
            for ex in dataset:
                score, info = evaluator(seed, ex)
                infos.append(info)
            return real_oa(seed, evaluator=lambda c, e: (0.0, {}), dataset=[], **kw)

        fake.optimize_anything = capture
        algo = OptimizeAnythingAlgorithm(seed_prompt="seed")
        ctx = _make_ctx(tmp_path, extras={
            "prompt_examples": EXAMPLES,
            "prompt_score_fn": lambda completion, expected: (0.5, f"needs more {expected}"),
        })
        assert algo.train(ctx).status == "completed"
        assert infos and infos[0]["Feedback"] == "needs more alpha"
        assert "Generated" in infos[0]

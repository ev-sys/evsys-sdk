"""03 — campaign of N runs from a single matrix.

One `matrix:` block expands into a list of runs by taking the cartesian
product of the axes. Useful for hyperparameter sweeps; the canonical input
to evolutionary algorithms.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from trajectory_labs import load_yaml, run_experiment

HERE = Path(__file__).parent

CFG = {
    "name": "example_03_matrix",
    "output_dir": str(HERE / "outputs" / "03"),
    "log_store": {"kind": "jsonl"},
    "matrix": {
        "base_run": {
            "name": "base",
            "data": {
                "source_kind": "in_memory",
                "rows": [{"query": "x", "tool_slug": "FAKE_TOOL", "toolkit": "FAKE", "description": "..."}],
                "transforms": [{"kind": "jsonl_to_chat", "params": {"user_template": "Query: {query}", "assistant_template": "<answer>{tool_slug}</answer>"}}],
            },
            "model": {"name": "tiny/fake"},
            "algorithm": {
                "kind": "mock_sft",
                "params": {"num_epochs": 1, "batch_size": 1},
            },
            "backend": {"kind": "mock"},
            "eval": {"enabled": False},
        },
        "axes": {
            "algorithm.params.num_epochs": [1, 2, 3],
            "algorithm.params.lora_rank": [4, 8],
        },
        "name_template": "sweep__e{algorithm.params.num_epochs}__r{algorithm.params.lora_rank}",
    },
}


def main():
    cfg = load_yaml(CFG)
    print(f"Matrix expanded into {len(cfg.runs)} runs:")
    for r in cfg.runs:
        print(f"  - {r.name}")

    results = run_experiment(cfg)
    for r in results:
        print(f"  {r.run_id}: {r.status} loss={r.metrics.get('train/final_loss', 0):.3f}")


if __name__ == "__main__":
    main()

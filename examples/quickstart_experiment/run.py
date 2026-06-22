"""Quickstart experiment — run an experiment defined in YAML.

Run the mock wiring test (default), or real local training:

    python examples/quickstart_experiment/run.py                    # config.yaml      (mock)
    python examples/quickstart_experiment/run.py config_local.yaml  # config_local.yaml (real local SFT)

The config is the canonical surface; this runner just loads it and hands it to
the runner. You could equally run it from the CLI:

    evsys run examples/quickstart_experiment/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

from evsys_sdk import load_yaml, run_experiment

HERE = Path(__file__).parent


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_yaml(HERE / config_name)
    # Keep artifacts beside this experiment, regardless of where you run from.
    cfg.output_dir = str(HERE / "outputs")

    [result] = run_experiment(cfg)

    print(f"Status: {result.status}")
    print(f"Artifacts: {list(result.artifacts.keys())}")
    print(f"Metrics: {result.metrics}")


if __name__ == "__main__":
    main()

"""Generic runner for the example configs — load a YAML and run it.

    python examples/configs/run.py examples/configs/sft.yaml
    python examples/configs/run.py examples/configs/rl.yaml
    python examples/configs/run.py examples/configs/sdft.yaml

The config is the canonical surface; this runner just loads it and hands it to
``run_experiment``. You could equally run it from the CLI:

    evsys run examples/configs/sft.yaml

These three configs all use ``backend: {kind: tinker}`` — real training needs
``TINKER_API_KEY`` set and charges a small amount against your Tinker quota.
To check a config without training, validate it first:

    evsys validate examples/configs/sft.yaml --deep
"""

from __future__ import annotations

import sys
from pathlib import Path

from evsys_sdk import load_yaml, run_experiment


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(f"usage: python {Path(__file__).name} <config.yaml>")

    cfg = load_yaml(sys.argv[1])

    [result] = run_experiment(cfg)

    print(f"Status: {result.status}")
    print(f"Artifacts: {list(result.artifacts.keys())}")
    print(f"Metrics: {result.metrics}")


if __name__ == "__main__":
    main()

"""SFT walkthrough runner — loads config.yaml and runs the experiment.

The config is the canonical surface; this runner just loads it, points the
output dir beside this file, and hands it to ``run_experiment``. You could
equally run it from the CLI:

    evsys run examples/sft_walkthrough/config.yaml

Run from the REPO ROOT so the relative data paths in config.yaml
(``examples/sft_walkthrough/data/...``) resolve against the working directory:

    .venv/bin/python examples/sft_walkthrough/run.py

Real training needs TINKER_API_KEY and charges a small amount. To only check
the wiring with no GPU / network / cost, validate instead:

    .venv/bin/evsys validate examples/sft_walkthrough/config.yaml --deep
"""

from __future__ import annotations

from pathlib import Path

from evsys_sdk import Experiment, load_yaml

HERE = Path(__file__).parent


def main() -> None:
    cfg = load_yaml(HERE / "config.yaml")
    # Keep artifacts beside this experiment, regardless of where you run from.
    cfg.output_dir = str(HERE / "outputs")

    # Use Experiment (not run_experiment) so the experiment-scope callbacks —
    # e.g. local_logger writing metrics.jsonl / summary.md / experiment.md — fire.
    result = Experiment(cfg).run()
    arm = result.best_arm or (result.arms[0] if result.arms else None)

    print(f"Status:     {result.status}")
    if arm is not None:
        print(f"Metrics:    {arm.run_result.metrics}")
        print(f"Artifacts:  {list(arm.run_result.artifacts.keys())}")
    print(f"Conclusion: {result.conclusion}")
    print(f"Logs:       {HERE / 'outputs'} — metrics.jsonl, predictions/, summary.md, experiment.md")


if __name__ == "__main__":
    main()

"""RL walkthrough runner — loads config.yaml and runs the experiment.

Run from the REPO ROOT so the relative data paths resolve:

    .venv/bin/python examples/rl_walkthrough/run.py

Real training needs TINKER_API_KEY and charges a small amount. To check the
wiring with no GPU / network / cost, validate instead:

    .venv/bin/evsys validate examples/rl_walkthrough/config.yaml --deep
"""

from __future__ import annotations

from pathlib import Path

from evsys_sdk import Experiment, load_yaml

HERE = Path(__file__).parent


def main() -> None:
    cfg = load_yaml(HERE / "config.yaml")
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

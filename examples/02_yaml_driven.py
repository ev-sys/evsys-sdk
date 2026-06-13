"""02 — drive the same experiment from a YAML file.

The library's API is YAML-canonical: any experiment expressible in Python is
also expressible as a YAML file. This means external evolutionary algorithms
can mutate the YAML directly without needing to know any Python.
"""

from __future__ import annotations

from pathlib import Path

from evsys_sdk import dump_yaml, load_yaml, run_experiment

HERE = Path(__file__).parent


YAML = """
name: example_02_yaml_driven
output_dir: ./examples/outputs/02
log_store:
  kind: jsonl
run:
  name: yaml_run
  data:
    source_kind: in_memory
    rows:
      - {query: "save a contact", tool_slug: OUTLOOK_CREATE_CONTACT, toolkit: OUTLOOK, description: "..."}
      - {query: "edit a slack message", tool_slug: SLACK_UPDATES_A_SLACK_MESSAGE, toolkit: SLACK, description: "..."}
    transforms:
      - kind: jsonl_to_chat
        params: {user_template: "Query: {query}", assistant_template: "<answer>{tool_slug}</answer>"}
  model:
    name: tiny/fake
  algorithm:
    kind: mock_sft
    params:
      num_epochs: 1
      batch_size: 1
  backend:
    kind: mock
  eval:
    enabled: false
"""


def main():
    yaml_path = HERE / "outputs" / "02" / "example.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(YAML)

    cfg = load_yaml(yaml_path)
    print("Parsed config:")
    print(dump_yaml(cfg)[:500])

    [result] = run_experiment(cfg)
    print(f"Status: {result.status}")


if __name__ == "__main__":
    main()

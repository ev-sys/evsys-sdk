
from pathlib import Path

def test_output_dir_defaults_to_experiment_dir_evsys_timestamp(tmp_path):
    """Unset output_dir → <experiment_dir>/.evsys/<YYYY-MM-DD_HH-MM-SS>
    (next to the config, not a repo-root .evsys/)."""
    exp = tmp_path / "experiments" / "my_exp"
    exp.mkdir(parents=True)
    (exp / "config.yaml").write_text(
        "name: my_exp\n"
        "run:\n"
        "  name: r\n"
        "  data: {source_kind: in_memory, rows: [{q: Q}]}\n"
        "  model: {name: m}\n"
        "  algorithm: {kind: sft}\n"
        "  backend: {kind: mock}\n"
    )
    from evsys_sdk.yaml_loader import load_yaml
    cfg = load_yaml(exp / "config.yaml")
    out = Path(cfg.output_dir)
    assert out.parent == exp / ".evsys"            # under the experiment dir's .evsys/
    assert out.name[:4].isdigit() and "_" in out.name   # a timestamp leaf


def test_explicit_output_dir_is_respected(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "name: x\noutput_dir: ./custom/out\n"
        "run:\n  name: r\n  data: {source_kind: in_memory, rows: [{q: Q}]}\n"
        "  model: {name: m}\n  algorithm: {kind: sft}\n  backend: {kind: mock}\n"
    )
    from evsys_sdk.yaml_loader import load_yaml
    assert load_yaml(tmp_path / "config.yaml").output_dir == "./custom/out"

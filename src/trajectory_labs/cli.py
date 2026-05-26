"""trajex CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_validate(args: argparse.Namespace) -> int:
    from .yaml_loader import validate_yaml

    errors = validate_yaml(args.path, deep=args.deep)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"OK: {args.path} parses and validates.")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .runner import run_experiment

    results = run_experiment(args.path)
    summary = []
    for r in results:
        summary.append(
            {
                "run_id": r.run_id,
                "status": r.status,
                "metrics": r.metrics,
                "error": r.error,
            }
        )
    out = json.dumps(summary, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(out)
    print(out)
    return 0 if all(r.status == "completed" for r in results) else 2


def _cmd_list(args: argparse.Namespace) -> int:
    from .registry import (
        list_algorithms,
        list_backends,
        list_data_stores,
        list_inferences,
        list_log_stores,
        list_metrics,
        list_transforms,
        list_verifiers,
    )

    for kind, fn in [
        ("algorithms", list_algorithms),
        ("backends", list_backends),
        ("verifiers", list_verifiers),
        ("metrics", list_metrics),
        ("transforms", list_transforms),
        ("data_stores", list_data_stores),
        ("log_stores", list_log_stores),
        ("inference", list_inferences),
    ]:
        items = fn()
        if args.kind and args.kind != kind:
            continue
        print(f"{kind}:")
        for item in items:
            print(f"  - {item}")
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    from .registry import schema_for

    schema = schema_for(args.kind, args.name)
    print(json.dumps(schema, indent=2))
    return 0


def _cmd_eval_model(args: argparse.Namespace) -> int:
    from .eval import ModelEvalConfig, evaluate_model, format_summary_markdown
    from .registry import get_inference

    inf_cls = get_inference(args.inference_kind)
    client_kwargs: dict = {"model_name": args.model_name}
    if args.adapter_path:
        client_kwargs["adapter_path"] = args.adapter_path
    if args.checkpoint_path:
        client_kwargs["checkpoint_path"] = args.checkpoint_path
    client = inf_cls(**client_kwargs)

    cfg = ModelEvalConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_attempts=args.max_attempts,
        batch_size=args.batch_size,
    )
    artifacts = evaluate_model(
        dataset_path=args.dataset,
        aliases_path=args.aliases,
        secondary_aliases_path=args.secondary_aliases,
        client=client,
        config=cfg,
        output_dir=args.output_dir,
    )
    print(format_summary_markdown(artifacts.summary, title=f"Model eval ({args.inference_kind})"))
    rr = artifacts.summary.retry_report
    return 0 if rr.get("total_failures", 0) == 0 or not args.fail_on_retries else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trajex", description="Trajectory experiments CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Validate a YAML experiment file.")
    p_val.add_argument("path")
    p_val.add_argument("--deep", action="store_true", help="Also validate kind/params blocks against registry schemas.")
    p_val.set_defaults(func=_cmd_validate)

    p_run = sub.add_parser("run", help="Run an experiment.")
    p_run.add_argument("path")
    p_run.add_argument("--output", "-o", default=None, help="Where to write the run summary JSON.")
    p_run.set_defaults(func=_cmd_run)

    p_list = sub.add_parser("list", help="List registered extensions.")
    p_list.add_argument("--kind", default=None, help="Filter by registry kind (e.g. 'algorithms').")
    p_list.set_defaults(func=_cmd_list)

    p_sch = sub.add_parser("schema", help="Print JSON schema for a registered extension.")
    p_sch.add_argument("kind", help="One of: algorithm, backend, verifier, metric, transform, data_store, log_store, inference_client")
    p_sch.add_argument("name")
    p_sch.set_defaults(func=_cmd_schema)

    p_eval = sub.add_parser("eval", help="Run a model eval (alias-aware, retry-wrapped).")
    eval_sub = p_eval.add_subparsers(dest="eval_cmd", required=True)

    p_em = eval_sub.add_parser("model", help="Evaluate a model checkpoint over the eval set.")
    p_em.add_argument("--dataset", required=True)
    p_em.add_argument("--aliases", required=True)
    p_em.add_argument("--secondary-aliases", default=None)
    p_em.add_argument("--output-dir", required=True)
    p_em.add_argument("--inference-kind", required=True, choices=["local", "tinker", "mock"])
    p_em.add_argument("--model-name", required=True)
    p_em.add_argument("--adapter-path", default=None, help="Local PEFT adapter dir (for --inference-kind=local).")
    p_em.add_argument("--checkpoint-path", default=None, help="Tinker checkpoint path (for --inference-kind=tinker).")
    p_em.add_argument("--max-tokens", type=int, default=256)
    p_em.add_argument("--temperature", type=float, default=0.0)
    p_em.add_argument("--max-attempts", type=int, default=5)
    p_em.add_argument("--batch-size", type=int, default=1, help="Submit prompts in chunks of this size (needs generate_batch on the inference client).")
    p_em.add_argument("--fail-on-retries", action="store_true")
    p_em.set_defaults(func=_cmd_eval_model)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""evsys CLI."""

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


def _cmd_init_project(args: argparse.Namespace) -> int:
    from .project_init import init_project

    try:
        path = init_project(args.path, name=args.name, force=args.force)
    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"OK: scaffolded research project at {path}")
    return 0


def _cmd_benchmark_upload(args: argparse.Namespace) -> int:
    from .benchmark_upload import upload_benchmark
    from .store import EvsysStore

    store = EvsysStore(project_id=args.project_id)
    try:
        result = upload_benchmark(store, args.path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    payload = {
        "benchmark_id": result.benchmark_id,
        "name": result.name,
        "version": result.version,
        "content_hash": result.content_hash,
        "status": result.status,
        "n_tasks": result.n_tasks,
    }
    print(json.dumps(payload, indent=2))
    print(f"\n# paste into your experiment config.yaml:")
    print(f"# metadata.benchmark.id: {result.benchmark_id}")
    return 0


def _cmd_benchmark_run(args: argparse.Namespace) -> int:
    """Score a local benchmark on a closed / API model (no training)."""
    from .benchmark import Benchmark
    from .benchmark_run import run_benchmark

    try:
        bench = Benchmark.from_dir(args.path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    metrics = run_benchmark(
        bench,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        limit=args.limit,
    )
    print(json.dumps({"model": args.model, "benchmark": bench.name, "metrics": metrics}, indent=2))
    return 0


def _cmd_new_experiment(args: argparse.Namespace) -> int:
    from datetime import datetime

    from .new_experiment import new_experiment

    today = None
    if args.date:
        try:
            today = datetime.strptime(args.date, "%Y%m%d").date()
        except ValueError:
            print(f"ERROR: --date {args.date!r} is not in YYYYMMDD format", file=sys.stderr)
            return 1
    try:
        path = new_experiment(args.project_root, args.slug, today=today)
    except (FileExistsError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"OK: scaffolded experiment at {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evsys", description="EvolvingSystems experiments CLI.")
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

    p_init = sub.add_parser("init-project", help="Scaffold a new research-project layout.")
    p_init.add_argument("path", help="Target directory (created if missing).")
    p_init.add_argument("--name", default=None, help="Project name (defaults to dir basename).")
    p_init.add_argument("--force", action="store_true", help="Scaffold into a non-empty dir.")
    p_init.set_defaults(func=_cmd_init_project)

    p_bench = sub.add_parser("benchmark", help="Manage harbor-format benchmarks.")
    bench_sub = p_bench.add_subparsers(dest="bench_cmd", required=True)
    p_bup = bench_sub.add_parser("upload", help="Register a local benchmark dir with the dashboard.")
    p_bup.add_argument("path", help="Path to data/benchmark/<name>/.")
    p_bup.add_argument("--project-id", default=None, help="Override EVSYS_PROJECT_ID.")
    p_bup.set_defaults(func=_cmd_benchmark_upload)

    p_brun = bench_sub.add_parser(
        "run", help="Score a local benchmark on a closed/API model (no training)."
    )
    p_brun.add_argument("path", help="Path to data/benchmark/<name>/.")
    p_brun.add_argument(
        "--model", required=True,
        help="litellm model id, e.g. anthropic/claude-opus-4-1 or openai/gpt-4o.",
    )
    p_brun.add_argument("--num-samples", type=int, default=1)
    p_brun.add_argument("--max-tokens", type=int, default=512)
    p_brun.add_argument("--temperature", type=float, default=0.0)
    p_brun.add_argument("--limit", type=int, default=None, help="Cap on tasks scored.")
    p_brun.set_defaults(func=_cmd_benchmark_run)

    p_new = sub.add_parser("new-experiment", help="Create experiments/<YYYYMMDD>_<slug>/{config.yaml,run.py}.")
    p_new.add_argument("slug", help="Short kebab/snake name for the experiment.")
    p_new.add_argument("--project-root", default=".", help="Repo root (default: current dir).")
    p_new.add_argument("--date", default=None, help="Override the date prefix (YYYYMMDD).")
    p_new.set_defaults(func=_cmd_new_experiment)

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

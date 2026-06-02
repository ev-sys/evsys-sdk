"""09 — Composio doc-based tool retrieval: generate embeddings + score pass@k.

End-to-end demo of the doc-based tool-discovery approach:

  1. Load (query, tool_slug, tool-doc) rows produced by composio-bench's
     export_training_data.py (the composio_doc_pairs format).
  2. Build a tool catalog and EMBED every tool doc into a vector.
  3. For each held-out query, embed it and retrieve the top-k nearest tools.
  4. Score retrieval quality with the SDK's pass_at_k_retrieval metric.

Two embedding backends:
  --backend hashing   (default) zero-dependency numpy bag-of-words baseline
  --backend st        sentence-transformers semantic embeddings
                      (pip install trajectory-labs[embedding])

Run
---
    cd trajectory-labs-sdk
    python examples/09_composio_embedding_retrieval.py
    python examples/09_composio_embedding_retrieval.py --backend st
    python examples/09_composio_embedding_retrieval.py --data /path/to/doc_pairs.jsonl

The default --data path points at the sibling composio-bench checkout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_labs import get_inference, get_metric, get_transform

HERE = Path(__file__).parent
DEFAULT_DATA = HERE.parent.parent / "composio-bench" / "data" / "training" / "doc_pairs.jsonl"


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Data file not found: {path}\n"
            "Generate it first:  cd composio-bench && "
            "python scripts/export_training_data.py"
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_client(backend: str, model: str, dim: int, corpus: list[dict]):
    return get_inference("embedding_retrieval")(
        backend="sentence_transformers" if backend == "st" else "hashing",
        model_name=model,
        dim=dim,
        corpus_rows=corpus,      # doc text comes from each row's `positive` field
        text_field="positive",
        id_field="tool_slug",
        top_k=10,
    )


def score(client, eval_rows: list[dict], label: str) -> None:
    preds = [{"candidates": client.retrieve(r["query"], top_k=10)} for r in eval_rows]
    tgts = [{"tool_slug": r["tool_slug"]} for r in eval_rows]
    print(f"\n{label}")
    for k in (1, 3, 5, 10):
        s = get_metric("pass_at_k_retrieval")(k=k).compute(predictions=preds, targets=tgts)
        bar = "#" * int(s * 40)
        print(f"  pass@{k:<2} = {s:.3f}  {bar}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="doc_pairs.jsonl path")
    ap.add_argument("--backend", choices=["hashing", "st"], default="hashing")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="sentence-transformers model (used with --backend st)")
    ap.add_argument("--dim", type=int, default=512, help="hashing embedder dimension")
    ap.add_argument("--finetune", action="store_true",
                    help="hold out 20%% of queries, fine-tune with embedding_sft, "
                         "and compare base vs tuned retrieval (forces --backend st)")
    ap.add_argument("--epochs", type=int, default=3, help="fine-tune epochs")
    args = ap.parse_args()

    rows = load_rows(Path(args.data))
    # Materialize (anchor, positive) doc text from the generic export rows.
    rows = list(get_transform("composio_doc_pairs")()(rows))
    print(f"Loaded {len(rows)} (query, tool, doc) rows from {args.data}")

    # Corpus = every unique tool doc (always retrievable).
    corpus = list({r["tool_slug"]: r for r in rows}.values())
    print(f"Tool catalog: {len(corpus)} unique tools")

    if not args.finetune:
        # --- Single-model retrieval demo (retrieval over all query rows) ---
        print(f"Embedding backend: {args.backend}")
        print("\nGenerating embeddings for all tool docs...")
        client = build_client(args.backend, args.model, args.dim, corpus)
        print(f"Embedded {client.doc_matrix.shape[0]} tool docs "
              f"into {client.doc_matrix.shape[1]}-dim vectors")
        score(client, rows, "Retrieval quality (pass@k — correct tool within top-k):")

        print("\nSample retrievals (top-3):")
        for r in rows[:5]:
            top3 = client.retrieve(r["query"], top_k=3)
            hit = "OK " if r["tool_slug"] in top3 else "MISS"
            print(f"  [{hit}] {r['query'][:55]!r}")
            print(f"         want: {r['tool_slug']}")
            print(f"         got:  {top3}")
        return

    # --- Held-out before/after fine-tuning comparison (semantic backend) ---
    import random
    import tempfile
    from trajectory_labs import (
        ExperimentConfig, RunConfig, DataConfig, ModelConfig,
        AlgorithmConfig, BackendConfig, EvalConfig, run_experiment,
    )

    random.seed(42)
    random.shuffle(rows)
    n_test = int(len(rows) * 0.2)
    test, train = rows[:n_test], rows[n_test:]
    print(f"Held-out split: {len(train)} train pairs | {len(test)} test pairs")

    base = build_client("st", args.model, args.dim, corpus)
    score(base, test, "BASE (pretrained, no fine-tuning) — held-out test:")

    print(f"\nFine-tuning with embedding_sft ({args.epochs} epochs)...")
    outdir = tempfile.mkdtemp()
    cfg = ExperimentConfig(
        name="ft", output_dir=outdir,
        run=RunConfig(
            name="run",
            data=DataConfig(source_kind="in_memory", rows=train),
            model=ModelConfig(name=args.model),
            algorithm=AlgorithmConfig(kind="embedding_sft", params={
                "base_model": args.model, "num_epochs": args.epochs, "batch_size": 16}),
            backend=BackendConfig(kind="mock"),
            eval=EvalConfig(enabled=False),
        ),
    )
    res = run_experiment(cfg)
    if res[0].status != "completed":
        raise SystemExit(f"fine-tune failed: {res[0].error}")

    tuned = build_client("st", res[0].artifacts["final_checkpoint"], args.dim, corpus)
    score(tuned, test, "TUNED (after embedding_sft on train pairs) — held-out test:")


if __name__ == "__main__":
    main()

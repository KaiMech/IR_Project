# scripts/retrievers/run_retrieve_anserini.py
# This is an example of how to create a run file for a model.
# Here, we have done this for the first baseline model.
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

# Make repo root importable so we can import utils_io
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils_io import (
    find_repo_root, get_queries_path, iter_queries_jsonl, write_trec_run
)

def main():
    ap = argparse.ArgumentParser(
        description="Create a TREC runfile (Top-k per query) using Anserini/Pyserini BM25."
    )
    ap.add_argument("--split", required=True, choices=["train","dev1","dev2","dev3","test"])
    ap.add_argument("--runid", required=True, help="Run ID (column 6 in TREC run)")
    ap.add_argument("--outdir", default="runs", help="Output directory for the runfile (relative to repo root)")
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--queries", default=None,
                    help="Optional path to *-queries.jsonl (otherwise we use data/tot25/)")
    ap.add_argument("--index", default=None,
                    help="Path to Anserini index (default: env ANSERINI_INDEX)")
    ap.add_argument("--k1", type=float, default=0.9, help="BM25 k1")
    ap.add_argument("--b",  type=float, default=0.4, help="BM25 b")
    args = ap.parse_args()

    index_dir = args.index or os.environ.get("ANSERINI_INDEX")
    if not index_dir or not os.path.isdir(index_dir):
        raise SystemExit(
            "Anserini index not found.\n"
            "Pass --index /path/to/index or set env var ANSERINI_INDEX."
        )

    try:
        # UPDATED: use the current Pyserini API
        from pyserini.search.lucene import LuceneSearcher
    except Exception as e:
        raise SystemExit(
            "Pyserini import failed (or Java missing).\n"
            "Install: pip install pyserini\n"
            "Ensure a recent Java (e.g., 17/21): `java -version`\n"
            "If you used an older API, replace SimpleSearcher with LuceneSearcher.\n"
            f"Details: {e}"
        )

    root = find_repo_root()
    outdir = (root / args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_path = outdir / f"{args.split}.anserini_bm25.run"
    if run_path.exists():
        run_path.unlink()

    # Queries
    qpath = Path(args.queries) if args.queries else get_queries_path(args.split, root)
    if not qpath.exists():
        raise SystemExit(f"Queries not found: {qpath}")

    # Searcher (LuceneSearcher is the modern equivalent of SimpleSearcher)
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=args.k1, b=args.b)

    # Retrieve & write
    n = 0
    for qid, query in iter_queries_jsonl(qpath):
        hits = searcher.search(query, k=args.topk)
        # hits -> [(docid, score)]
        hit_pairs = [(h.docid, float(h.score)) for h in hits]
        write_trec_run(run_path, qid, hit_pairs, runid=args.runid, k=args.topk)
        n += 1
        if n % 25 == 0:
            print(f"{n} queries...")

    print(f"OK: {n} queries -> {run_path}")

if __name__ == "__main__":
    main()

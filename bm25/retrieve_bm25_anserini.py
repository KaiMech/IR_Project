#!/usr/bin/env python3
# scripts/retrieve_bm25_anserini.py
import argparse, json, os, sys
from pathlib import Path

def iter_queries_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id") or obj.get("id") or "").strip()
            qtext = (obj.get("query") or obj.get("title") or obj.get("text") or "").strip()
            if not qid or not qtext:
                continue
            yield qid, qtext

def write_trec_run(run_path: Path, qid: str, pairs, runid: str):

    with open(run_path, "a", encoding="utf-8") as out:
        for rank, (docid, score) in enumerate(pairs, start=1):
            out.write(f"{qid} Q0 {docid} {rank} {score:.6f} {runid}\n")

def main():
    ap = argparse.ArgumentParser(
        description="Create a TREC runfile using Anserini/Pyserini BM25 (LuceneSearcher)."
    )
    ap.add_argument("--index_dir", required=True, help="Path to Lucene index directory")
    ap.add_argument("--queries_path", required=True, help="Path to *-queries-*.jsonl")
    ap.add_argument("--run_path", required=True, help="Output path to write the TREC runfile")
    ap.add_argument("--runid", default="bm25_anserini_k1_0.9_b_0.4", help="Column 6 in TREC run")
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b",  type=float, default=0.4)
    args = ap.parse_args()

    try:
        from pyserini.search.lucene import LuceneSearcher
    except Exception as e:
        sys.exit(f"Pyserini nicht verfügbar: {e}\nInstalliere: pip install pyserini")

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        sys.exit(f"Index nicht gefunden: {index_dir}")

    qpath = Path(args.queries_path)
    if not qpath.exists():
        sys.exit(f"Queries nicht gefunden: {qpath}")

    run_path = Path(args.run_path)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    if run_path.exists():
        run_path.unlink()

    searcher = LuceneSearcher(str(index_dir))

    searcher.set_bm25(k1=args.k1, b=args.b)

    n = 0
    for qid, qtext in iter_queries_jsonl(qpath):
        hits = searcher.search(qtext, k=args.topk)
        pairs = [(h.docid, float(h.score)) for h in hits]
        write_trec_run(run_path, qid, pairs, runid=args.runid)
        n += 1
        if n % 25 == 0:
            print(f"{n} queries...")

    print(f"OK: {n} queries -> {run_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, gzip
from pathlib import Path

import pandas as pd
import pyterrier as pt

from contextlib import contextmanager
@contextmanager
def tracking(export_file_path=None):
    yield

def _open(path: Path):
    s = str(path)
    if s.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def load_topics_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"qid": str(obj["query_id"]), "query": obj["query"]})
    return pd.DataFrame(rows, columns=["qid", "query"])

def main():
    ap = argparse.ArgumentParser(description="Retrieve with PyTerrier BM25 (Terrier defaults), write TREC run")
    ap.add_argument("--index_dir", required=True, help="Path to Terrier index directory")
    ap.add_argument("--queries_path", required=True, help="Path to *-queries-*.jsonl")
    ap.add_argument("--run_path", required=True, help="Output run path (.run or .run.gz)")
    ap.add_argument("--runid", required=True, help="Run ID (column 6 in TREC run)")
    ap.add_argument("--topk", type=int, default=1000, help="Top-k per query")
    args = ap.parse_args()

    index_dir = Path(args.index_dir).resolve()
    qpath = Path(args.queries_path).resolve()
    run_path = Path(args.run_path).resolve()
    run_path.parent.mkdir(parents=True, exist_ok=True)
    if run_path.exists():
        run_path.unlink()

    try:
        pt.java.init()
    except Exception:
        pass

    index = pt.IndexFactory.of(str(index_dir))
    bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=args.topk)

    tokeniser = pt.java.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
    topics = load_topics_jsonl(qpath)
    topics["query"] = topics["query"].apply(lambda s: " ".join(tokeniser.getTokens(s)))

    with tracking(export_file_path=run_path.parent / "retrieval-ir-metadata.yml"):
        run_df = bm25(topics)

        pt.io.write_results(run_df, str(run_path), run_name=args.runid)
        print(f"OK: {len(topics)} queries -> {run_path}")

if __name__ == "__main__":
    main()

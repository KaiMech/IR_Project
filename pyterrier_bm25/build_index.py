#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, gzip
from pathlib import Path
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

def iter_docs_jsonl(corpus_path: Path):
    with _open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield {"docno": str(obj["id"]), "text": obj.get("text", "")}

def main():
    ap = argparse.ArgumentParser(description="Build PyTerrier BM25 index from ToT JSONL corpus")
    ap.add_argument("--corpus", required=True, help="Path to *-corpus.jsonl or .jsonl.gz")
    ap.add_argument("--index_dir", required=True, help="Output directory for Terrier index")
    args = ap.parse_args()

    corpus = Path(args.corpus).resolve()
    index_dir = Path(args.index_dir).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    try:
        pt.java.init()
    except Exception:
        pass

    marker = index_dir / "index-ir-metadata.yml"
    if marker.exists():
        print(f"OK: index exists at {index_dir}")
        return

    with tracking(export_file_path=marker):

        indexer = pt.IterDictIndexer(
            str(index_dir),
            overwrite=True,
            meta={"docno": 100, "text": 20480}
        )
        indexer.index(iter_docs_jsonl(corpus))
        print(f"OK: index created at {index_dir}")

if __name__ == "__main__":
    main()

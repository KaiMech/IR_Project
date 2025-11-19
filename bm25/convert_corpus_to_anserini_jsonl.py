#!/usr/bin/env python3
# scripts/convert_corpus_to_anserini_jsonl.py
import argparse, gzip, json
from pathlib import Path

def iter_jsonl(path: Path):
    if str(path).endswith('.gz'):
        f = gzip.open(path, 'rt', encoding='utf-8')
    else:
        f = open(path, 'r', encoding='utf-8')
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser(
        description="Convert a Wikipedia-like JSONL to Anserini JsonCollection JSONL (id, contents)."
    )
    ap.add_argument("--input", required=True, help="Path to *.jsonl or *.jsonl.gz (with fields id,title,text)")
    ap.add_argument("--output_dir", required=True, help="Directory to write JsonCollection files into")
    ap.add_argument("--outfile", default="docs.jsonl", help="Output file name inside output_dir")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.outfile

    n = 0
    with open(out_path, 'w', encoding='utf-8') as out:
        for obj in iter_jsonl(in_path):
            doc_id = str(obj.get("id", "")).strip()
            title = (obj.get("title") or "").strip()
            text  = (obj.get("text")  or "").strip()
            if not doc_id:
                continue
            contents = title + ("\n" if title and text else "") + text
            out.write(json.dumps({"id": doc_id, "contents": contents}, ensure_ascii=False) + "\n")
            n += 1

    print(f"OK: wrote {n} docs to {out_path}")

if __name__ == "__main__":
    main()

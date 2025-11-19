#!/usr/bin/env python3
# scripts/build_index.py
import argparse, shutil, subprocess, sys
from pathlib import Path

def require_java():
    try:
        out = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser(description="Build a Lucene index from an Anserini JsonCollection using Pyserini.")
    ap.add_argument("--input_dir", required=True, help="Directory containing JsonCollection .jsonl files (e.g., docs.jsonl)")
    ap.add_argument("--index_dir", required=True, help="Output index directory")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    if not require_java():
        sys.exit("Java nicht gefunden. Bitte eine aktuelle JDK (17/21) installieren (java -version).")

    input_dir = Path(args.input_dir).resolve()
    index_dir = Path(args.index_dir).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(input_dir),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(args.threads),
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"OK: index created at {index_dir}")

if __name__ == "__main__":
    main()

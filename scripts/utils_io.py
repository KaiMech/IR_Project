# scripts/utils_io.py
from __future__ import annotations
import gzip, json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

# ---------- Datei-Handling ----------
def open_text_auto(path: Path | str):
    p = str(path)
    return gzip.open(p, "rt", encoding="utf-8", newline="") if p.endswith(".gz") \
           else open(p, "r", encoding="utf-8", newline="")

# ---------- Repo-Root finden ----------
def find_repo_root(start: Path | None = None) -> Path:
    root = (start or Path.cwd()).resolve()
    while not (root / ".git").exists() and root != root.parent:
        root = root.parent
    return root

# ---------- Split -> lokale Datei-Pfade ----------
_QUERIES_FN = {
    "train": "train-2025-queries.jsonl",
    "dev1" : "dev1-2025-queries.jsonl",
    "dev2" : "dev2-2025-queries.jsonl",
    "dev3" : "dev3-2025-queries.jsonl",
    "test" : "test-2025-queries.jsonl",
}
_QRELS_FN = {
    "train": "train-2025-qrel.txt",
    "dev1" : "dev1-2025-qrel.txt",
    "dev2" : "dev2-2025-qrel.txt",
    "dev3" : "dev3-2025-qrel.txt",
}

def get_queries_path(split: str, root: Path | None = None) -> Path:
    root = find_repo_root(root)
    path = root / "data" / "tot25" / _QUERIES_FN[split]
    return path

def get_qrels_path(split: str, root: Path | None = None) -> Optional[Path]:
    if split not in _QRELS_FN:  # test hat keine qrels
        return None
    root = find_repo_root(root)
    return root / "data" / "tot25" / _QRELS_FN[split]

# ---------- Reader ----------
def iter_queries_jsonl(path: Path | str) -> Iterable[Tuple[str, str]]:
    with open_text_auto(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            yield obj["query_id"], obj["query"]

def load_qrels_trec(path: Path | str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4: 
                continue
            qid, _zero, pid, rel = parts[:4]
            qrels.setdefault(qid, {})[pid] = int(rel)
    return qrels

# ---------- TREC Run I/O ----------
def write_trec_run_line(fh, qid: str, pid: str | int, rank: int, score: float, runid: str):
    fh.write(f"{qid} Q0 {pid} {rank} {score:.6f} {runid}\n")

def write_trec_run(run_path: Path | str, qid: str, hits: List[Tuple[str | int, float]], runid: str, k: int = 1000):
    # sortieren + deduplizieren
    seen = set()
    ranked: List[Tuple[str | int, float]] = []
    for pid, score in sorted(hits, key=lambda x: -x[1]):
        if pid in seen: continue
        seen.add(pid)
        ranked.append((pid, score))
        if len(ranked) >= k: break
    with open(run_path, "a", encoding="utf-8") as f:
        for rank, (pid, score) in enumerate(ranked, 1):
            write_trec_run_line(f, qid, pid, rank, score, runid)

def load_trec_run(path: Path | str) -> Dict[str, Dict[str, float]]:
    run: Dict[str, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid, _q0, docno, _rank, score, _runid = line.split()
            run.setdefault(qid, {})[docno] = float(score)
    return run

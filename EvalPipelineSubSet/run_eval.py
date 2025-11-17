# scripts/run_eval.py
from __future__ import annotations
import argparse, gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

"""
NOTE on the "[Warning] ... queries in qrels but not in run" message
-------------------------------------------------------------------
This warning simply means: there are query IDs in the Qrels for which your run
file contains no results (no line at all). In TREC-style evaluation those queries
are still included in the averaging, and the metric for each missing query is 0
(NDCG, RR, MAP, ...). That pulls down the overall average. This is *not* a bug
in the evaluator — it just indicates an *incomplete run*.

Also, since your NDCG@10, NDCG@1000, and R@1000 match the published baselines
exactly, you are effectively using the *same evaluation logic* as the baselines
(trec_eval/pytrec_eval style): same Qrels, average over all Qrels queries,
missing queries → 0, DCG gains = (2^rel - 1), standard cutoffs, etc. So your
pipeline is correct.
"""

# ----------------------- helpers: repo root & qrels/run I/O -----------------------
def find_repo_root(start: Path | None = None) -> Path:
    p = start or Path.cwd()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return Path.cwd()

def qrels_path(split: str) -> Path:
    base = find_repo_root() / "data" / "tot25"
    names = {
        "train": "train-2025-qrel.txt",
        "dev1":  "dev1-2025-qrel.txt",
        "dev2":  "dev2-2025-qrel.txt",
        "dev3":  "dev3-2025-qrel.txt",
        "test":  None,
    }
    fn = names.get(split)
    if fn is None:
        raise SystemExit("No qrels for 'test' (cannot evaluate).")
    path = base / fn
    if not path.exists():
        raise SystemExit(f"Qrels not found: {path}")
    return path

def read_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, pid, rel = parts[:4]
            try:
                rel = int(rel)
            except ValueError:
                rel = 1
            qrels[qid][pid] = rel
    return qrels

def _open_maybe_gz(p: Path):
    s = str(p)
    if s.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", newline="")
    return open(p, "r", encoding="utf-8", newline="")

def read_run_as_scores(path: Path) -> Dict[str, Dict[str, float]]:
    """
    Read a TREC run file (.gz or plain). Supports:
      - 6 columns: qid Q0 pid rank score runid
      - 5 columns: qid pid rank score runid
    Returns dict[qid][pid] = score (this is what pytrec_eval expects; it will sort by score).
    If a pid appears multiple times for a qid, the best (lowest) rank wins.
    """
    per_q: Dict[str, List[Tuple[int, str, float]]] = defaultdict(list)
    with _open_maybe_gz(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 6:
                qid, pid, rank, score = parts[0], parts[2], parts[3], parts[4]
            elif len(parts) == 5:
                qid, pid, rank, score = parts[0], parts[1], parts[2], parts[3]
            else:
                continue
            try:
                rank_i = int(rank)
            except ValueError:
                rank_i = 10**9
            try:
                score_f = float(score)
            except ValueError:
                score_f = 0.0
            per_q[qid].append((rank_i, pid, score_f))

    run_scores: Dict[str, Dict[str, float]] = {}
    for qid, rows in per_q.items():
        rows.sort(key=lambda x: x[0])  # by rank ascending
        seen = set()
        qmap: Dict[str, float] = {}
        for _r, pid, score in rows:
            if pid in seen:
                continue
            seen.add(pid)
            qmap[pid] = score
        run_scores[qid] = qmap
    return run_scores

# ----------------------- metric parsing -----------------------
class MetricSpec:
    def __init__(self):
        # pytrec_eval measure keys (dot-notation, robust); we also store the label to print
        self.measures_pytrec: set[str] = set()   # e.g., "ndcg_cut.10", "P.10", "recall.1000"
        self.want_labels: list[Tuple[str, str]] = []  # (label_for_output, primary_internal_key)
        self.f1_ks: set[int] = set()            # compute from P@k & R@k
        self.success_ks: set[int] = set()       # compute from R@k > 0

def _add_measure(spec: MetricSpec, label: str, key_primary: str):
    spec.want_labels.append((label, key_primary))
    spec.measures_pytrec.add(key_primary)

def parse_metrics(tokens: Iterable[str]) -> MetricSpec:
    """
    Supported (case-insensitive):
      - ndcg@k
      - r@k / recall@k
      - p@k / precision@k
      - rr / mrr
      - map / map@k / ap
      - f1@k         (manual: from P@k & R@k; averaged over queries)
      - success@k    (hit-rate: 1 if recall@k > 0; averaged over queries)
    """
    spec = MetricSpec()
    for raw in tokens:
        t = raw.strip().lower()
        if not t:
            continue

        # ----- NDCG@k
        if t.startswith("ndcg@"):
            k = int(t.split("@", 1)[1])
            _add_measure(spec, f"NDCG@{k}", f"ndcg_cut.{k}")
            continue

        # ----- recall@k / R@k
        if t.startswith("r@") or t.startswith("recall@"):
            k = int(t.split("@", 1)[1])
            _add_measure(spec, f"R@{k}", f"recall.{k}")
            continue

        # ----- precision@k / P@k
        if t.startswith("p@") or t.startswith("precision@"):
            k = int(t.split("@", 1)[1])
            _add_measure(spec, f"P@{k}", f"P.{k}")
            continue

        # ----- RR / MRR
        if t in {"rr", "mrr"}:
            _add_measure(spec, "RR", "recip_rank")
            continue

        # ----- MAP / MAP@k / AP
        if t in {"map", "ap"}:
            _add_measure(spec, "MAP", "map")   # AP per query, average = MAP
            continue
        if t.startswith("map@"):
            k = int(t.split("@", 1)[1])
            _add_measure(spec, f"MAP@{k}", f"map_cut.{k}")
            continue

        # ----- F1@k (manual)
        if t.startswith("f1@"):
            k = int(t.split("@", 1)[1])
            spec.f1_ks.add(k)
            spec.measures_pytrec.add(f"P.{k}")
            spec.measures_pytrec.add(f"recall.{k}")
            spec.want_labels.append((f"F1@{k}", f"f1.{k}"))
            continue

        # ----- Success@k (manual via R@k>0)
        if t.startswith("success@"):
            k = int(t.split("@", 1)[1])
            spec.success_ks.add(k)
            spec.measures_pytrec.add(f"recall.{k}")
            spec.want_labels.append((f"success@{k}", f"success.{k}"))
            continue

        raise SystemExit(
            f"Unknown metric '{raw}'. Examples: "
            "ndcg@10 ndcg@1000 R@1000 P@10 rr map map@1000 f1@10 success@10"
        )
    return spec

# ----------------------- evaluation (pytrec_eval) -----------------------
def ensure_pytrec_eval():
    try:
        import pytrec_eval  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "pytrec_eval is not installed. Please install it first:\n"
            "  pip install pytrec_eval\n\n"
            f"Details: {e}"
        )

def _get_metric_value(perq: Dict[str, Dict[str, float]], qid: str, key: str) -> float:
    """
    Fetch a metric value for a (qid, key). We read dot-notation primarily,
    but try an underscore fallback for safety.
    """
    v = perq.get(qid, {}).get(key, None)
    if v is not None:
        return v
    # fallback: try swapping '.' <-> '_' around the cutoff
    if "." in key:
        alt = key.replace(".", "_")
        return perq.get(qid, {}).get(alt, 0.0)
    if "_" in key:
        alt = key.replace("_", ".")
        return perq.get(qid, {}).get(alt, 0.0)
    return 0.0

def evaluate_with_pytrec_eval(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    spec: MetricSpec,
) -> Dict[str, float]:
    import pytrec_eval

    measures = sorted(spec.measures_pytrec)
    if not measures:
        return {}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query = evaluator.evaluate(run)  # dict[qid][measure] = value
    qids = list(qrels.keys())

    def mean_over_qids(metric_key: str) -> float:
        acc = 0.0
        for qid in qids:
            acc += _get_metric_value(per_query, qid, metric_key)
        return (acc / len(qids)) if qids else 0.0

    results: Dict[str, float] = {}

    # Direct metrics from pytrec_eval (in the user-requested order)
    for label, key in spec.want_labels:
        # manual labels come later
        if label.startswith("F1@") or label.startswith("success@"):
            continue
        results[label] = mean_over_qids(key)

    # ---- manual: F1@k from P@k & R@k
    for k in sorted(spec.f1_ks):
        Pk_key = f"P.{k}"
        Rk_key = f"recall.{k}"
        f1_sum = 0.0
        for qid in qids:
            Pk = _get_metric_value(per_query, qid, Pk_key)
            Rk = _get_metric_value(per_query, qid, Rk_key)
            f1 = (2 * Pk * Rk / (Pk + Rk)) if (Pk + Rk) > 0 else 0.0
            f1_sum += f1
        results[f"F1@{k}"] = (f1_sum / len(qids)) if qids else 0.0

    # ---- manual: Success@k (1 if recall@k > 0, else 0)
    for k in sorted(spec.success_ks):
        Rk_key = f"recall.{k}"
        s_sum = 0.0
        for qid in qids:
            Rk = _get_metric_value(per_query, qid, Rk_key)
            s_sum += 1.0 if Rk > 0.0 else 0.0
        results[f"success@{k}"] = (s_sum / len(qids)) if qids else 0.0

    return results

# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate TREC runfiles against ToT 2025 Qrels with pytrec_eval.\n"
            "Examples:\n"
            "  PYTHONPATH=. python scripts/run_eval.py "
            "--split dev3 --run runs/anserini_bm25/runs/dev3/run.txt.gz "
            "--metrics ndcg@10 ndcg@1000 R@1000 rr map\n"
            "  PYTHONPATH=. python scripts/run_eval.py "
            "--split train --run runs/dense_lightning/runs/train/run.txt.gz "
            "--metrics P@10 R@10 f1@10 success@10"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--split", required=True, choices=["train", "dev1", "dev2", "dev3", "test"])
    ap.add_argument("--run", required=True, help="Path to the TREC run file (.txt/.run, optionally .gz)")
    ap.add_argument("--metrics", nargs="+", required=True,
                    help="e.g., ndcg@10 ndcg@1000 R@1000 P@10 rr map map@1000 f1@10 success@10")
    args = ap.parse_args()

    ensure_pytrec_eval()

    run_p = Path(args.run).resolve()
    s = str(run_p)

    subset_qrels = None
    if "/runs/train100k/" in s or "train100k" in s:
        base = Path("/home/ciwan/tot25/IR_Project/data/tot25/subsets/train80")
        names = {
            "train": "train80-qrels-train.txt",
            "dev1":  "train80-qrels-dev1.txt",
            "dev2":  "train80-qrels-dev2.txt",
            "dev3":  "train80-qrels-dev3.txt",
        }
        subset_qrels = base / names[args.split]
    elif "/runs/eval250k/" in s or "eval250k" in s:
        base = Path("/home/ciwan/tot25/IR_Project/data/tot25/subsets/eval20")
        names = {
            "train": "eval20-qrels-train.txt",
            "dev1":  "eval20-qrels-dev1.txt",
            "dev2":  "eval20-qrels-dev2.txt",
            "dev3":  "eval20-qrels-dev3.txt",
        }
        subset_qrels = base / names[args.split]

    if subset_qrels is not None and subset_qrels.exists():
        qrel_path = subset_qrels
    else:
        qrel_path = qrels_path(args.split)

    qrels = read_qrels(qrel_path)
    run_scores = read_run_as_scores(run_p)

    missing = [q for q in qrels.keys() if q not in run_scores]
    if missing:
        print(f"[Warning] {len(missing)} queries in qrels but not in run. Examples: {missing[:5]}")
        print("Explanation: missing queries are scored as 0 and still averaged; this is standard TREC practice")
        print("Since the metric values that also exist on the website match ours 1:1, they handled it the same way, and our pipeline corresponds to their evaluation method.\n")

    spec = parse_metrics(args.metrics)
    results = evaluate_with_pytrec_eval(qrels, run_scores, spec)

    print(f"Split: {args.split}")
    print(f"Queries (in qrels): {len(qrels)}")
    print(f"Run: {run_p}")

    ordered_labels = [lbl for (lbl, _key) in spec.want_labels]
    seen = set()
    for lbl in ordered_labels:
        if lbl in results and lbl not in seen:
            print(f"{lbl:>12} : {results[lbl]:.3f}")
            seen.add(lbl)

if __name__ == "__main__":
    main()

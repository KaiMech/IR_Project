#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generates two deduplicated subsets from the TREC ToT 2025 Wikipedia corpus:

  1) train100k-corpus.jsonl.gz  (targets 100,000 docs; contains the 80% “gold” side)
  2) eval250k-corpus.jsonl.gz   (targets 250,000 docs; contains the 20% “gold” side)

  Terminology
- What we mean with “Gold” / “gold document ID”: For each query, the original qrels file specifies exactly one
  relevant document (its Wikipedia page ID (the value of the “id” field in the corpus)).
  Whenever we say “include the gold doc IDs”, we mean: include those exact document lines
  from the main corpus into the subset. (We chose this terminology, because so we can write less while creating this comment.)

Additionally produces:
  - train80-queries.jsonl / eval20-queries.jsonl          (aggregate files with full query objects)
  - train80-qrels.txt     / eval20-qrels.txt              (aggregate qrels)
  - train80-queries-{split}.jsonl / eval20-queries-{split}.jsonl   (per split: train/dev1/dev2/dev3)
  - train80-qrels-{split}.txt     / eval20-qrels-{split}.txt       (per split: train/dev1/dev2/dev3)

Companion files:
  - train_ids.txt / eval_ids.txt
  - train_queries_80.jsonl / eval_queries_20.jsonl (just the selected QID lists)
  - stats.json
  - missing_gold.log (QIDs whose gold doc is not present in the main corpus (in our case the script runs perfect and we don't have any missing gold docs)) 

80/20 split:
  For each split (train, dev1, dev2, dev3), query IDs are split 80%/20% (seed-controlled).
  Gold Wikipedia IDs for the 80% side go to the 100k subset; for the 20% side to the 250k subset.

Additions:
  - For each gold document, add K neighbors above and below it according to the offsets-order.
  - For each query and for each baseline model (Anserini BM25, PyTerrier BM25, Dense Retrieval),
    add the Top-N from its runfile. Cross-model overlaps per query are skipped and replaced by
    deeper ranks until each model contributes N unique IDs. No duplicates with already collected
    IDs (gold/neighbors/previous additions) within the same target subset.
  - Processing order is: train → dev1 → dev2 → dev3, ensuring later splits do not add IDs
    already introduced by earlier splits for the same target subset (as required).

Filling to target sizes:
  If, after all steps, the subset has fewer than the target size, fill with random, not-yet-used
  document IDs sampled from the original corpus. Overlaps *between* the two subsets due to random
  filling are allowed; within a subset, duplicates are forbidden.

Writing:
  Lines are copied 1:1 from the original large corpus (no re-serialization) to preserve exact
  formatting and content.

Notes:
  - Query/qrels outputs include only QIDs whose gold doc actually ended up in the final subset.

"""

import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path
from collections import defaultdict

# --------------------------
# Pfad-Defaults, if u want to run the code, please change DEFAULT_DATA_ROOT accordingly.
# --------------------------

DEFAULT_DATA_ROOT = "/home/ciwan/tot25/IR_Project/data/tot25"

# Qrels / Queries
QRELS = {
    "train": "train-2025-qrel.txt",
    "dev1":  "dev1-2025-qrel.txt",
    "dev2":  "dev2-2025-qrel.txt",
    "dev3":  "dev3-2025-qrel.txt",
}
QUERIES = {
    "train": "train-2025-queries.jsonl",
    "dev1":  "dev1-2025-queries.jsonl",
    "dev2":  "dev2-2025-queries.jsonl",
    "dev3":  "dev3-2025-queries.jsonl",
}

# Runfiles (3 Models × 4 Splits)
RUNFILES = {
    "anserini": {
        "train": "run_anserini_train.txt",
        "dev1":  "run_anserini_dev1.txt",
        "dev2":  "run_anserini_dev2.txt",
        "dev3":  "run_anserini_dev3.txt",
    },
    "pyterrier": {
        "train": "run_pyterrier_train.txt",
        "dev1":  "run_pyterrier_dev1.txt",
        "dev2":  "run_pyterrier_dev2.txt",
        "dev3":  "run_pyterrier_dev3.txt",
    },
    "retrieval": {
        "train": "run_retrieval_train.txt",
        "dev1":  "run_retrieval_dev1.txt",
        "dev2":  "run_retrieval_dev2.txt",
        "dev3":  "run_retrieval_dev3.txt",
    },
}

# Corpus & Offsets
CORPUS_FILE  = "trec-tot-2025-corpus.jsonl.gz"
OFFSETS_FILE = "trec-tot-2025-offsets.jsonl.gz"

# --------------------------
# Utils
# --------------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_neighbors_arg(s: str):
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        lo = int(a)
        hi = int(b)
        if lo < 0 or hi < lo:
            raise ValueError("Invalid range for --neighbors")
        return ("range", (lo, hi))
    else:
        k = int(s)
        if k < 0:
            raise ValueError("Invalid range for --neighbors")
        return ("fixed", k)

def read_qrels(path: Path):

    mapping = {}
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            qid = parts[0]
            docid = parts[2]
            mapping[qid] = docid
    return mapping

def read_queries_jsonl_map(path: Path):

    mp = {}
    if not path.exists():
        return mp
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = str(obj.get("query_id", "")).strip()
            if qid:
                mp[qid] = obj
    return mp

def partition_qids_80_20(qrels_qid2doc, seed: int):

    qids = sorted(qrels_qid2doc.keys(), key=lambda x: (len(x), x))
    rnd = random.Random(seed)
    rnd.shuffle(qids)
    n80 = int(len(qids) * 0.8 + 1e-9)
    set80 = set(qids[:n80])
    set20 = set(qids[n80:])
    return set80, set20

def read_offsets_order(offsets_gz_path: Path):

    order_ids = []
    id2idx = {}
    with gzip.open(offsets_gz_path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            docid = str(o.get("id", "")).strip()
            if not docid:
                continue
            order_ids.append(docid)
            id2idx[docid] = idx
    return order_ids, id2idx

def neighbors_for(docid: str, id2idx, order_ids, neighbors_mode, rng: random.Random):

    if docid not in id2idx:
        return []
    idx = id2idx[docid]
    n_total = len(order_ids)

    if neighbors_mode[0] == "fixed":
        k = int(neighbors_mode[1])
    else:
        lo, hi = neighbors_mode[1]
        k = rng.randint(lo, hi)

    neigh = []
    for d in range(1, k + 1):
        j = idx - d
        if j >= 0:
            neigh.append(order_ids[j])
    for d in range(1, k + 1):
        j = idx + d
        if j < n_total:
            neigh.append(order_ids[j])

    return neigh

def read_runfile_grouped(path: Path):

    grouped = defaultdict(list)
    if not path.exists():
        return grouped
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            qid = parts[0]
            pid = parts[2]
            grouped[qid].append(pid)
    return grouped

def add_topn_unique_for_query(qid, ranked_list, n_target, dest_set, seen_set_global, seen_this_query):

    added = 0
    for pid in ranked_list:
        if pid in seen_set_global:
            continue
        if pid in seen_this_query:
            continue
        seen_this_query.add(pid)
        dest_set.add(pid)
        seen_set_global.add(pid)
        added += 1
        if added >= n_target:
            return added, False
    return added, True  # exhausted

def write_selected_corpora(original_corpus_gz: Path,
                           out_train_gz: Path,
                           out_eval_gz: Path,
                           train_ids: set,
                           eval_ids: set,
                           stats: dict):

    count_train = 0
    count_eval = 0
    if out_train_gz.exists():
        out_train_gz.unlink()
    if out_eval_gz.exists():
        out_eval_gz.unlink()

    with gzip.open(original_corpus_gz, "rt", encoding="utf-8") as fin, \
         gzip.open(out_train_gz, "wt", encoding="utf-8") as ftrain, \
         gzip.open(out_eval_gz, "wt", encoding="utf-8") as feval:
        for line in fin:
            try:
                obj = json.loads(line)
                did = str(obj.get("id", "")).strip()
            except Exception:
                continue
            if did in train_ids:
                ftrain.write(line)
                count_train += 1
            if did in eval_ids:
                feval.write(line)
                count_eval += 1

    stats["written_train_lines"] = count_train
    stats["written_eval_lines"] = count_eval

def save_jsonl(path: Path, rows):
    with open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_qrels_txt(path: Path, qrels_map_by_split: dict, selected_qids_by_split: dict):

    with open(path, "wt", encoding="utf-8") as f:
        for split in ["train", "dev1", "dev2", "dev3"]:
            qrels_map = qrels_map_by_split.get(split, {})
            sel_qids  = selected_qids_by_split.get(split, set())
            for qid in sorted(sel_qids, key=lambda x: (len(x), x)):
                did = qrels_map.get(qid)
                if not did:
                    continue
                f.write(f"{qid} 0 {did} 1\n")

def write_qrels_txt_per_split(base_dir: Path, prefix: str, qrels_map_by_split: dict, selected_qids_by_split: dict):

    for split in ["train", "dev1", "dev2", "dev3"]:
        out_path = base_dir / f"{prefix}-qrels-{split}.txt"
        with open(out_path, "wt", encoding="utf-8") as f:
            qrels_map = qrels_map_by_split.get(split, {})
            sel_qids  = selected_qids_by_split.get(split, set())
            for qid in sorted(sel_qids, key=lambda x: (len(x), x)):
                did = qrels_map.get(qid)
                if not did:
                    continue
                f.write(f"{qid} 0 {did} 1\n")

def write_queries_jsonl(path: Path, queries_map_by_split: dict, selected_qids_by_split: dict):

    rows = []
    for split in ["train", "dev1", "dev2", "dev3"]:
        qmap = queries_map_by_split.get(split, {})
        sel  = selected_qids_by_split.get(split, set())
        for qid in sorted(sel, key=lambda x: (len(x), x)):
            obj = qmap.get(qid)
            if obj:
                rows.append(obj)
    save_jsonl(path, rows)

def write_queries_jsonl_per_split(base_dir: Path, prefix: str, queries_map_by_split: dict, selected_qids_by_split: dict):

    for split in ["train", "dev1", "dev2", "dev3"]:
        out_path = base_dir / f"{prefix}-queries-{split}.jsonl"
        rows = []
        qmap = queries_map_by_split.get(split, {})
        sel  = selected_qids_by_split.get(split, set())
        for qid in sorted(sel, key=lambda x: (len(x), x)):
            obj = qmap.get(qid)
            if obj:
                rows.append(obj)
        save_jsonl(out_path, rows)

def main():
    ap = argparse.ArgumentParser(description="Build 100k/250k subcorpora + matching queries/qrels (incl. per-split files) for TREC ToT 2025.")
    ap.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                    help="Root directory with the provided files.")
    ap.add_argument("--neighbors", type=str, default="1-3",
                    help="K neighbors per page: fixed number '2' or range '1-3'.")
    ap.add_argument("--per-model", type=int, default=50,
                    help="Top-N per model and query (default 50).")
    ap.add_argument("--train-size", type=int, default=100000,
                    help="Target size of the train subset.")
    ap.add_argument("--eval-size", type=int, default=250000,
                    help="Target size of the eval subset.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for reproducibility.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir   = data_root / "subsets"
    ensure_dir(out_dir)

    neighbors_mode = parse_neighbors_arg(args.neighbors)
    rnd = random.Random(args.seed)

    corpus_path  = data_root / CORPUS_FILE
    offsets_path = data_root / OFFSETS_FILE


    qrels_all = {}
    for split, fname in QRELS.items():
        path = data_root / fname
        if not path.exists():
            eprint(f"[WARN] Missing qrels: {path}")
        qrels_all[split] = read_qrels(path)


    queries_map_all = {}
    for split, fname in QUERIES.items():
        path = data_root / fname
        queries_map_all[split] = read_queries_jsonl_map(path)


    eighty_qids = {}
    twenty_qids = {}
    for split in ["train", "dev1", "dev2", "dev3"]:
        qrels_map = qrels_all.get(split, {})
        s80, s20 = partition_qids_80_20(qrels_map, seed=args.seed + hash(split) % 100000)
        eighty_qids[split] = s80
        twenty_qids[split] = s20

    gold_train_ids = set()
    gold_eval_ids  = set()
    missing_gold_log = []

    eprint("[INFO] Reading offsets/order…")
    order_ids, id2idx = read_offsets_order(offsets_path)
    all_doc_ids_set = set(order_ids)

    for split in ["train", "dev1", "dev2", "dev3"]:
        qrels_map = qrels_all.get(split, {})
        for qid, did in qrels_map.items():
            if qid in eighty_qids[split]:
                if did in all_doc_ids_set:
                    gold_train_ids.add(did)
                else:
                    missing_gold_log.append((split, qid, did))
            elif qid in twenty_qids[split]:
                if did in all_doc_ids_set:
                    gold_eval_ids.add(did)
                else:
                    missing_gold_log.append((split, qid, did))

    eprint(f"[INFO] Gold Docs: train={len(gold_train_ids)} eval={len(gold_eval_ids)}")
    if missing_gold_log:
        eprint(f"[WARN] {len(missing_gold_log)} gold doc IDs are missing from the corpus/offsets (see missing_gold.log)")

    eprint("[INFO] Adding neighbors for gold documents…")
    train_ids = set(gold_train_ids)
    eval_ids  = set(gold_eval_ids)

    for did in list(gold_train_ids):
        neigh = neighbors_for(did, id2idx, order_ids, neighbors_mode, rnd)
        for nd in neigh:
            train_ids.add(nd)

    for did in list(gold_eval_ids):
        neigh = neighbors_for(did, id2idx, order_ids, neighbors_mode, rnd)
        for nd in neigh:
            eval_ids.add(nd)


    models_order = ["anserini", "pyterrier", "retrieval"]
    per_model = max(0, int(args.per_model))

    eprint("[INFO] Reading runfiles and adding top-N per query/model…")
    seen_train = set(train_ids)  
    seen_eval  = set(eval_ids)

    for split in ["train", "dev1", "dev2", "dev3"]:
        runs_for_split = {}
        for model in models_order:
            run_path = data_root / RUNFILES[model][split]
            if not run_path.exists():
                eprint(f"[WARN] Missing runfile: {run_path}")
                runs_for_split[model] = {}
            else:
                runs_for_split[model] = read_runfile_grouped(run_path)

        qrels_map = qrels_all.get(split, {})
        qids_in_split = list(qrels_map.keys())
        qids_in_split.sort(key=lambda x: (len(x), x))

        for qid in qids_in_split:
            target_set_name = "train" if (qid in eighty_qids[split]) else "eval"
            if (qid not in eighty_qids[split]) and (qid not in twenty_qids[split]):
                continue

            if target_set_name == "train":
                dest_set = train_ids
                seen_global = seen_train
            else:
                dest_set = eval_ids
                seen_global = seen_eval

            seen_this_query = set()
            for model in models_order:
                ranked = runs_for_split.get(model, {}).get(qid, [])
                if not ranked:
                    continue
                add_topn_unique_for_query(
                    qid, ranked, per_model, dest_set, seen_global, seen_this_query
                )

    def fill_random_to_size(dest_set: set, target_size: int, rng: random.Random, label: str):
        if len(dest_set) > target_size:
            if label == "train":
                gold_keep = gold_train_ids
            else:
                gold_keep = gold_eval_ids
            keep = set(gold_keep)
            for did in order_ids:
                if len(keep) >= target_size:
                    break
                if did in dest_set and did not in keep:
                    keep.add(did)
            return keep

        need = target_size - len(dest_set)
        if need <= 0:
            return dest_set

        shuffled_indices = list(range(len(order_ids)))
        rng.shuffle(shuffled_indices)
        added = 0
        for idx in shuffled_indices:
            did = order_ids[idx]
            if did not in dest_set:
                dest_set.add(did)
                added += 1
                if added >= need:
                    break
        return dest_set

    eprint("[INFO] Filling with random documents up to target sizes…")
    train_ids = fill_random_to_size(train_ids, args.train_size, rnd, "train")
    eval_ids  = fill_random_to_size(eval_ids,  args.eval_size,  rnd, "eval")


    out_dir.mkdir(parents=True, exist_ok=True)
    out_train_gz = out_dir / "train100k-corpus.jsonl.gz"
    out_eval_gz  = out_dir / "eval250k-corpus.jsonl.gz"

    stats = {
        "seed": args.seed,
        "neighbors_mode": args.neighbors,
        "per_model": args.per_model,
        "target_sizes": {"train": args.train_size, "eval": args.eval_size},
        "counts": {
            "gold_train": len(gold_train_ids),
            "gold_eval": len(gold_eval_ids),
            "train_total_ids": len(train_ids),
            "eval_total_ids": len(eval_ids),
        }
    }

    eprint("[INFO] Write the subcorpora (1:1 lines from the original)...")
    write_selected_corpora(
        original_corpus_gz=corpus_path,
        out_train_gz=out_train_gz,
        out_eval_gz=out_eval_gz,
        train_ids=train_ids,
        eval_ids=eval_ids,
        stats=stats
    )


    eprint("[INFO] Writing accompanying files...")
    with open(out_dir / "train_ids.txt", "wt", encoding="utf-8") as f:
        for did in train_ids:
            f.write(f"{did}\n")
    with open(out_dir / "eval_ids.txt", "wt", encoding="utf-8") as f:
        for did in eval_ids:
            f.write(f"{did}\n")

    train_query_rows = []
    eval_query_rows  = []
    for split in ["train", "dev1", "dev2", "dev3"]:
        for qid in sorted(eighty_qids[split], key=lambda x: (len(x), x)):
            train_query_rows.append({"split": split, "query_id": qid})
        for qid in sorted(twenty_qids[split], key=lambda x: (len(x), x)):
            eval_query_rows.append({"split": split, "query_id": qid})

    save_jsonl(out_dir / "train_queries_80.jsonl", train_query_rows)
    save_jsonl(out_dir / "eval_queries_20.jsonl",  eval_query_rows)

    selected_train_qids_by_split = {s: set() for s in ["train", "dev1", "dev2", "dev3"]}
    selected_eval_qids_by_split  = {s: set() for s in ["train", "dev1", "dev2", "dev3"]}

    for split in ["train", "dev1", "dev2", "dev3"]:
        qrels_map = qrels_all.get(split, {})
        # 80% -> train-Set
        for qid in eighty_qids[split]:
            did = qrels_map.get(qid)
            if did and did in train_ids:
                selected_train_qids_by_split[split].add(qid)
        # 20% -> eval-Set
        for qid in twenty_qids[split]:
            did = qrels_map.get(qid)
            if did and did in eval_ids:
                selected_eval_qids_by_split[split].add(qid)

    write_queries_jsonl(
        path = out_dir / "train80-queries.jsonl",
        queries_map_by_split = queries_map_all,
        selected_qids_by_split = selected_train_qids_by_split
    )
    write_queries_jsonl(
        path = out_dir / "eval20-queries.jsonl",
        queries_map_by_split = queries_map_all,
        selected_qids_by_split = selected_eval_qids_by_split
    )
    write_qrels_txt(
        path = out_dir / "train80-qrels.txt",
        qrels_map_by_split = qrels_all,
        selected_qids_by_split = selected_train_qids_by_split
    )
    write_qrels_txt(
        path = out_dir / "eval20-qrels.txt",
        qrels_map_by_split = qrels_all,
        selected_qids_by_split = selected_eval_qids_by_split
    )

    write_queries_jsonl_per_split(
        base_dir = out_dir,
        prefix = "train80",
        queries_map_by_split = queries_map_all,
        selected_qids_by_split = selected_train_qids_by_split
    )
    write_queries_jsonl_per_split(
        base_dir = out_dir,
        prefix = "eval20",
        queries_map_by_split = queries_map_all,
        selected_qids_by_split = selected_eval_qids_by_split
    )
    write_qrels_txt_per_split(
        base_dir = out_dir,
        prefix = "train80",
        qrels_map_by_split = qrels_all,
        selected_qids_by_split = selected_train_qids_by_split
    )
    write_qrels_txt_per_split(
        base_dir = out_dir,
        prefix = "eval20",
        qrels_map_by_split = qrels_all,
        selected_qids_by_split = selected_eval_qids_by_split
    )

    stats["counts"].update({
        "train80_qids_total": sum(len(v) for v in selected_train_qids_by_split.values()),
        "eval20_qids_total":  sum(len(v) for v in selected_eval_qids_by_split.values()),
        "train80_qids_by_split": {k: len(v) for k, v in selected_train_qids_by_split.items()},
        "eval20_qids_by_split":  {k: len(v) for k, v in selected_eval_qids_by_split.items()},
    })

    if missing_gold_log:
        with open(out_dir / "missing_gold.log", "wt", encoding="utf-8") as f:
            for split, qid, did in missing_gold_log:
                f.write(f"{split}\t{qid}\t{did}\n")

    with open(out_dir / "stats.json", "wt", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    eprint("[DONE] Done. Outputs at:", out_dir)
    eprint(f"       - {out_train_gz.name} (Lines written: {stats.get('written_train_lines')})")
    eprint(f"       - {out_eval_gz.name}  (Lines written: {stats.get('written_eval_lines')})")
    eprint("       - train80-queries.jsonl, eval20-queries.jsonl")
    eprint("       - train80-qrels.txt,    eval20-qrels.txt")
    eprint("       - train80-queries-*.jsonl, eval20-queries-*.jsonl (per split)")
    eprint("       - train80-qrels-*.txt,    eval20-qrels-*.txt    (per split)")
    eprint("       - train_ids.txt, eval_ids.txt")
    eprint("       - train_queries_80.jsonl, eval_queries_20.jsonl")
    eprint("       - stats.json, missing_gold.log (if available)")

if __name__ == "__main__":
    main()

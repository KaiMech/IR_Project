#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script validates the generated ToT-2025 subsets and their query/qrels files.

It performs the following checks:

1) Coverage & correctness of qrels per split:
   - For each split (train, dev1, dev2, dev3), it reads the ORIGINAL qrels (e.g., dev1-2025-qrel.txt),
     then reads your split qrels:
       * subsets/train80-qrels-{split}.txt
       * subsets/eval20-qrels-{split}.txt
   - It verifies:
       a) The union of 80%-qids and 20%-qids equals the original qids, except for qids listed in
          subsets/missing_gold.log (whose gold doc is absent from the main corpus). Any further gaps are errors.
       b) For every qid in each split qrels file, the docid matches the one in the original qrels.
       c) For every qid in train80 split qrels, the docid is included in train_ids.txt and also present inside
          train100k-corpus.jsonl.gz. Likewise eval20 split qrels -> eval_ids.txt and eval250k-corpus.jsonl.gz.

2) Subset corpus integrity:
   - Ensures that train100k-corpus.jsonl.gz has EXACTLY 100,000 lines and eval250k-corpus.jsonl.gz EXACTLY 250,000 lines.
     (Change these constants if you built with different targets.)
   - Ensures there are no duplicate IDs within each subset.
   - Confirms every ID in train_ids.txt / eval_ids.txt appears exactly once in the corresponding subset file.
   - Confirms the subset lines are BYTE-FOR-BYTE equal to the original corpus lines (1:1 copying).
     To do this efficiently, it scans the original corpus ONCE and stores a SHA-256 hash for each selected ID,
     then compares hashes of the subset lines to the original hashes (no content rewriting allowed).

3) Reporting and exit code:
   - Prints a clear PASS/FAIL summary for each check.
   - Exits with code 0 if all checks pass, otherwise exits 1.

"""

import argparse
import gzip
import json
import sys
import hashlib
from pathlib import Path
from collections import defaultdict

DEFAULT_DATA_ROOT = "/home/ciwan/tot25/IR_Project/data/tot25"
SUBSETS_DIRNAME   = "subsets"

QRELS = {
    "train": "train-2025-qrel.txt",
    "dev1":  "dev1-2025-qrel.txt",
    "dev2":  "dev2-2025-qrel.txt",
    "dev3":  "dev3-2025-qrel.txt",
}
SPLITS = ["train", "dev1", "dev2", "dev3"]

TRAIN_IDS_FILE = "train_ids.txt"
EVAL_IDS_FILE  = "eval_ids.txt"
TRAIN_GZ       = "train100k-corpus.jsonl.gz"
EVAL_GZ        = "eval250k-corpus.jsonl.gz"

TRAIN_QRELS_SPLIT_PATTERN = "train80-qrels-{}.txt" 
EVAL_QRELS_SPLIT_PATTERN  = "eval20-qrels-{}.txt"

MISSING_LOG = "missing_gold.log"

EXPECTED_TRAIN_SIZE = 100_000
EXPECTED_EVAL_SIZE  = 250_000


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def read_qrels(path: Path):

    mp = {}
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            qid = parts[0]
            did = parts[2]
            mp[qid] = did
    return mp


def read_ids(path: Path):

    s = set()
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x:
                s.add(x)
    return s


def read_missing_log(path: Path):

    res = defaultdict(set)
    if not path.exists():
        return res
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t")
            if len(parts) >= 2:
                sp = parts[0]
                qid = parts[1]
                res[sp].add(qid)
    return res


def parse_gz_corpus_ids_and_hashes(gz_path: Path, wanted_ids: set):

    hashes = {}
    seen_cnt = 0
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            raw = line  
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            did = str(obj.get("id", "")).strip()
            if did in wanted_ids:
                h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                hashes[did] = h
                seen_cnt += 1
                if len(hashes) == len(wanted_ids):
                    break
    return hashes, seen_cnt


def count_gz_lines_and_collect_ids(gz_path: Path):

    cnt = 0
    ids = set()
    ids_in_order = []
    hashes = {}
    dup = False
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            raw = line
            cnt += 1
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            did = str(obj.get("id", "")).strip()
            if did in ids:
                dup = True
            ids.add(did)
            ids_in_order.append(did)
            hashes[did] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return cnt, ids, ids_in_order, hashes, dup


def validate_qrels_coverage(data_root: Path, subsets_dir: Path):

    ok = True
    msgs = []

    missing = read_missing_log(subsets_dir / MISSING_LOG)

    qrels_orig = {}
    qrels_train_by_split = {}
    qrels_eval_by_split  = {}

    for split in SPLITS:

        orig_path = data_root / QRELS[split]
        if not orig_path.exists():
            ok = False
            msgs.append(f"[FAIL] Original qrels missing: {orig_path}")
            continue
        orig = read_qrels(orig_path)
        qrels_orig[split] = orig

        train_split_path = subsets_dir / (TRAIN_QRELS_SPLIT_PATTERN.format(split))
        eval_split_path  = subsets_dir / (EVAL_QRELS_SPLIT_PATTERN.format(split))

        if not train_split_path.exists():
            ok = False
            msgs.append(f"[FAIL] Missing produced qrels: {train_split_path}")
            train_split = {}
        else:
            train_split = read_qrels(train_split_path)

        if not eval_split_path.exists():
            ok = False
            msgs.append(f"[FAIL] Missing produced qrels: {eval_split_path}")
            eval_split = {}
        else:
            eval_split = read_qrels(eval_split_path)

        qrels_train_by_split[split] = train_split
        qrels_eval_by_split[split]  = eval_split

        orig_qids = set(orig.keys())
        combined  = set(train_split.keys()) | set(eval_split.keys())
        missing_in_combined = orig_qids - combined
        missing_expected    = missing.get(split, set())

        if missing_in_combined != missing_expected:
            ok = False
            extra = missing_in_combined - missing_expected
            missing_but_found = missing_expected - missing_in_combined
            msgs.append(f"[FAIL] Qrels coverage mismatch for split '{split}': "
                        f"union(train80+eval20) != original - missing_gold. "
                        f"Extra missing={len(extra)}; Missing-but-found={len(missing_but_found)}")
            if extra:
                msgs.append(f"       Extra-missing QIDs (sample up to 20): {sorted(list(extra))[:20]}")
            if missing_but_found:
                msgs.append(f"       In missing_gold.log but PRESENT in union (sample up to 20): "
                            f"{sorted(list(missing_but_found))[:20]}")
        else:
            msgs.append(f"[PASS] Qrels coverage matches original (minus missing_gold) for split '{split}'. "
                        f"Orig QIDs={len(orig_qids)}, Union={len(combined)}, Missing={len(missing_expected)}")

        wrong_docs = []
        for qid, did in train_split.items():
            if qid in orig and orig[qid] != did:
                wrong_docs.append((qid, did, orig[qid]))
        for qid, did in eval_split.items():
            if qid in orig and orig[qid] != did:
                wrong_docs.append((qid, did, orig[qid]))
        if wrong_docs:
            ok = False
            msgs.append(f"[FAIL] {len(wrong_docs)} qrels doc mismatches in split '{split}'. "
                        f"(sample up to 10)")
            for qid, got, exp in wrong_docs[:10]:
                msgs.append(f"       qid={qid} got={got} expected={exp}")
        else:
            msgs.append(f"[PASS] All qrels docIDs match the original for split '{split}'.")

    return ok, msgs, qrels_train_by_split, qrels_eval_by_split, qrels_orig


def validate_subset_membership_against_qrels(
    subsets_dir: Path,
    train_ids: set,
    eval_ids: set,
    qrels_train_by_split: dict,
    qrels_eval_by_split: dict,
    train_ids_in_gz: set,
    eval_ids_in_gz: set
):

    ok = True
    msgs = []
    # Train
    for split in SPLITS:
        missing_ids = [did for did in qrels_train_by_split.get(split, {}).values() if did not in train_ids]
        missing_in_gz = [did for did in qrels_train_by_split.get(split, {}).values() if did not in train_ids_in_gz]
        if missing_ids:
            ok = False
            msgs.append(f"[FAIL] {len(missing_ids)} train80 qrels docIDs NOT in train_ids.txt (split={split}). "
                        f"(sample up to 10): {missing_ids[:10]}")
        else:
            msgs.append(f"[PASS] All train80 qrels docIDs are in train_ids.txt (split={split}).")
        if missing_in_gz:
            ok = False
            msgs.append(f"[FAIL] {len(missing_in_gz)} train80 qrels docIDs NOT found in train100k-corpus.jsonl.gz "
                        f"(split={split}). (sample up to 10): {missing_in_gz[:10]}")
        else:
            msgs.append(f"[PASS] All train80 qrels docIDs appear in train100k-corpus.jsonl.gz (split={split}).")

    for split in SPLITS:
        missing_ids = [did for did in qrels_eval_by_split.get(split, {}).values() if did not in eval_ids]
        missing_in_gz = [did for did in qrels_eval_by_split.get(split, {}).values() if did not in eval_ids_in_gz]
        if missing_ids:
            ok = False
            msgs.append(f"[FAIL] {len(missing_ids)} eval20 qrels docIDs NOT in eval_ids.txt (split={split}). "
                        f"(sample up to 10): {missing_ids[:10]}")
        else:
            msgs.append(f"[PASS] All eval20 qrels docIDs are in eval_ids.txt (split={split}).")
        if missing_in_gz:
            ok = False
            msgs.append(f"[FAIL] {len(missing_in_gz)} eval20 qrels docIDs NOT found in eval250k-corpus.jsonl.gz "
                        f"(split={split}). (sample up to 10): {missing_in_gz[:10]}")
        else:
            msgs.append(f"[PASS] All eval20 qrels docIDs appear in eval250k-corpus.jsonl.gz (split={split}).")

    return ok, msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                    help="Folder containing original files and the 'subsets' output folder.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    subsets_dir = data_root / SUBSETS_DIRNAME

    train_ids_path = subsets_dir / TRAIN_IDS_FILE
    eval_ids_path  = subsets_dir / EVAL_IDS_FILE
    train_gz_path  = subsets_dir / TRAIN_GZ
    eval_gz_path   = subsets_dir / EVAL_GZ
    corpus_gz_path = data_root / "trec-tot-2025-corpus.jsonl.gz"

    required = [
        train_ids_path, eval_ids_path,
        train_gz_path,  eval_gz_path,
        corpus_gz_path
    ] + [data_root / QRELS[s] for s in SPLITS] + \
        [subsets_dir / TRAIN_QRELS_SPLIT_PATTERN.format(s) for s in SPLITS] + \
        [subsets_dir / EVAL_QRELS_SPLIT_PATTERN.format(s)  for s in SPLITS]

    missing_files = [str(p) for p in required if not p.exists()]
    if missing_files:
        eprint("[FATAL] Missing required files:")
        for p in missing_files:
            eprint("  -", p)
        sys.exit(1)

    train_ids = read_ids(train_ids_path)
    eval_ids  = read_ids(eval_ids_path)

    t_cnt, t_ids_in_gz, t_ids_order, t_hashes, t_dup = count_gz_lines_and_collect_ids(train_gz_path)
    e_cnt, e_ids_in_gz, e_ids_order, e_hashes, e_dup = count_gz_lines_and_collect_ids(eval_gz_path)

    ok_all = True
    report = []

    if t_cnt != EXPECTED_TRAIN_SIZE:
        ok_all = False
        report.append(f"[FAIL] train100k lines = {t_cnt}, expected {EXPECTED_TRAIN_SIZE}.")
    else:
        report.append(f"[PASS] train100k has exactly {EXPECTED_TRAIN_SIZE} lines.")

    if e_cnt != EXPECTED_EVAL_SIZE:
        ok_all = False
        report.append(f"[FAIL] eval250k lines = {e_cnt}, expected {EXPECTED_EVAL_SIZE}.")
    else:
        report.append(f"[PASS] eval250k has exactly {EXPECTED_EVAL_SIZE} lines.")

    if t_dup or len(t_ids_in_gz) != t_cnt:
        ok_all = False
        report.append(f"[FAIL] Duplicated IDs detected in train100k subset.")
    else:
        report.append(f"[PASS] No duplicate IDs within train100k subset.")

    if e_dup or len(e_ids_in_gz) != e_cnt:
        ok_all = False
        report.append(f"[FAIL] Duplicated IDs detected in eval250k subset.")
    else:
        report.append(f"[PASS] No duplicate IDs within eval250k subset.")

    if train_ids != t_ids_in_gz:
        ok_all = False
        missing_in_gz   = train_ids - t_ids_in_gz
        unexpected_in_gz = t_ids_in_gz - train_ids
        report.append(f"[FAIL] train_ids.txt does not match train100k content. "
                      f"missing_in_gz={len(missing_in_gz)}, unexpected_in_gz={len(unexpected_in_gz)}")
    else:
        report.append(f"[PASS] train_ids.txt matches train100k contents.")

    if eval_ids != e_ids_in_gz:
        ok_all = False
        missing_in_gz   = eval_ids - e_ids_in_gz
        unexpected_in_gz = e_ids_in_gz - eval_ids
        report.append(f"[FAIL] eval_ids.txt does not match eval250k content. "
                      f"missing_in_gz={len(missing_in_gz)}, unexpected_in_gz={len(unexpected_in_gz)}")
    else:
        report.append(f"[PASS] eval_ids.txt matches eval250k contents.")

    selected_union = train_ids | eval_ids
    orig_hashes, seen_from_orig = parse_gz_corpus_ids_and_hashes(corpus_gz_path, selected_union)
    if len(orig_hashes) != len(selected_union):
        ok_all = False
        missing_from_orig = selected_union - set(orig_hashes.keys())
        report.append(f"[FAIL] {len(missing_from_orig)} selected IDs not found in original corpus. "
                      f"(sample up to 10): {list(sorted(missing_from_orig))[:10]}")
    else:
        report.append(f"[PASS] All selected IDs were found in the original corpus during scan.")

    bad_train = [did for did in t_ids_in_gz if t_hashes.get(did) != orig_hashes.get(did)]
    bad_eval  = [did for did in e_ids_in_gz if e_hashes.get(did) != orig_hashes.get(did)]
    if bad_train:
        ok_all = False
        report.append(f"[FAIL] {len(bad_train)} lines in train100k differ from original corpus "
                      f"(should be 1:1 copied). Sample up to 10: {bad_train[:10]}")
    else:
        report.append(f"[PASS] All train100k lines are byte-identical to original corpus.")

    if bad_eval:
        ok_all = False
        report.append(f"[FAIL] {len(bad_eval)} lines in eval250k differ from original corpus "
                      f"(should be 1:1 copied). Sample up to 10: {bad_eval[:10]}")
    else:
        report.append(f"[PASS] All eval250k lines are byte-identical to original corpus.")

    ok_cov, msgs_cov, qrels_train_by_split, qrels_eval_by_split, qrels_orig = validate_qrels_coverage(
        data_root, subsets_dir
    )
    report.extend(msgs_cov)
    if not ok_cov:
        ok_all = False

    ok_mem, msgs_mem = validate_subset_membership_against_qrels(
        subsets_dir,
        train_ids,
        eval_ids,
        qrels_train_by_split,
        qrels_eval_by_split,
        t_ids_in_gz,
        e_ids_in_gz
    )
    report.extend(msgs_mem)
    if not ok_mem:
        ok_all = False

    print("\n=== VALIDATION REPORT ===")
    for line in report:
        print(line)

    if ok_all:
        print("\n[OVERALL PASS] All checks succeeded.")
        sys.exit(0)
    else:
        print("\n[OVERALL FAIL] Some checks failed. See messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

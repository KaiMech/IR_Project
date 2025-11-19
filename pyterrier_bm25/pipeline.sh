#!/usr/bin/env bash
set -euo pipefail


TRAIN_CORPUS="/home/ciwan/tot25/IR_Project/data/tot25/subsets/train80/train100k-corpus.jsonl.gz"

TRAIN_Q_DEV1="/home/ciwan/tot25/IR_Project/data/tot25/subsets/train80/train80-queries-dev1.jsonl"
TRAIN_Q_DEV2="/home/ciwan/tot25/IR_Project/data/tot25/subsets/train80/train80-queries-dev2.jsonl"
TRAIN_Q_DEV3="/home/ciwan/tot25/IR_Project/data/tot25/subsets/train80/train80-queries-dev3.jsonl"
TRAIN_Q_TRAIN="/home/ciwan/tot25/IR_Project/data/tot25/subsets/train80/train80-queries-train.jsonl"

EVAL_CORPUS="/home/ciwan/tot25/IR_Project/data/tot25/subsets/eval20/eval250k-corpus.jsonl.gz"

EVAL_Q_DEV1="/home/ciwan/tot25/IR_Project/data/tot25/subsets/eval20/eval20-queries-dev1.jsonl"
EVAL_Q_DEV2="/home/ciwan/tot25/IR_Project/data/tot25/subsets/eval20/eval20-queries-dev2.jsonl"
EVAL_Q_DEV3="/home/ciwan/tot25/IR_Project/data/tot25/subsets/eval20/eval20-queries-dev3.jsonl"
EVAL_Q_TRAIN="/home/ciwan/tot25/IR_Project/data/tot25/subsets/eval20/eval20-queries-train.jsonl"


ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="${ROOT}/pyterrier_bm25"

WORKDIR="${ROOT}/pyterrier_bm25/pyterrier_bm25_output"
INDEX_DIR="${WORKDIR}/indexes"
RUNS_DIR="${WORKDIR}/runs"

mkdir -p "${INDEX_DIR}" "${RUNS_DIR}"

INDEX_TRAIN="${INDEX_DIR}/train100k"
INDEX_EVAL="${INDEX_DIR}/eval250k"
RUNS_TRAIN="${RUNS_DIR}/train100k"
RUNS_EVAL="${RUNS_DIR}/eval250k"

mkdir -p "${INDEX_TRAIN}" "${INDEX_EVAL}" "${RUNS_TRAIN}" "${RUNS_EVAL}"

echo "[1/3] Build PyTerrier index (train100k)"
python3 "${SCRIPTS}/build_index.py" \
  --corpus "${TRAIN_CORPUS}" \
  --index_dir "${INDEX_TRAIN}"

echo "[1/3] Build PyTerrier index (eval250k)"
python3 "${SCRIPTS}/build_index.py" \
  --corpus "${EVAL_CORPUS}" \
  --index_dir "${INDEX_EVAL}"

RUNID_TRAIN="bm25_pyterrier_default_train100k"
TOPK=1000

echo "[2/3] Runs on train100k index (train80 splits)"
python3 "${SCRIPTS}/retrieve_pyterrier_bm25.py" \
  --index_dir "${INDEX_TRAIN}" \
  --queries_path "${TRAIN_Q_TRAIN}" \
  --run_path "${RUNS_TRAIN}/train80-train.pyterrier_bm25.run.gz" \
  --runid "${RUNID_TRAIN}" \
  --topk ${TOPK}

python3 "${SCRIPTS}/retrieve_pyterrier_bm25.py" \
  --index_dir "${INDEX_TRAIN}" \
  --queries_path "${TRAIN_Q_DEV1}" \
  --run_path "${RUNS_TRAIN}/train80-dev1.pyterrier_bm25.run.gz" \
  --runid "${RUNID_TRAIN}" \
  --topk ${TOPK}

python3 "${SCRIPTS}/retrieve_pyterrier_bm25.py" \
  --index_dir "${INDEX_TRAIN}" \
  --queries_path "${TRAIN_Q_DEV2}" \
  --run_path "${RUNS_TRAIN}/train80-dev2.pyterrier_bm25.run.gz" \
  --runid "${RUNID_TRAIN}" \
  --topk ${TOPK}

python3 "${SCRIPTS}/retrieve_pyterrier_bm25.py" \
  --index_dir "${INDEX_TRAIN}" \
  --queries_path "${TRAIN_Q_DEV3}" \
  --run_path "${RUNS_TRAIN}/train80-dev3.pyterrier_bm25.run.gz" \
  --runid "${RUNID_TRAIN}" \
  --topk ${TOPK}

RUNID_EVAL="bm25_pyterrier_default_eval250k"

echo "[3/3] Runs on eval250k index (eval20 splits)"
python3 "${SCRIPTS}/retrieve_pyterrier_bm25.py" \
  --index_dir "${INDEX_EVAL}" \
  --queries_path "${EVAL_Q_TRAIN}" \
  --run_path "${RUNS_EVAL}/eval20-train.pyterrier_bm25.run.gz" \
  --runid "${RUNID_EVAL}" \
  --topk ${TOPK}

python3 "${SCRIPTS}/retrieve_pyterrier_bm25.py" \
  --index_dir "${INDEX_EVAL}" \
  --queries_path "${EVAL_Q_DEV1}" \
  --run_path "${RUNS_EVAL}/eval20-dev1.pyterrier_bm25.run.gz" \
  --runid "${RUNID_EVAL}" \
  --topk ${TOPK}

python3 "${SCRIPTS}/retrieve_pyterrier_bm25.py" \
  --index_dir "${INDEX_EVAL}" \
  --queries_path "${EVAL_Q_DEV2}" \
  --run_path "${RUNS_EVAL}/eval20-dev2.pyterrier_bm25.run.gz" \
  --runid "${RUNID_EVAL}" \
  --topk ${TOPK}

python3 "${SCRIPTS}/retrieve_pyterrier_bm25.py" \
  --index_dir "${INDEX_EVAL}" \
  --queries_path "${EVAL_Q_DEV3}" \
  --run_path "${RUNS_EVAL}/eval20-dev3.pyterrier_bm25.run.gz" \
  --runid "${RUNID_EVAL}" \
  --topk ${TOPK}

echo "Fertig. Runfiles liegen unter:"
echo "  ${RUNS_TRAIN}"
echo "  ${RUNS_EVAL}"

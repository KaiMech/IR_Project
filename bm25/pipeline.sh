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
SCRIPTS="${ROOT}/bm25"


WORKDIR="${ROOT}/bm25/bm25_anserini_output"
JSONL_DIR="${WORKDIR}/jsonl"
INDEX_DIR="${WORKDIR}/indexes"
RUNS_DIR="${WORKDIR}/runs"

mkdir -p "${JSONL_DIR}" "${INDEX_DIR}" "${RUNS_DIR}"

JSONL_TRAIN="${JSONL_DIR}/train100k"
JSONL_EVAL="${JSONL_DIR}/eval250k"
INDEX_TRAIN="${INDEX_DIR}/train100k"
INDEX_EVAL="${INDEX_DIR}/eval250k"
RUNS_TRAIN="${RUNS_DIR}/train100k"
RUNS_EVAL="${RUNS_DIR}/eval250k"

mkdir -p "${JSONL_TRAIN}" "${JSONL_EVAL}" "${INDEX_TRAIN}" "${INDEX_EVAL}" "${RUNS_TRAIN}" "${RUNS_EVAL}"

echo "[1/4] Convert train100k corpus -> JsonCollection"
python3 "${SCRIPTS}/convert_corpus_to_anserini_jsonl.py" \
  --input "${TRAIN_CORPUS}" \
  --output_dir "${JSONL_TRAIN}" \
  --outfile "docs.jsonl"

echo "[1/4] Convert eval250k corpus -> JsonCollection"
python3 "${SCRIPTS}/convert_corpus_to_anserini_jsonl.py" \
  --input "${EVAL_CORPUS}" \
  --output_dir "${JSONL_EVAL}" \
  --outfile "docs.jsonl"

echo "[2/4] Build index (train100k)"
python3 "${SCRIPTS}/build_index.py" \
  --input_dir "${JSONL_TRAIN}" \
  --index_dir "${INDEX_TRAIN}" \
  --threads 8

echo "[2/4] Build index (eval250k)"
python3 "${SCRIPTS}/build_index.py" \
  --input_dir "${JSONL_EVAL}" \
  --index_dir "${INDEX_EVAL}" \
  --threads 8

RUNID_TRAIN="bm25_anserini_k1_0.9_b_0.4_train100k"
TOPK=1000

echo "[3/4] Runs on train100k index (train80 splits)"
python3 "${SCRIPTS}/retrieve_bm25_anserini.py" \
  --index_dir "${INDEX_TRAIN}" \
  --queries_path "${TRAIN_Q_TRAIN}" \
  --run_path "${RUNS_TRAIN}/train80-train.anserini_bm25.run" \
  --runid "${RUNID_TRAIN}" \
  --topk ${TOPK}

python3 "${SCRIPTS}/retrieve_bm25_anserini.py" \
  --index_dir "${INDEX_TRAIN}" \
  --queries_path "${TRAIN_Q_DEV1}" \
  --run_path "${RUNS_TRAIN}/train80-dev1.anserini_bm25.run" \
  --runid "${RUNID_TRAIN}" \
  --topk ${TOPK}

python3 "${SCRIPTS}/retrieve_bm25_anserini.py" \
  --index_dir "${INDEX_TRAIN}" \
  --queries_path "${TRAIN_Q_DEV2}" \
  --run_path "${RUNS_TRAIN}/train80-dev2.anserini_bm25.run" \
  --runid "${RUNID_TRAIN}" \
  --topk ${TOPK}

python3 "${SCRIPTS}/retrieve_bm25_anserini.py" \
  --index_dir "${INDEX_TRAIN}" \
  --queries_path "${TRAIN_Q_DEV3}" \
  --run_path "${RUNS_TRAIN}/train80-dev3.anserini_bm25.run" \
  --runid "${RUNID_TRAIN}" \
  --topk ${TOPK}

RUNID_EVAL="bm25_anserini_k1_0.9_b_0.4_eval250k"

echo "[4/4] Runs on eval250k index (eval20 splits)"
python3 "${SCRIPTS}/retrieve_bm25_anserini.py" \
  --index_dir "${INDEX_EVAL}" \
  --queries_path "${EVAL_Q_TRAIN}" \
  --run_path "${RUNS_EVAL}/eval20-train.anserini_bm25.run" \
  --runid "${RUNID_EVAL}" \
  --topk ${TOPK}

python3 "${SCRIPTS}/retrieve_bm25_anserini.py" \
  --index_dir "${INDEX_EVAL}" \
  --queries_path "${EVAL_Q_DEV1}" \
  --run_path "${RUNS_EVAL}/eval20-dev1.anserini_bm25.run" \
  --runid "${RUNID_EVAL}" \
  --topk ${TOPK}

python3 "${SCRIPTS}/retrieve_bm25_anserini.py" \
  --index_dir "${INDEX_EVAL}" \
  --queries_path "${EVAL_Q_DEV2}" \
  --run_path "${RUNS_EVAL}/eval20-dev2.anserini_bm25.run" \
  --runid "${RUNID_EVAL}" \
  --topk ${TOPK}

python3 "${SCRIPTS}/retrieve_bm25_anserini.py" \
  --index_dir "${INDEX_EVAL}" \
  --queries_path "${EVAL_Q_DEV3}" \
  --run_path "${RUNS_EVAL}/eval20-dev3.anserini_bm25.run" \
  --runid "${RUNID_EVAL}" \
  --topk ${TOPK}

echo "Fertig. Runfiles liegen unter:"
echo "  ${RUNS_TRAIN}"
echo "  ${RUNS_EVAL}"

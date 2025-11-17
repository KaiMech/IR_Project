#!/usr/bin/env bash
set -euo pipefail


export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true
export TORCH_FLOAT32_MATMUL_PRECISION=high
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTORCH_NUM_THREADS=2

START_QUERY_BS=256

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="${ROOT}/dense_retrieval"

TRAIN_CORPUS_SRC="${ROOT}/data/tot25/subsets/train80/train100k-corpus.jsonl.gz"
EVAL_CORPUS_SRC="${ROOT}/data/tot25/subsets/eval20/eval250k-corpus.jsonl.gz"

TRAIN_Q_TRAIN_SRC="${ROOT}/data/tot25/subsets/train80/train80-queries-train.jsonl"
TRAIN_Q_DEV1_SRC="${ROOT}/data/tot25/subsets/train80/train80-queries-dev1.jsonl"
TRAIN_Q_DEV2_SRC="${ROOT}/data/tot25/subsets/train80/train80-queries-dev2.jsonl"
TRAIN_Q_DEV3_SRC="${ROOT}/data/tot25/subsets/train80/train80-queries-dev3.jsonl"

EVAL_Q_TRAIN_SRC="${ROOT}/data/tot25/subsets/eval20/eval20-queries-train.jsonl"
EVAL_Q_DEV1_SRC="${ROOT}/data/tot25/subsets/eval20/eval20-queries-dev1.jsonl"
EVAL_Q_DEV2_SRC="${ROOT}/data/tot25/subsets/eval20/eval20-queries-dev2.jsonl"
EVAL_Q_DEV3_SRC="${ROOT}/data/tot25/subsets/eval20/eval20-queries-dev3.jsonl"

LIR_ROOT="${SCRIPTS}/lir_data"
TRAIN_LIR_DIR="${LIR_ROOT}/train80"
EVAL_LIR_DIR="${LIR_ROOT}/eval20"
mkdir -p "${TRAIN_LIR_DIR}" "${EVAL_LIR_DIR}"

TRAIN_CORPUS_LIR="${TRAIN_LIR_DIR}/train100k-corpus-lir.jsonl"
EVAL_CORPUS_LIR="${EVAL_LIR_DIR}/eval250k-corpus-lir.jsonl"

TRAIN_Q_TRAIN_LIR="${TRAIN_LIR_DIR}/train80-queries-train-lir.jsonl"
TRAIN_Q_DEV1_LIR="${TRAIN_LIR_DIR}/train80-queries-dev1-lir.jsonl"
TRAIN_Q_DEV2_LIR="${TRAIN_LIR_DIR}/train80-queries-dev2-lir.jsonl"
TRAIN_Q_DEV3_LIR="${TRAIN_LIR_DIR}/train80-queries-dev3-lir.jsonl"

EVAL_Q_TRAIN_LIR="${EVAL_LIR_DIR}/eval20-queries-train-lir.jsonl"
EVAL_Q_DEV1_LIR="${EVAL_LIR_DIR}/eval20-queries-dev1-lir.jsonl"
EVAL_Q_DEV2_LIR="${EVAL_LIR_DIR}/eval20-queries-dev2-lir.jsonl"
EVAL_Q_DEV3_LIR="${EVAL_LIR_DIR}/eval20-queries-dev3-lir.jsonl"

OUT_ROOT="${SCRIPTS}/dense_retrieval_output"
RUNS_TRAIN="${OUT_ROOT}/runs/train100k"
RUNS_EVAL="${OUT_ROOT}/runs/eval250k"
mkdir -p "${RUNS_TRAIN}" "${RUNS_EVAL}"

convert_corpus () {
  local SRC="$1"; local DST="$2"
  echo "[conv] corpus -> ${DST}"
  python3 - "$SRC" "$DST" << 'PY'
import sys, json, gzip, os
src, dst = sys.argv[1], sys.argv[2]
def oopen(p):
    return gzip.open(p, 'rt', encoding='utf-8', newline='') if p.endswith('.gz') \
           else open(p, 'r', encoding='utf-8', newline='')
os.makedirs(os.path.dirname(dst), exist_ok=True)
n_in = n_out = 0
with oopen(src) as fin, open(dst, 'w', encoding='utf-8', newline='') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        n_in += 1
        try:
            obj = json.loads(line)
        except Exception:
            continue
        did = obj.get('doc_id') or obj.get('id')
        title = obj.get('title') or ''
        text  = obj.get('text') or ''
        if not did:
            continue
        merged = (f"{title}\n{text}").strip() if title else text.strip()
        if not merged:
            continue
        fout.write(json.dumps({'doc_id': str(did), 'text': merged}, ensure_ascii=False) + "\n")
        n_out += 1
print(f"[conv] corpus read={n_in} wrote={n_out}")
PY
}

convert_queries () {
  local SRC="$1"; local DST="$2"
  echo "[conv] queries -> ${DST}"
  python3 - "$SRC" "$DST" << 'PY'
import sys, json, gzip, os
src, dst = sys.argv[1], sys.argv[2]
def oopen(p):
    return gzip.open(p, 'rt', encoding='utf-8', newline='') if p.endswith('.gz') \
           else open(p, 'r', encoding='utf-8', newline='')
os.makedirs(os.path.dirname(dst), exist_ok=True)
n_in = n_out = 0
with oopen(src) as fin, open(dst, 'w', encoding='utf-8', newline='') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        n_in += 1
        try:
            obj = json.loads(line)
        except Exception:
            continue
        qid = obj.get('query_id') or obj.get('qid') or obj.get('id')
        q   = obj.get('query') or obj.get('text') or obj.get('title')
        if not qid or not q:
            continue
        fout.write(json.dumps({'query_id': str(qid), 'text': str(q), 'query': str(q)}, ensure_ascii=False) + "\n")
        n_out += 1
print(f"[conv] queries read={n_in} wrote={n_out}")
PY
}

rm -f \
  "${TRAIN_CORPUS_LIR}" "${EVAL_CORPUS_LIR}" \
  "${TRAIN_Q_TRAIN_LIR}" "${TRAIN_Q_DEV1_LIR}" "${TRAIN_Q_DEV2_LIR}" "${TRAIN_Q_DEV3_LIR}" \
  "${EVAL_Q_TRAIN_LIR}"  "${EVAL_Q_DEV1_LIR}"  "${EVAL_Q_DEV2_LIR}"  "${EVAL_Q_DEV3_LIR}"

convert_corpus  "${TRAIN_CORPUS_SRC}" "${TRAIN_CORPUS_LIR}"
convert_corpus  "${EVAL_CORPUS_SRC}"  "${EVAL_CORPUS_LIR}"

convert_queries "${TRAIN_Q_TRAIN_SRC}" "${TRAIN_Q_TRAIN_LIR}"
convert_queries "${TRAIN_Q_DEV1_SRC}"  "${TRAIN_Q_DEV1_LIR}"
convert_queries "${TRAIN_Q_DEV2_SRC}"  "${TRAIN_Q_DEV2_LIR}"
convert_queries "${TRAIN_Q_DEV3_SRC}"  "${TRAIN_Q_DEV3_LIR}"
convert_queries "${EVAL_Q_TRAIN_SRC}"  "${EVAL_Q_TRAIN_LIR}"
convert_queries "${EVAL_Q_DEV1_SRC}"   "${EVAL_Q_DEV1_LIR}"
convert_queries "${EVAL_Q_DEV2_SRC}"   "${EVAL_Q_DEV2_LIR}"
convert_queries "${EVAL_Q_DEV3_SRC}"   "${EVAL_Q_DEV3_LIR}"

K=1000

echo "[1/2] Runs on TRAIN 100k-Corpus (train80-splits) with QUERY_BS=${START_QUERY_BS}"

declare -A TRAIN_SPLITS
TRAIN_SPLITS["train80-train"]="${TRAIN_Q_TRAIN_LIR}"
TRAIN_SPLITS["train80-dev1"]="${TRAIN_Q_DEV1_LIR}"
TRAIN_SPLITS["train80-dev2"]="${TRAIN_Q_DEV2_LIR}"
TRAIN_SPLITS["train80-dev3"]="${TRAIN_Q_DEV3_LIR}"

for NAME in "${!TRAIN_SPLITS[@]}"; do
  QPATH="${TRAIN_SPLITS[$NAME]}"
  echo "  -> ${NAME} (${QPATH})"

  python3 "${SCRIPTS}/baseline_dense_local.py" \
    --corpus "${TRAIN_CORPUS_LIR}" \
    --queries "${QPATH}" \
    --out_dir "${RUNS_TRAIN}" \
    --k ${K} \
    --query-batch-size "${START_QUERY_BS}" \
    --gpu

  QSTEM="$(basename "${QPATH}" .jsonl)"
  RUN_SRC="${RUNS_TRAIN}/${QSTEM}.run"
  RUN_DST="${RUNS_TRAIN}/${NAME}.dense_retrieval.run.gz"

  if [[ -f "${RUN_SRC}" ]]; then
    gzip -f "${RUN_SRC}"
    mv -f "${RUN_SRC}.gz" "${RUN_DST}"
    echo "     OK: ${RUN_DST}"
  else
    echo "     WARN: no run file found for ${NAME} (expected: ${RUN_SRC})"
  fi
done

echo "[2/2] Runs on EVAL 250k-Corpus (eval20-splits) with QUERY_BS=${START_QUERY_BS}"

declare -A EVAL_SPLITS
EVAL_SPLITS["eval20-train"]="${EVAL_Q_TRAIN_LIR}"
EVAL_SPLITS["eval20-dev1"]="${EVAL_Q_DEV1_LIR}"
EVAL_SPLITS["eval20-dev2"]="${EVAL_Q_DEV2_LIR}"
EVAL_SPLITS["eval20-dev3"]="${EVAL_Q_DEV3_LIR}"

for NAME in "${!EVAL_SPLITS[@]}"; do
  QPATH="${EVAL_SPLITS[$NAME]}"
  echo "  -> ${NAME} (${QPATH})"

  python3 "${SCRIPTS}/baseline_dense_local.py" \
    --corpus "${EVAL_CORPUS_LIR}" \
    --queries "${QPATH}" \
    --out_dir "${RUNS_EVAL}" \
    --k ${K} \
    --query-batch-size "${START_QUERY_BS}" \
    --gpu

  QSTEM="$(basename "${QPATH}" .jsonl)"
  RUN_SRC="${RUNS_EVAL}/${QSTEM}.run"
  RUN_DST="${RUNS_EVAL}/${NAME}.dense_retrieval.run.gz"

  if [[ -f "${RUN_SRC}" ]]; then
    gzip -f "${RUN_SRC}"
    mv -f "${RUN_SRC}.gz" "${RUN_DST}"
    echo "     OK: ${RUN_DST}"
  else
    echo "     WARN: no run file found for ${NAME} (expected: ${RUN_SRC})"
  fi
done

echo
echo "Done. Everything is now located under:"
echo "  LIR data:   ${LIR_ROOT}"
echo "  Runs train: ${RUNS_TRAIN}"
echo "  Runs eval:  ${RUNS_EVAL}"

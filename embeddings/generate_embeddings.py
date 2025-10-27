import os
import json
from pathlib import Path
import torch
import numpy as np
from transformers import DPRContextEncoder, AutoTokenizer, BertTokenizerFast
from contextlib import nullcontext
import math
from tqdm.auto import tqdm
from multiprocessing import Process, Queue, cpu_count
from threading import Thread
import queue
import sys

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OUTDIR = Path('context_embeddings_shard05')
OUTDIR.mkdir(exist_ok=True)

CORPUS_JSONL = '/kaggle/input/ir-project-shard-01/corpus_sample_2000000-2499999.jsonl'

model = DPRContextEncoder.from_pretrained('facebook--dpr-ctx_encoder-single-nq-base')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', use_fast=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

def iter_corpus():
    with open(CORPUS_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            yield str(obj['id']), obj.get('text', '')

def _tokenize_for_encoder(texts, tokenizer, *, MAX_LEN=128, STRIDE=0, exclude_special_in_weight=True):
    pad_mode = "longest" if STRIDE == 0 else "max_length"
    toks = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=pad_mode,
        max_length=MAX_LEN,
        stride=STRIDE,
        return_overflowing_tokens=(STRIDE > 0),
        return_attention_mask=True,
        return_length=True,
        return_special_tokens_mask=exclude_special_in_weight,
    )

    def as_tensor(x, dtype=None):
        if x is None: return None
        t = torch.as_tensor(x)
        return t.to(dtype) if dtype is not None else t

    tokenized = {
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
        "length": as_tensor(toks.get("length"), torch.long),
        "special_tokens_mask": toks.get("special_tokens_mask"),
    }
    if "overflow_to_sample_mapping" in toks:
        doc_map = as_tensor(toks.pop("overflow_to_sample_mapping"), torch.long)
    else:
        doc_map = torch.arange(len(texts), dtype=torch.long)
    return tokenized, doc_map


@torch.inference_mode()
def encode_doc_batch_fast(
    tokenized, doc_map_cpu, model,
    *, CHUNK_BATCH=32, DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    pooling="auto", weight_by_tokens=True, ACCUM_ON_CPU=True, amp_dtype="bf16",
):
    input_ids = tokenized["input_ids"]; attention_mask = tokenized["attention_mask"]
    num_chunks = int(input_ids.size(0))
    D = int(doc_map_cpu.max().item()) + 1 if num_chunks > 0 else 0
    if num_chunks == 0 or D == 0:
        hidden = int(getattr(model.config, "projection_dim", None) or getattr(model.config, "hidden_size", 0) or 0)
        return np.zeros((0, hidden), dtype=np.float32)

    if tokenized.get("length") is not None:
        valid_tokens_cpu = tokenized["length"].view(-1, 1)
    else:
        valid_tokens_cpu = attention_mask.sum(dim=1, keepdim=True).to(torch.long)
    if tokenized.get("special_tokens_mask") is not None:
        stm = tokenized["special_tokens_mask"]
        nonpad = attention_mask
        nonspecial = (torch.as_tensor(stm) == 0).to(nonpad.dtype)
        weight_tokens_cpu = (nonpad * nonspecial).sum(dim=1, keepdim=True).to(torch.long)
    else:
        weight_tokens_cpu = valid_tokens_cpu

    device_accum = torch.device("cpu") if ACCUM_ON_CPU else DEVICE
    hidden = int(
        getattr(model.config, "projection_dim", None)
        or getattr(model.config, "hidden_size", None)
        or model(input_ids=input_ids[:1].to(DEVICE), attention_mask=attention_mask[:1].to(DEVICE)).last_hidden_state.shape[-1]
    )
    sums = torch.zeros(D, hidden, dtype=torch.float32, device=device_accum)
    counts = torch.zeros(D, 1, dtype=torch.float32, device=device_accum)

    if next(model.parameters()).device != DEVICE:
        model = model.to(DEVICE).eval()

    use_amp = (DEVICE.type == "cuda") and (str(amp_dtype).lower() in ("bf16","bfloat16","fp16","float16"))
    amp_dtype_resolved = torch.bfloat16 if "bf" in str(amp_dtype).lower() else torch.float16
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype_resolved) if use_amp else nullcontext()

    for s in range(0, num_chunks, CHUNK_BATCH):
        e = min(s + CHUNK_BATCH, num_chunks)
        ids_s  = input_ids[s:e].pin_memory().to(DEVICE, non_blocking=True)
        attn_s = attention_mask[s:e].pin_memory().to(DEVICE, non_blocking=True)
        vtok_s = weight_tokens_cpu[s:e].to(torch.float32)
        d_idx  = doc_map_cpu[s:e]

        with amp_ctx:
            out = model(input_ids=ids_s, attention_mask=attn_s)
            vecs = getattr(out, "pooler_output", None)
            if vecs is None and pooling in ("cls","auto"):
                last = getattr(out, "last_hidden_state", None)
                if last is not None: vecs = last[:,0,:]
            if vecs is None:
                last = out.last_hidden_state
                summed = (last * attn_s.unsqueeze(-1)).sum(dim=1)
                denom  = attn_s.sum(dim=1, keepdim=True).clamp_min(1)
                vecs = summed / denom
            vecs = vecs.to(torch.float32)

        w = vtok_s.clamp_min(1.0) if weight_by_tokens else torch.ones_like(vtok_s)
        vecs_cpu = vecs.detach().to("cpu", non_blocking=True)
        w_cpu    = w.detach().to("cpu", non_blocking=True)
        d_idx_cpu= d_idx.to("cpu", non_blocking=True).long()
        sums.index_add_(0, d_idx_cpu, vecs_cpu * w_cpu)
        counts.index_add_(0, d_idx_cpu, w_cpu)

    embs = sums / counts.clamp_min(1.0)
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    out = embs.detach().cpu().numpy().astype("float32")
    if out.ndim == 1: out = out[None, :]
    return out


def build_embeddings_fast(
    *, iter_corpus, tokenizer, model, OUTDIR,
    DOCS_PER_CALL=256, SHARD_SIZE_DOCS=50_000, MAX_LEN=128, STRIDE=0, CHUNK_BATCH=32, amp_dtype="bf16",
    exclude_special_in_weight=True,
):
    OUTDIR.mkdir(parents=True, exist_ok=True)

    def bucket_idx(text):
        n = len(text)
        if n < 300: return 0
        elif n < 1200: return 1
        else: return 2

    q = queue.Queue(maxsize=32)

    def producer():
        queues = {0: [], 1: [], 2: []}
        ids    = {0: [], 1: [], 2: []}
        for doc_id, text in iter_corpus():
            b = bucket_idx(text)
            queues[b].append(text); ids[b].append(str(doc_id))
            if len(queues[b]) >= DOCS_PER_CALL:
                toks, doc_map = _tokenize_for_encoder(queues[b], tokenizer, MAX_LEN=MAX_LEN, STRIDE=STRIDE,
                                                      exclude_special_in_weight=exclude_special_in_weight)
                q.put((ids[b], toks, doc_map))
                queues[b].clear(); ids[b].clear()
        for b in (0,1,2):
            if queues[b]:
                toks, doc_map = _tokenize_for_encoder(queues[b], tokenizer, MAX_LEN=MAX_LEN, STRIDE=STRIDE,
                                                      exclude_special_in_weight=exclude_special_in_weight)
                q.put((ids[b], toks, doc_map))
        q.put(None)

    t = Thread(target=producer, daemon=True); t.start()

    shard_idx = 0
    buffer_blocks, buffer_ids = [], []
    is_tty = sys.stderr.isatty()
    pbar = tqdm(
        desc="Encoding passages",
        total=500000,
        dynamic_ncols=True,
        mininterval=0.3,
        smoothing=0.0,
        leave=True,
        file=sys.stdout,
        disable=False,
    )

    def maybe_flush():
        nonlocal shard_idx, buffer_blocks, buffer_ids
        rows = sum(b.shape[0] for b in buffer_blocks if isinstance(b, np.ndarray) and b.ndim == 2)
        if rows >= SHARD_SIZE_DOCS:
            V = np.concatenate(buffer_blocks, axis=0).astype(np.float32)
            np.save(OUTDIR / f"emb_{shard_idx:05d}.npy", V)
            (OUTDIR / f"emb_{shard_idx:05d}.txt").write_text("\n".join(buffer_ids), encoding="utf-8")
            shard_idx += 1
            buffer_blocks.clear(); buffer_ids.clear()

    while True:
        item = q.get()
        if item is None:
            break
        ids_batch, toks, doc_map = item
        V = encode_doc_batch_fast(
            tokenized=toks, doc_map_cpu=doc_map, model=model,
            CHUNK_BATCH=CHUNK_BATCH, amp_dtype=amp_dtype
        )
        if not isinstance(V, np.ndarray): V = np.asarray(V, dtype=np.float32)
        if V.ndim == 1: V = V[None, :]
        if V.shape[0] == 0:
            continue
        if len(ids_batch) != V.shape[0]:
            ids_batch = ids_batch[:V.shape[0]]
        buffer_blocks.append(V); buffer_ids.extend(ids_batch)
        pbar.update(V.shape[0])
        maybe_flush()

    if buffer_blocks:
        V = np.concatenate(buffer_blocks, axis=0).astype(np.float32)
        np.save(OUTDIR / f"emb_{shard_idx:05d}.npy", V)
        (OUTDIR / f"emb_{shard_idx:05d}.txt").write_text("\n".join(buffer_ids), encoding="utf-8")

    pbar.close()
    t.join()
    print("Done building embeddings (threaded producer).")

build_embeddings_fast(
    iter_corpus=iter_corpus,     
    tokenizer=tokenizer,
    model=model,
    OUTDIR=OUTDIR,
    DOCS_PER_CALL=64,
    MAX_LEN=256,
    STRIDE=0,
    amp_dtype="bf16",
)
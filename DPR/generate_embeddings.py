from __future__ import annotations
import os
import json
from pathlib import Path
import torch
import numpy as np
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, AutoConfig
from contextlib import nullcontext
import math
from tqdm.auto import tqdm
from multiprocessing import Process, Queue, cpu_count
from threading import Thread
import queue
import sys
import argparse
import safetensors.torch

def iter_corpus(corpus_path):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            yield str(obj['query_id']), obj.get('query', '')

def _tokenize_for_encoder(
    texts,
    tokenizer,
    *,
    MAX_LEN=128,
    STRIDE=0,
    exclude_special_in_weight=True,
):
    """
    Tokenize a batch of texts for DPR-style encoders.

    - Uses slow DPR tokenizer with overflow.
    - Does NOT rely on return_tensors='pt' inside HF to avoid the
      'overflowing tokens of different lengths' error.
    - Pads everything to fixed MAX_LEN.
    """
    # Let HF do truncation + overflow, but NOT tensor conversion / padding
    toks = tokenizer(
        texts,
        truncation=True,
        padding=False,                 # raw lists
        max_length=MAX_LEN,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_attention_mask=True,
        return_length=True,
        return_special_tokens_mask=exclude_special_in_weight,
        return_tensors=None,           # critical: avoid ragged → tensor inside HF
    )

    input_ids_list = toks["input_ids"]          # list[list[int]]
    attn_mask_list = toks["attention_mask"]     # list[list[int]]
    stm_list       = toks.get("special_tokens_mask", None)

    # Fixed length for this model; keeps things simple and regular
    seq_len = MAX_LEN
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0  # DPR/BERT vocab: 0 is usually [PAD]

    def pad(seq, pad_value):
        if len(seq) >= seq_len:
            return seq[:seq_len]
        return seq + [pad_value] * (seq_len - len(seq))

    padded_input_ids = [pad(ids, pad_id) for ids in input_ids_list]
    padded_attn_mask = [pad(mask, 0) for mask in attn_mask_list]

    if stm_list is not None:
        # Mark padding as "special" (1), consistent with HF behavior
        padded_stm = [pad(stm, 1) for stm in stm_list]
    else:
        padded_stm = None

    def as_tensor(x, dtype=None):
        if x is None:
            return None
        t = torch.as_tensor(x)
        return t.to(dtype) if dtype is not None else t

    tokenized = {
        "input_ids": as_tensor(padded_input_ids, torch.long),
        "attention_mask": as_tensor(padded_attn_mask, torch.long),
        "length": as_tensor(toks.get("length"), torch.long),  # original valid length
        "special_tokens_mask": as_tensor(padded_stm, torch.long) if padded_stm is not None else None,
    }

    # overflow_to_sample_mapping tells you which chunk came from which original text
    if "overflow_to_sample_mapping" in toks:
        doc_map = as_tensor(toks.pop("overflow_to_sample_mapping"), torch.long)
    else:
        doc_map = torch.arange(len(texts), dtype=torch.long)

    return tokenized, doc_map


@torch.inference_mode()
def encode_doc_batch_fast(
    tokenized,
    doc_map_cpu,
    model,
    *,
    CHUNK_BATCH=256,
    DEVICE=torch.device("cuda"),
    pooling="auto",
    weight_by_tokens=True,
    ACCUM_ON_CPU=False,
    amp_dtype="fp16",
):
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    num_chunks = int(input_ids.size(0))
    D = int(doc_map_cpu.max().item()) + 1 if num_chunks > 0 else 0

    if num_chunks == 0 or D == 0:
        probe_model = getattr(model, "module", model)
        hidden = int(
            getattr(probe_model.config, "projection_dim", None)
            or getattr(probe_model.config, "hidden_size", 0)
            or 0
        )
        return np.zeros((0, hidden), dtype=np.float32)

    # per-chunk token counts / weights (CPU)
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

    probe_model = getattr(model, "module", model)
    hidden = int(
        getattr(probe_model.config, "projection_dim", None)
        or getattr(probe_model.config, "hidden_size", None)
        or probe_model(
            input_ids=input_ids[:1].to(DEVICE),
            attention_mask=attention_mask[:1].to(DEVICE),
        ).last_hidden_state.shape[-1]
    )

    sums = torch.zeros(D, hidden, dtype=torch.float32, device=device_accum)
    counts = torch.zeros(D, 1, dtype=torch.float32, device=device_accum)

    use_amp = (
        DEVICE.type == "cuda"
        and (str(amp_dtype).lower() in ("bf16", "bfloat16", "fp16", "float16"))
    )
    amp_dtype_resolved = torch.bfloat16 if "bf" in str(amp_dtype).lower() else torch.float16
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype_resolved)
        if use_amp
        else nullcontext()
    )

    # microbatches for GPU
    for s in range(0, num_chunks, CHUNK_BATCH):
        e = min(s + CHUNK_BATCH, num_chunks)

        ids_s  = input_ids[s:e]
        attn_s = attention_mask[s:e]
        vtok_s = weight_tokens_cpu[s:e].to(torch.float32)
        d_idx  = doc_map_cpu[s:e].long()

        ids_s  = ids_s.to(DEVICE, non_blocking=True)
        attn_s = attn_s.to(DEVICE, non_blocking=True)

        with amp_ctx:
            out = model(input_ids=ids_s, attention_mask=attn_s)

            vecs = getattr(out, "pooler_output", None)
            if vecs is None and pooling in ("cls", "auto"):
                last = getattr(out, "last_hidden_state", None)
                if last is not None:
                    vecs = last[:, 0, :]
            if vecs is None:
                last = out.last_hidden_state
                summed = (last * attn_s.unsqueeze(-1)).sum(dim=1)
                denom  = attn_s.sum(dim=1, keepdim=True).clamp_min(1)
                vecs   = summed / denom  # mean pooling

            vecs = vecs.to(torch.float32)

        if ACCUM_ON_CPU:
            vecs_accum = vecs.detach().to("cpu", non_blocking=True)
            w_accum    = vtok_s.detach().to("cpu", non_blocking=True)
            d_idx_acc  = d_idx.to("cpu", non_blocking=True)
        else:
            vecs_accum = vecs
            w_accum    = vtok_s.to(DEVICE, non_blocking=True)
            d_idx_acc  = d_idx.to(DEVICE, non_blocking=True)

        w_accum = w_accum.clamp_min(1.0) if weight_by_tokens else torch.ones_like(w_accum)

        sums.index_add_(0, d_idx_acc, vecs_accum * w_accum)
        counts.index_add_(0, d_idx_acc, w_accum)

    embs = sums / counts.clamp_min(1.0)
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)

    embs_cpu = embs.detach().to("cpu")
    out = embs_cpu.numpy().astype("float32")
    if out.ndim == 1:
        out = out[None, :]
    return out


def _batch_iter_corpus(corpus_path, iter_corpus, batch_size):
    """
    Group (doc_id, text) from iter_corpus() into batches.
    """
    buf_ids, buf_texts = [], []
    for doc_id, text in iter_corpus(corpus_path):
        buf_ids.append(str(doc_id))
        buf_texts.append(text)
        if len(buf_ids) >= batch_size:
            yield buf_ids, buf_texts
            buf_ids, buf_texts = [], []
    if buf_ids:
        yield buf_ids, buf_texts


def build_embeddings_fast(
    *,
    corpus_path,
    iter_corpus,                # callable or generator yielding (doc_id, text)
    tokenizer,
    model,
    OUTDIR: Path,
    DOCS_PER_CALL=256,
    SHARD_SIZE_DOCS=50_000,
    MAX_LEN=128,
    STRIDE=0,
    CHUNK_BATCH=256,
    amp_dtype="bf16",
    exclude_special_in_weight=True,
    DEVICE=torch.device("cuda"),
):
    """
    Build DPR-style passage embeddings in shards, in a straightforward,
    synchronous pipeline (no threads / queues).

    Guarantees:
    - Each row in emb_XXXXX.npy lines up 1:1 with each line in emb_XXXXX.txt.
    """

    OUTDIR.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    buffer_blocks = []  # list of np.ndarray [N_batch, dim]
    buffer_ids    = []  # list of str

    def flush_to_disk():
        nonlocal shard_idx, buffer_blocks, buffer_ids
        if not buffer_blocks:
            return

        V = np.concatenate(buffer_blocks, axis=0).astype(np.float32)
        assert V.shape[0] == len(buffer_ids), (
            f"INTERNAL ERROR before flush: {V.shape[0]} embeddings vs "
            f"{len(buffer_ids)} ids"
        )

        np.save(OUTDIR / f"emb_{shard_idx:05d}.npy", V)
        with open(OUTDIR / f"emb_{shard_idx:05d}.txt", "w", encoding="utf-8") as f:
            for _id in buffer_ids:
                f.write(_id + "\n")

        shard_idx += 1
        buffer_blocks = []
        buffer_ids    = []

    pbar = tqdm(
        desc="Encoding passages",
        total=None,
        dynamic_ncols=True,
        mininterval=0.3,
        smoothing=0.0,
        leave=True,
        file=sys.stdout,
        disable=False,
    )

    # loop over batches of raw docs
    for ids_batch, texts_batch in _batch_iter_corpus(corpus_path, iter_corpus, DOCS_PER_CALL):
        # tokenization (with overflow)
        toks, doc_map = _tokenize_for_encoder(
            texts_batch,
            tokenizer,
            MAX_LEN=MAX_LEN,
            STRIDE=STRIDE,
            exclude_special_in_weight=exclude_special_in_weight,
        )

        # encode; returns one embedding per *original* doc (using doc_map)
        V = encode_doc_batch_fast(
            tokenized=toks,
            doc_map_cpu=doc_map,
            model=model,
            CHUNK_BATCH=CHUNK_BATCH,
            DEVICE=DEVICE,
            amp_dtype=amp_dtype,
        )

        if not isinstance(V, np.ndarray):
            V = np.asarray(V, dtype=np.float32)
        if V.ndim == 1:
            V = V[None, :]
        if V.shape[0] == 0:
            continue

        # We expect exactly one embedding per original document
        if len(ids_batch) != V.shape[0]:
            raise RuntimeError(
                "Embedding/doc_id misalignment detected.\n"
                f"- len(ids_batch) = {len(ids_batch)}\n"
                f"- V.shape[0]     = {V.shape[0]}\n"
                f"- doc_map may not have been reduced to 1 embedding per doc."
            )

        buffer_blocks.append(V.astype(np.float32, copy=False))
        buffer_ids.extend(ids_batch)
        pbar.update(V.shape[0])

        current_rows = sum(block.shape[0] for block in buffer_blocks)
        if current_rows >= SHARD_SIZE_DOCS:
            flush_to_disk()

    flush_to_disk()
    pbar.close()

    print("Done building embeddings.")

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings using DPR models')
    parser.add_argument(
        '--finetuned', 
        action='store_true', 
        default=False, 
        help='Use a finetuned model instead of a pretrained one from HF hub. If --finetuned is present, it is interpreted as active.\
            If you do not want to use a finetuned model, simply omit this argument.'
    )
    parser.add_argument(
        '--finetuned_model_path', 
        type=str, 
        default=None, 
        help='Path to the finetuned model, if finetuning is selected'
    )
    parser.add_argument(
        '--use_query_encoder', 
        action='store_true', 
        default=False, 
        help='Indicates whether the question encoder (True) or context encoder (False) is used. If --use_query_encoder is present,\
            it is interpreted as True. If you want to use the context encoder, simply omit this argument.'
    )
    parser.add_argument('--corpus_path', type=str, help='Path to the jsonl file containing the corpus')
    parser.add_argument('--target_path', type=str, help='Location of the resulting embeddings')
    parser.add_argument('--device', type=str, default=None, help='cpu or cuda')
    
    args = parser.parse_args()
    
    print(f'Running with args\n\tfinetunded={args.finetuned}, path={args.finetuned_model_path}\
        \n\tmodel={'query_encoder' if args.use_query_encoder else 'context_encoder'}\
        \n\tcorpus_path={args.corpus_path}\n\ttarget_path={args.target_path}\n\tdevice={args.device}')
    
    q_name = "facebook/dpr-question_encoder-single-nq-base"
    c_name = "facebook/dpr-ctx_encoder-single-nq-base"
    
    OUTDIR = Path(args.target_path)
    OUTDIR.mkdir(exist_ok=True)

    CORPUS_JSONL = Path(args.corpus_path)
    
    finetuned = args.finetuned
    use_query = args.use_query_encoder
    
    if finetuned:
        if args.finetuned_model_path is None:
            parser.error('--finetuned_model_path is required if --finetuned=True')
        else:
            finetuned_path = Path(args.finetuned_model_path)
            dirs = []
            for dir in finetuned_path.iterdir():
                dirs.append(dir)
            config_files = [p for p in dirs if p.name == 'config.json']
            tensor_files = [p for p in dirs if p.name == 'model.safetensors']
            if len(config_files) > 0 and len(tensor_files) > 0:
                config_path = config_files[0]
                tensor_path = tensor_files[0]
                config = AutoConfig.from_pretrained(config_path)
                state_dict = safetensors.torch.load_file(tensor_path)
                if use_query:
                    model = DPRQuestionEncoder(config)
                    model.load_state_dict(state_dict)
                    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_name)
                else:
                    model = DPRContextEncoder(config)
                    model.load_state_dict(state_dict)
                    tokenizer = DPRContextEncoderTokenizer.from_pretrained(c_name)
            else:
                parser.error('The directory for the finetuned model needs to contain files config.json and model.safetensors')
    else:
        if use_query:
            model = DPRQuestionEncoder.from_pretrained(q_name)
            tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_name)
        else:
            model = DPRContextEncoder.from_pretrained(c_name)
            tokenizer = DPRContextEncoderTokenizer.from_pretrained(c_name)
                
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
        
    if args.device is None:
        main_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        main_device = torch.device(args.device)
    

    model = model.to(main_device)
    model.eval()

    if torch.cuda.device_count() > 1:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    else:
        print("Single GPU / CPU mode")
        
    build_embeddings_fast(
        corpus_path=CORPUS_JSONL,
        iter_corpus=iter_corpus,
        tokenizer=tokenizer,
        model=model,
        OUTDIR=OUTDIR,
        DOCS_PER_CALL=16,
        MAX_LEN=256,
        STRIDE=32,
        CHUNK_BATCH=16,
        amp_dtype="fp16",
        DEVICE=main_device,
    )

    
if __name__ == '__main__':
    main()
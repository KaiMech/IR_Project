#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Tuple
from contextlib import nullcontext

import click
import torch
from transformers import AutoTokenizer, AutoModel


os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.set_float32_matmul_precision("high")

MODEL_NAME = "sbhargav/baseline-distilbert-tot24"
MAX_LENGTH = 512  


def _open_text(path: Path):
    """Öffnet .jsonl oder .jsonl.gz transparent im Textmodus."""
    import gzip

    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def load_corpus(corpus_path: Path) -> Tuple[List[str], List[str]]:
    """Liest LIR-Corpus (.jsonl) -> (doc_ids, texts)."""
    doc_ids: List[str] = []
    texts: List[str] = []

    with _open_text(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            did = str(obj["doc_id"])
            text = str(obj["text"])
            doc_ids.append(did)
            texts.append(text)

    print(f"[data] Corpus loaded: {len(doc_ids)} documents")
    return doc_ids, texts


def load_queries(queries_path: Path) -> Tuple[List[str], List[str]]:
    """Liest LIR-Queries (.jsonl) -> (query_ids, texts)."""
    qids: List[str] = []
    texts: List[str] = []

    with _open_text(queries_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id") or obj.get("qid") or obj.get("id"))
            qtext = obj.get("query") or obj.get("text") or obj.get("title")
            if not qid or not qtext:
                continue
            qids.append(qid)
            texts.append(str(qtext))

    print(f"[data] Queries loaded: {len(qids)} queries")
    return qids, texts


def encode_texts_auto_bs(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    target_batch_size: int,
    desc: str = "",
) -> torch.Tensor:
    """
    Encodiert Texte zu normalisierten Embeddings mittels mean pooling + L2-Norm.
    Versucht mehrere Batchgrößen (target, target/2, ..., >=32) und reduziert bei CUDA OOM.
    Nutzt auf GPU mixed precision (autocast), wie Lightning precision=16-mixed.
    """
    n = len(texts)
    if n == 0:
        return torch.empty(0, model.config.hidden_size, device=device)

    tries = []
    bs = max(1, int(target_batch_size))
    while bs >= 32:
        tries.append(bs)
        bs //= 2
    if not tries:
        tries = [min(32, max(1, n))]

    last_err: Exception | None = None

    for bs_try in tries:
        print(f"[encode-auto] Trying batch_size={bs_try} for {desc}", flush=True)
        all_embs: List[torch.Tensor] = []
        model.eval()
        try:
            start = 0
            while start < n:
                end = min(start + bs_try, n)
                batch_texts = texts[start:end]
                print(f"[encode] {desc} {start}/{n} .. {end}/{n}", flush=True)

                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                if device.type == "cuda":
                    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
                else:
                    amp_ctx = nullcontext()

                with torch.no_grad(), amp_ctx:
                    outputs = model(**enc)
                    last_hidden = outputs.last_hidden_state 
                    mask = enc["attention_mask"].unsqueeze(-1) 

                    masked = last_hidden * mask
                    sums = masked.sum(dim=1)  
                    lengths = mask.sum(dim=1).clamp(min=1)  
                    mean = sums / lengths

                    mean = torch.nn.functional.normalize(mean, p=2, dim=1)

                    mean = mean.to(torch.float32)

                all_embs.append(mean)

                del enc, outputs, last_hidden, mask, masked, sums, lengths, mean
                start = end

            embs = torch.cat(all_embs, dim=0)
            assert embs.shape[0] == n
            print(f"[encode-auto] SUCCESS with batch_size={bs_try} for {desc}")
            return embs

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and "cuda" in msg and device.type == "cuda":
                last_err = e
                print(
                    f"[encode-auto] CUDA OOM with batch_size={bs_try}, "
                    f"trying smaller batch size...",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise

    raise SystemExit(f"Encoding failed for all batch sizes. Last error:\n{last_err}")


def run_search(
    corpus_path: Path,
    queries_path: Path,
    out_dir: Path,
    k: int,
    query_batch_size: int,
    use_gpu: bool,
):
    """
    Führt Dense-Retrieval durch:
    - lädt DistilBERT Checkpoint
    - encodiert alle Docs & Queries (mit GPU + Batch-Fallback)
    - nutzt einen einfachen Cache für Doc-Embeddings pro Corpus + Out-Dir
    - berechnet Cosine-Similarity
    - schreibt TREC-Runfile in out_dir
    """
    corpus_path = corpus_path.resolve()
    queries_path = queries_path.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[search] corpus={corpus_path}")
    print(f"[search] queries={queries_path}")

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"[search] device={device}")

    print(f"[model] Lade {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)

    doc_ids, doc_texts = load_corpus(corpus_path)
    q_ids, q_texts = load_queries(queries_path)

    if not doc_ids or not q_ids:
        raise SystemExit("[error] Keine Docs oder Queries gefunden – kann nichts suchen.")

    cache_dir = out_dir / "_doc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"{corpus_path.stem}__{MODEL_NAME.replace('/', '_')}.pt"
    cache_path = cache_dir / cache_name

    if cache_path.exists():
        print(f"[cache] Loading doc embeddings from {cache_path}")
        cache_obj = torch.load(cache_path, map_location="cpu")
        cached_ids = list(cache_obj.get("doc_ids", []))
        doc_embs = cache_obj.get("embs", None)
        if doc_embs is None or not cached_ids:
            raise SystemExit(f"[cache] Invalid cache in {cache_path}")
        if cached_ids != doc_ids:
            raise SystemExit(
                "[cache] Error: Doc IDs in cache do not match the current corpus."
            )
    else:
        doc_target_bs = max(32, query_batch_size)
        print(f"[encode] Encoding docs with target batch_size={doc_target_bs}")
        doc_embs = encode_texts_auto_bs(
            doc_texts,
            tokenizer,
            model,
            device=device,
            target_batch_size=doc_target_bs,
            desc="Docs",
        )
        print(f"[cache] Saving doc embeddings to {cache_path}")
        torch.save({"doc_ids": doc_ids, "embs": doc_embs}, cache_path)

    print(f"[encode] Encoding queries with target batch_size={query_batch_size}")
    q_embs = encode_texts_auto_bs(
        q_texts,
        tokenizer,
        model,
        device=device,
        target_batch_size=query_batch_size,
        desc="Queries",
    )

    doc_embs = doc_embs.cpu()
    q_embs = q_embs.cpu()
    if device.type == "cuda":
        del model
        torch.cuda.empty_cache()

    D, H = doc_embs.shape
    Q, H2 = q_embs.shape
    assert H == H2

    k_eff = min(k, D)
    print(f"[search] Calculating scores: D={D}, Q={Q}, topk={k_eff}")

    with torch.no_grad():
        scores = doc_embs @ q_embs.T 

    top_vals, top_idx = scores.topk(k_eff, dim=0)  
    top_vals = top_vals.cpu()
    top_idx = top_idx.cpu()

    run_basename = f"{queries_path.stem}.run"
    run_path = out_dir / run_basename
    print(f"[run] Writing runfile to {run_path}")

    with run_path.open("w", encoding="utf-8", newline="") as fout:
        for q_pos, qid in enumerate(q_ids):
            doc_indices_for_q = top_idx[:, q_pos].tolist()
            scores_for_q = top_vals[:, q_pos].tolist()

            for rank, (doc_row, score) in enumerate(
                zip(doc_indices_for_q, scores_for_q), start=1
            ):
                doc_id = doc_ids[doc_row]
                fout.write(
                    f"{qid} Q0 {doc_id} {rank} {score:.6f} dense_retrieval\n"
                )

    print(f"[run] Finished, runfile written: {run_path}")
    print(
        "[debug] Contents of out_dir:",
        [p.name for p in sorted(out_dir.iterdir())],
    )


@click.command()
@click.option(
    "--corpus",
    type=Path,
    required=True,
    help="Pfad zum LIR-Korpus (.jsonl oder .jsonl.gz).",
)
@click.option(
    "--queries",
    type=Path,
    required=True,
    help="Pfad zu den LIR-Queries (.jsonl[.gz]).",
)
@click.option(
    "--out_dir",
    type=Path,
    required=True,
    help="Output-Verzeichnis für Runfiles.",
)
@click.option(
    "--k",
    type=int,
    default=1000,
    show_default=True,
    help="Anzahl zurückgegebener Dokumente pro Query.",
)
@click.option(
    "--query-batch-size",
    type=int,
    default=1024,
    show_default=True,
    help="Start-Batch-Size fürs Encoden der Queries (und ungefähr für Docs).",
)
@click.option(
    "--gpu/--cpu",
    "use_gpu",
    default=True,
    show_default=True,
    help="GPU verwenden, falls verfügbar (mit --cpu auf CPU erzwingen).",
)
def main(
    corpus: Path,
    queries: Path,
    out_dir: Path,
    k: int,
    query_batch_size: int,
    use_gpu: bool,
):
    corpus = Path(corpus)
    queries = Path(queries)
    out_dir = Path(out_dir)

    print(f"[dense] Starting Dense Retrieval: corpus={corpus.name}, queries={queries.name}")
    run_search(
        corpus_path=corpus,
        queries_path=queries,
        out_dir=out_dir,
        k=k,
        query_batch_size=query_batch_size,
        use_gpu=use_gpu,
    )


if __name__ == "__main__":
    main()

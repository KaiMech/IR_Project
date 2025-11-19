import json
import gzip
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi
import numpy as np


# Data Processing Functions

def load_corpus(path: Path) -> dict[str, str]:
    """
    Loads the *full document* corpus from a .jsonl.gz file.
    Expected format: {"id": "...", "url": "...","title": "...", "text": "..."}
    """
    print("Loading corpus...")
    corpus = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading corpus"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                doc_id = data.get("id")
                if not doc_id:
                    continue

                text = (data.get("title", "") + " " + data.get("text", "")).strip()
                corpus[doc_id] = text
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(corpus):,} documents from corpus.\n")
    return corpus

#### Chunking corpus without overlap

# def chunk_corpus(
#     corpus: dict[str, str],
#     chunk_size: int = 350
# ) -> tuple[dict[str, str], dict[str, str]]:
#     """
#     Chunks the corpus texts into smaller pieces based on word count.
#     """
#     print(f"Chunking corpus into {chunk_size}-word chunks...")
#     chunked_corpus = {}
#     chunk_to_doc_id = {}

#     for doc_id, text in tqdm(corpus.items(), desc="Chunking documents"):

#         tokens = text.split()

#         chunk_index = 0
#         for i in range(0, len(tokens), chunk_size):
#             chunk_tokens = tokens[i : i + chunk_size]
#             chunk_text = " ".join(chunk_tokens)
#             chunk_id = f"{doc_id}_{chunk_index}"

#             chunked_corpus[chunk_id] = chunk_text
#             chunk_to_doc_id[chunk_id] = doc_id

#             chunk_index += 1

#     print(f"Created {len(chunked_corpus):,} chunks from {len(corpus):,} documents.\n")
#     return chunked_corpus, chunk_to_doc_id

def chunk_corpus(
    corpus: dict[str, str],
    chunk_size: int = 100,
    stride: int = 50
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Chunks the corpus texts into smaller pieces based on word count,
    using a sliding window with overlap.

    Overlap = chunk_size - stride
    """
    if stride > chunk_size:
        print(f"Warning: Stride ({stride}) is > chunk_size ({chunk_size}). Setting stride = chunk_size.")
        stride = chunk_size

    print(f"Chunking corpus into {chunk_size}-word chunks with {stride}-word stride...")
    chunked_corpus = {}
    chunk_to_doc_id = {}

    for doc_id, text in tqdm(corpus.items(), desc="Chunking documents"):
        tokens = text.split()

        if not tokens: # Handle empty documents
            continue

        chunk_index = 0
        # Use a sliding window with 'stride'
        for i in range(0, len(tokens), stride):
            # The chunk starts at 'i' and goes to 'i + chunk_size'
            chunk_tokens = tokens[i : i + chunk_size]

            if not chunk_tokens:
                continue

            chunk_text = " ".join(chunk_tokens)
            chunk_id = f"{doc_id}_{chunk_index}"

            chunked_corpus[chunk_id] = chunk_text
            chunk_to_doc_id[chunk_id] = doc_id

            chunk_index += 1

            # Stop if this chunk has already reached the end of the text
            # This check prevents a final, tiny, non-overlapped chunk
            if i + chunk_size >= len(tokens):
                break

    print(f"Created {len(chunked_corpus):,} chunks from {len(corpus):,} documents.\n")
    return chunked_corpus, chunk_to_doc_id

def load_queries(path: Path) -> dict[str, str]:
    """
    Loads queries from a .jsonl file.
    Expected format: {"query_id": "...", "query": "..."}
    """
    print(f"Loading queries from: {path.name}")
    queries = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {path.name}"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                query_id = data.get("query_id")
                query_text = data.get("query")

                queries[query_id] = query_text
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(queries):,} queries from {path.name}.\n")
    return queries

def load_qrels(path: Path) -> dict[str, set[str]]:
    """
    Loads *document-level* query-relevance pairs (qrels).
    The format is typically: query-id \t placeholder \t document-id \t relevance
    """
    print(f"Loading Qrels from: {path.name}")
    qrels = defaultdict(set)
    if not path.exists():
        print(f"Warning: Qrels file not found, skipping: {path}")
        return qrels

    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {path.name}"):
            if not line.strip():
                continue
            try:
                query_id, _, doc_id, _ = line.strip().split()
                qrels[query_id].add(doc_id)
            except ValueError:
                continue
    print(f"Loaded relevance info for {len(qrels):,} queries from {path.name}.\n")
    return qrels

# def create_chunk_qrels(
#     doc_qrels: dict[str, set[str]],
#     chunk_to_doc_id: dict[str, str]
# ) -> dict[str, set[str]]:
#     """
#     Converts document-level qrels to chunk-level qrels.

#     This function is necessary to map all relevant documents to their constituent chunks.
#     """
#     print("Converting document qrels to chunk qrels...")
#     chunk_qrels = defaultdict(set)

#     # Invert the doc_qrels for faster lookup: doc_id -> set(query_ids)
#     doc_to_queries = defaultdict(set)
#     for query_id, doc_ids in doc_qrels.items():
#         for doc_id in doc_ids:
#             doc_to_queries[doc_id].add(query_id)

#     # Iterate over all chunks
#     for chunk_id, doc_id in tqdm(chunk_to_doc_id.items(), desc="Mapping qrels"):
#         # Find all queries relevant to this chunk's parent doc
#         query_ids = doc_to_queries.get(doc_id)

#         if query_ids:
#             # Add this chunk_id to the qrels for all relevant queries
#             for query_id in query_ids:
#                 chunk_qrels[query_id].add(chunk_id)

#     print(f"Converted doc-qrels to {len(chunk_qrels)} query-chunk mappings.\n")
#     return chunk_qrels

def create_bm25_positive_qrels(
    queries: dict[str, str],
    doc_qrels: dict[str, set[str]],
    chunked_corpus: dict[str, str],
    chunk_to_doc_id: dict[str, str]
) -> dict[str, set[str]]:
    """
    Finds the *single best* positive chunk for each (query, relevant_doc) pair
    using BM25 and creates new chunk-level qrels.
    """
    print("Finding best positive chunks using BM25...")

    # 1. Create a reverse map: doc_id -> list of its chunk_ids
    print("Building doc-to-chunk map...")
    doc_to_chunks = defaultdict(list)
    for chunk_id, doc_id in chunk_to_doc_id.items():
        doc_to_chunks[doc_id].append(chunk_id)

    positive_chunk_qrels = defaultdict(set)

    # 2. Iterate over all queries that have relevance judgments
    for query_id, relevant_doc_ids in tqdm(doc_qrels.items(), desc="BM25 Scoring"):
        query_text = queries.get(query_id)
        if not query_text:
            continue

        tokenized_query = query_text.lower().split()

        # 3. For each relevant document, find its single best chunk
        for doc_id in relevant_doc_ids:
            chunk_ids_for_doc = doc_to_chunks.get(doc_id)

            if not chunk_ids_for_doc:
                continue

            # Get the text for all chunks of this one document
            chunk_texts = [chunked_corpus[cid] for cid in chunk_ids_for_doc]

            # Tokenize them for BM25
            tokenized_chunks = [c.lower().split() for c in chunk_texts]

            if not tokenized_chunks:
                continue

            # 4. Initialize BM25 *only* on this doc's chunks
            try:
                bm25 = BM25Okapi(tokenized_chunks)
            except ValueError:
                # Can happen if all chunks are empty after tokenization
                continue

            # 5. Score the query against these chunks
            scores = bm25.get_scores(tokenized_query)

            # 6. Find the chunk with the highest score
            best_chunk_index = np.argmax(scores)
            best_chunk_id = chunk_ids_for_doc[best_chunk_index]

            # 7. Add this *single best chunk* as the positive
            positive_chunk_qrels[query_id].add(best_chunk_id)

    total_mappings = sum(len(c) for c in positive_chunk_qrels.values())
    print(f"Created {total_mappings:,} positive chunk mappings from {len(positive_chunk_qrels):,} queries.\n")
    return positive_chunk_qrels
import json
import gzip
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi
import numpy as np
import sqlite3
import os
import gc
import random
from torch.utils.data import Dataset
import torch

class DPRFineTuningDataset(Dataset):
    def __init__(self, training_triplets, q_tokenizer, c_tokenizer, max_length):
        self.triplets = training_triplets
        self.q_tokenizer = q_tokenizer
        self.c_tokenizer = c_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        # triplet: (query_text, positive_chunk_text, [negative_chunk_texts])
        query, pos_chunk, neg_chunks = self.triplets[idx]

        # Tokenize question
        q_inputs = self.q_tokenizer(
            query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Combine positive and negative contexts
        all_contexts = [pos_chunk] + neg_chunks

        # Tokenize all contexts
        c_inputs = self.c_tokenizer(
            all_contexts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # The positive sample is always the first one in the context batch
        positive_idx = torch.tensor(0, dtype=torch.long)

        return {
            "q_input_ids": q_inputs["input_ids"].squeeze(0),
            "q_attention_mask": q_inputs["attention_mask"].squeeze(0),
            "c_input_ids": c_inputs["input_ids"],
            "c_attention_mask": c_inputs["attention_mask"],
            "positive_idx": positive_idx
        }


class SqliteDict:
    """
    A dictionary-like wrapper around SQLite to store massive datasets on disk.
    Supports both .get() and brackets [key].
    """
    def __init__(self, db_path="temp_corpus.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        self.cursor.execute("CREATE TABLE IF NOT EXISTS data (key TEXT PRIMARY KEY, value TEXT)")
        self.cursor.execute("PRAGMA synchronous = OFF")
        self.cursor.execute("PRAGMA journal_mode = MEMORY")
        self.conn.commit()

    def __setitem__(self, key, value):
        self.cursor.execute("INSERT OR REPLACE INTO data (key, value) VALUES (?, ?)", (key, value))

    def __getitem__(self, key):
        self.cursor.execute("SELECT value FROM data WHERE key = ?", (key,))
        row = self.cursor.fetchone()
        if row:
            return row[0]
        raise KeyError(f"Key '{key}' not found in Disk Dictionary.")

    def get(self, key, default=None):
        self.cursor.execute("SELECT value FROM data WHERE key = ?", (key,))
        row = self.cursor.fetchone()
        return row[0] if row else default

    def keys(self):
        self.cursor.execute("SELECT key FROM data")
        return [row[0] for row in self.cursor.fetchall()]

    def close(self):
        self.conn.close()

    def commit(self):
        self.conn.commit()


# Data Processing Functions

def load_corpus(path: Path) -> dict[str, str]:
    """
    Loads the full document corpus from a .jsonl.gz file.
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
    Loads document-level query-relevance pairs (qrels).
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

def chunk_and_store_to_db(
    corpus: dict[str, str],
    db_path: str,
    chunk_size: int = 100,
    stride: int = 50
) -> dict[str, str]:
    """
    Chunks corpus using a SLIDING WINDOW and writes directly to SQLite.
    Returns ONLY the ID mapping.
    """
    if stride > chunk_size:
        print(f"Warning: Stride ({stride}) > Chunk Size ({chunk_size}). Setting stride = chunk_size.")
        stride = chunk_size

    print(f"Chunking (Size: {chunk_size}, Stride: {stride}) and streaming to disk ({db_path})...")

    # Initialize DB
    if os.path.exists(db_path):
        os.remove(db_path)

    db = SqliteDict(db_path)

    chunk_to_doc_id = {}
    count = 0

    for doc_id, text in tqdm(corpus.items(), desc="Chunking"):
        tokens = text.split()

        if not tokens:
            continue

        chunk_index = 0

        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i : i + chunk_size]

            if not chunk_tokens:
                continue

            chunk_text = " ".join(chunk_tokens)
            chunk_id = f"{doc_id}_{chunk_index}"

            db[chunk_id] = chunk_text

            chunk_to_doc_id[chunk_id] = doc_id

            chunk_index += 1
            count += 1

            if count % 10000 == 0:
                db.commit()

            # Stop if we have reached the end of the text to avoid
            # creating a tiny final chunk (unless the doc is smaller than chunk_size)
            if i + chunk_size >= len(tokens):
                break

    db.commit()
    db.close()

    print(f"Streamed {count:,} overlapped chunks to disk.\n")
    return chunk_to_doc_id

def create_chunk_qrels(
    doc_qrels: dict[str, set[str]],
    chunk_to_doc_id: dict[str, str]
) -> dict[str, set[str]]:
    """
    Converts document-level qrels to chunk-level qrels.

    This function is necessary to map all relevant documents to their constituent chunks.
    """
    print("Converting document qrels to chunk qrels...")
    chunk_qrels = defaultdict(set)

    # Invert the doc_qrels for faster lookup: doc_id -> set(query_ids)
    doc_to_queries = defaultdict(set)
    for query_id, doc_ids in doc_qrels.items():
        for doc_id in doc_ids:
            doc_to_queries[doc_id].add(query_id)

    for chunk_id, doc_id in tqdm(chunk_to_doc_id.items(), desc="Mapping qrels"):
        # Find all queries relevant to this chunk's parent doc
        query_ids = doc_to_queries.get(doc_id)

        if query_ids:
            # Add this chunk_id to the qrels for all relevant queries
            for query_id in query_ids:
                chunk_qrels[query_id].add(chunk_id)

    print(f"Converted doc-qrels to {len(chunk_qrels)} query-chunk mappings.\n")
    return chunk_qrels

def create_bm25_positive_qrels(
    queries: dict[str, str],
    doc_qrels: dict[str, set[str]],
    chunked_corpus: dict[str, str],
    chunk_to_doc_id: dict[str, str]
) -> dict[str, set[str]]:
    """
    Finds the single best positive chunk for each (query, relevant_doc) pair
    using BM25 and creates new chunk-level qrels.
    """
    print("Finding best positive chunks using BM25...")

    # 1) Create a reverse map: doc_id -> list of its chunk_ids
    print("Building doc-to-chunk map...")
    doc_to_chunks = defaultdict(list)
    for chunk_id, doc_id in chunk_to_doc_id.items():
        doc_to_chunks[doc_id].append(chunk_id)

    positive_chunk_qrels = defaultdict(set)

    # 2) Iterate over all queries that have relevance judgments
    for query_id, relevant_doc_ids in tqdm(doc_qrels.items(), desc="BM25 Scoring"):
        query_text = queries.get(query_id)
        if not query_text:
            continue

        tokenized_query = query_text.lower().split()

        # 3) For each relevant document, find its single best chunk
        for doc_id in relevant_doc_ids:
            chunk_ids_for_doc = doc_to_chunks.get(doc_id)

            if not chunk_ids_for_doc:
                continue

            # Get the text for all chunks of this one document
            chunk_texts = [chunked_corpus[cid] for cid in chunk_ids_for_doc]

            tokenized_chunks = [c.lower().split() for c in chunk_texts]

            if not tokenized_chunks:
                continue

            # 4) Initialize BM25 only on this doc's chunks
            try:
                bm25 = BM25Okapi(tokenized_chunks)
            except ValueError:
                continue

            # 5) Score the query against these chunks
            scores = bm25.get_scores(tokenized_query)

            # 6) Find the chunk with the highest score
            best_chunk_index = np.argmax(scores)
            best_chunk_id = chunk_ids_for_doc[best_chunk_index]

            # 7) Add this single best chunk as the positive
            positive_chunk_qrels[query_id].add(best_chunk_id)

    total_mappings = sum(len(c) for c in positive_chunk_qrels.values())
    print(f"Created {total_mappings:,} positive chunk mappings from {len(positive_chunk_qrels):,} queries.\n")
    return positive_chunk_qrels


def mine_hard_negatives_doc_level(
    queries: dict[str, str],
    corpus: dict[str, str],
    doc_to_chunks: dict[str, list],   # Map: doc_id -> [chunk_id_1, chunk_id_2...]
    positive_doc_map: dict[str, set[str]] # query_id -> set(positive_doc_ids)
) -> dict[str, str]:

    print("Tokenizing Documents for Hard Negative Mining (Truncated)...")

    doc_ids = list(corpus.keys())

    tokenized_corpus = []
    for doc_id in tqdm(doc_ids, desc="Tokenizing"):
        text = corpus[doc_id]
        # Split and keep only first 256 tokens
        tokens = text.lower().split()[:256]
        tokenized_corpus.append(tokens)

    print(f"Building BM25 Index on {len(doc_ids)} documents...")
    bm25 = BM25Okapi(tokenized_corpus)

    del tokenized_corpus
    gc.collect()

    hard_negatives_chunk_map = {} # query_id -> chunk_id

    print("Querying BM25 for Hard Negatives...")
    for query_id, query_text in tqdm(queries.items(), desc="Mining"):
        if query_id not in positive_doc_map:
            continue

        known_pos_doc_ids = positive_doc_map[query_id]
        tokenized_query = query_text.lower().split()

        # Get top 10 similar DOCUMENTS
        top_doc_ids = bm25.get_top_n(tokenized_query, doc_ids, n=10)

        for candidate_doc_id in top_doc_ids:
            # 1) Strict Check: The Document itself must not be a positive doc
            if candidate_doc_id not in known_pos_doc_ids:

                # 2) We found a Hard Negative DOCUMENT.
                # Now pick a chunk from it to be the Hard Negative CHUNK.
                possible_chunks = doc_to_chunks.get(candidate_doc_id)

                if possible_chunks:
                    # Randomly select one chunk from this distractor document
                    selected_chunk_id = random.choice(possible_chunks)
                    hard_negatives_chunk_map[query_id] = selected_chunk_id
                    break

    print(f"Mined {len(hard_negatives_chunk_map)} hard negative chunks.\n")
    return hard_negatives_chunk_map
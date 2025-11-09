import gzip
import json
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer, 
    InputExample, 
    losses, 
    models, 
    evaluation
)
from sentence_transformers.datasets import NoDuplicatesDataLoader
from tqdm.auto import tqdm
import torch

# 1 CONFIGURE YOUR PATHS AND SETTINGS

CORPUS_PATH = Path("/corpus.jsonl.gz")
QUERY_DIR = Path("/queries_folder/")
QRELS_DIR = Path("/qrels_folder/")

TRAIN_SET_KEY = "train"
EVAL_SET_KEY = "dev1"  # or dev2, dev3, test


QUERY_MODEL_NAME = "facebook/dpr-question_encoder-single-nq-base"
CONTEXT_MODEL_NAME = "facebook/dpr-context_encoder-single-nq-base"

OUTPUT_PATH = "tot-dpr-model"
BATCH_SIZE = 32
NUM_EPOCHS = 1
CHUNK_SIZE_IN_WORDS = 350

# 2 HELPER FUNCTIONS FOR DATA LOADING & CHUNKING

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
                print(f"Skipping bad JSON line: {line[:50]}...")
    print(f"Loaded {len(corpus):,} documents from corpus.\n")
    return corpus

def chunk_corpus(
    corpus: dict[str, str], 
    chunk_size: int = 350
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Chunks the corpus texts into smaller pieces based on word count.
    """
    print(f"Chunking corpus into {chunk_size}-word chunks...")
    chunked_corpus = {}
    chunk_to_doc_id = {}

    for doc_id, text in tqdm(corpus.items(), desc="Chunking documents"):
        
        tokens = text.split()
            
        chunk_index = 0
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = " ".join(chunk_tokens)
            chunk_id = f"{doc_id}_{chunk_index}"
            
            chunked_corpus[chunk_id] = chunk_text
            chunk_to_doc_id[chunk_id] = doc_id
            
            chunk_index += 1
            
    print(f"Created {len(chunked_corpus):,} chunks from {len(corpus):,} documents.\n")
    return chunked_corpus, chunk_to_doc_id

def load_queries(path: Path) -> dict[str, str]:
    """
    Loads queries from a .jsonl file.
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
                print(f"Skipping bad JSON line: {line[:50]}...")
    print(f"Loaded {len(queries):,} queries from {path.name}.\n")
    return queries

def load_qrels(path: Path) -> dict[str, str]:
    """
    Loads *document-level* query-relevance pairs (qrels).
    """
    print(f"Loading Qrels from: {path.name}")
    qrels = defaultdict(str)
    if not path.exists():
        print(f"Warning: Qrels file not found, skipping: {path}")
        return qrels
        
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {path.name}"):
            try:
                query_id, _, doc_id, _ = line.strip().split()
                qrels[query_id] = doc_id
            except ValueError:
                continue
    print(f"Loaded relevance info for {len(qrels):,} queries from {path.name}.\n")
    return qrels

def create_chunk_qrels(
    doc_qrels: dict[str, set[str]],
    chunk_to_doc_id: dict[str, str]
) -> dict[str, set[str]]:
    """
    Converts document-level qrels to chunk-level qrels.
    
    This assumes if a document is relevant, ALL of its chunks are
    considered relevant (a common simplification).
    """
    print("Converting document qrels to chunk qrels...")
    chunk_qrels = defaultdict(set)
    
    # Invert the doc_qrels for faster lookup: doc_id -> query_id
    doc_to_queries = defaultdict(str)
    for query_id, doc_id in doc_qrels.items():
        doc_to_queries[doc_id] = query_id
            
    # Iterate over all chunks
    for chunk_id, doc_id in tqdm(chunk_to_doc_id.items(), desc="Mapping qrels"):
        # Find all queries relevant to this chunk's parent doc
        query_id = doc_to_queries.get(doc_id)
        # Add this chunk_id to the qrels for that query
        chunk_qrels[query_id].add(chunk_id)

    print(f"Converted doc-qrels to {len(chunk_qrels)} query-chunk mappings.\n")
    return chunk_qrels

# 3 MAIN TRAINING FUNCTION 

def run_training(
    chunked_corpus: dict[str, str],  
    all_queries: dict[str, dict[str, str]],
    train_chunk_qrels: dict[str, set[str]], 
    eval_chunk_qrels: dict[str, set[str]],
    train_set_key: str,
    eval_set_key: str,
    query_model_name: str,
    context_model_name: str,
    output_path: str,
    batch_size: int,
    num_epochs: int
):
    """
    Executes the main model training and evaluation logic
    using an asymmetric DPR-style architecture on *chunks*.
    """
    
    # 1: Define the Asymmetric Models
    print("Initializing asymmetric DPR-style models...")
    
    word_embedding_model_q = models.Transformer(query_model_name, max_seq_length=512)
    pooling_model_q = models.Pooling(
        word_embedding_model_q.get_word_embedding_dimension(), 
        pooling_mode_cls_token=True
    )
    query_encoder = SentenceTransformer(modules=[word_embedding_model_q, pooling_model_q])

    word_embedding_model_c = models.Transformer(context_model_name, max_seq_length=350)
    pooling_model_c = models.Pooling(
        word_embedding_model_c.get_word_embedding_dimension(), 
        pooling_mode_cls_token=True
    )
    context_encoder = SentenceTransformer(modules=[word_embedding_model_c, pooling_model_c])

    # 2: Create Training Examples (from Chunks)
    print("Preparing training examples...")
    train_examples = []
    
    train_queries = all_queries.get(train_set_key)

    for query_id, pos_chunk_ids in tqdm(train_chunk_qrels.items(), desc="Creating examples"):
        query_text = train_queries.get(query_id)
            
        for chunk_id in pos_chunk_ids: 
            chunk_text = chunked_corpus.get(chunk_id) 
            
            train_examples.append(InputExample(texts=[query_text, chunk_text]))

    print(f"Created {len(train_examples):,} (query, positive_passage) pairs.\n")

    # 3: Configure DataLodaer and Loss
    train_dataloader = NoDuplicatesDataLoader(
        train_examples, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    train_loss = losses.MultipleNegativesRankingLoss(
        model=query_encoder, 
        doc_model=context_encoder
    )

    # 4: Set up an Evaluator (with Chunks)
    print("Setting up evaluator...")
    
    eval_queries = all_queries.get(eval_set_key)
    
    evaluator = evaluation.InformationRetrievalEvaluator(
        queries=eval_queries,         
        corpus=chunked_corpus,        
        relevant_docs=eval_chunk_qrels, 
        query_model=query_encoder,    
        corpus_model=context_encoder, 
        name=f"{eval_set_key}-eval",
        show_progress_bar=True,
        main_score="ndcg@10"
    )

    # 5: Train the Model
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    
    print(f"--- Starting Asymmetric Training ---")
    print(f"Query Model: {query_model_name}")
    print(f"Context Model: {context_model_name}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Output Path: {output_path}")
    print(f"Warmup Steps: {warmup_steps}")
    print(f"Training on: {train_set_key} ({len(train_examples)} examples)")
    print(f"Evaluating on: {eval_set_key} ({len(eval_queries or {})} queries)")
    print(f"-------------------------")

    query_encoder.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader),
        output_path=output_path,
        save_best_model=True,
        use_amp=True,
        checkpoint_save_steps=int(len(train_dataloader) * 0.25),
        checkpoint_path=f"{output_path}/checkpoints"
    )

    context_encoder_path = f"{output_path}-context"
    context_encoder.save(context_encoder_path)

    print(f"\nTraining complete.")
    print(f"Best query encoder saved to: {output_path}")
    print(f"Context encoder saved to: {context_encoder_path}")


if __name__ == "__main__":
    
    # 1: Load original documents
    corpus = load_corpus(CORPUS_PATH)
    
    # 2: Chunk the corpus
    chunked_corpus, chunk_to_doc_id = chunk_corpus(corpus, chunk_size=CHUNK_SIZE_IN_WORDS)
    
    # 3: Load all query sets
    print("--- Loading all query sets ---")
    query_keys = ["train", "dev1", "dev2", "dev3", "test"]
    all_queries = {}
    
    for key in query_keys:
        query_path = QUERY_DIR / f"{key}-queries.jsonl"
        
        if query_path.exists():
            all_queries[key] = load_queries(query_path)
        else:
            print(f"Warning: Query file not found, skipping: {query_path}")

    # 4: Load original document-level qrels
    print("--- Loading Document-Level Qrels ---")
    train_qrels_path = QRELS_DIR / f"{TRAIN_SET_KEY}-qrels.txt"
    train_doc_qrels = load_qrels(train_qrels_path)

    eval_qrels_path = QRELS_DIR / f"{EVAL_SET_KEY}-qrels.txt"
    eval_doc_qrels = load_qrels(eval_qrels_path)
    
    # 5: Convert doc-level qrels to chunk-level qrels
    train_chunk_qrels = create_chunk_qrels(train_doc_qrels, chunk_to_doc_id)
    eval_chunk_qrels = create_chunk_qrels(eval_doc_qrels, chunk_to_doc_id)
    
    # 6: Run the training
    run_training(
        chunked_corpus=chunked_corpus,    
        all_queries=all_queries,
        train_chunk_qrels=train_chunk_qrels,
        eval_chunk_qrels=eval_chunk_qrels,  
        train_set_key=TRAIN_SET_KEY,
        eval_set_key=EVAL_SET_KEY,
        query_model_name=QUERY_MODEL_NAME,
        context_model_name=CONTEXT_MODEL_NAME,
        output_path=OUTPUT_PATH,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS
    )
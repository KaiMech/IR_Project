import torch
import os
import random
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoderTokenizer,
    get_linear_schedule_with_warmup
)
from embeddings.utils import (
    load_corpus,
    chunk_corpus,
    load_queries,
    load_qrels,
    create_bm25_positive_qrels
)

# DPR Configuration
QUESTION_ENCODER_MODEL = "facebook/dpr-question_encoder-single-nq-base"
CONTEXT_ENCODER_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 4
NUM_NEGATIVES = 7
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DPR Dataset for Fine-Tuning

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

# Training Triplet Creation
def create_training_triplets(
    queries: dict[str, str],
    chunked_corpus: dict[str, str],
    chunk_qrels: dict[str, set[str]],
    num_negatives: int = NUM_NEGATIVES
) -> list[tuple[str, str, list[str]]]:
    """
    Creates (query, positive_chunk, [negative_chunks]) triplets for training.

    This version uses "safe" negatives, ensuring a sampled negative
    is not another known positive chunk for that query.
    """
    print(f"Creating training triplets with {num_negatives} negatives per positive sample...")
    triplets = []
    all_chunk_ids = list(chunked_corpus.keys())

    for query_id, relevant_chunk_ids in tqdm(chunk_qrels.items(), desc="Creating triplets"):
        query_text = queries.get(query_id)

        # Get *all* positive chunks for this query (from BM25 step)
        positive_set_for_query = chunk_qrels.get(query_id, set())

        if not query_text or not positive_set_for_query:
            continue

        # Build a pool of *valid* negatives (all chunks - all known positives)
        negative_pool = [
            cid for cid in all_chunk_ids
            if cid not in positive_set_for_query
        ]

        if not negative_pool: # Should be rare
            continue

        # Create a triplet for EACH positive chunk
        for pos_chunk_id in relevant_chunk_ids:
            pos_chunk_text = chunked_corpus.get(pos_chunk_id)

            if not pos_chunk_text:
                continue

            # Sample from the safe negative pool
            num_to_sample = min(num_negatives, len(negative_pool))
            random_neg_ids = random.sample(negative_pool, num_to_sample)

            neg_chunk_texts = [chunked_corpus[neg_id] for neg_id in random_neg_ids]

            triplets.append((query_text, pos_chunk_text, neg_chunk_texts))

    print(f"Created {len(triplets):,} training triplets.\n")
    return triplets

# def create_training_triplets(
#     queries: dict[str, str],
#     chunked_corpus: dict[str, str],
#     chunk_qrels: dict[str, set[str]],
#     num_negatives: int = NUM_NEGATIVES
# ) -> list[tuple[str, str, list[str]]]:
#     """
#     Creates (query, positive_chunk, [negative_chunks]) triplets for training.

#     CORRECTION: Creates a triplet for *every* relevant chunk associated with a query.
#     """
#     print(f"Creating training triplets with {num_negatives} negatives per positive sample...")
#     triplets = []
#     all_chunk_ids = list(chunked_corpus.keys())

#     # 1. Iterate over queries that have at least one relevant chunk
#     for query_id, relevant_chunk_ids in tqdm(chunk_qrels.items(), desc="Creating triplets"):
#         query_text = queries.get(query_id)

#         if not query_text or not relevant_chunk_ids:
#             continue

#         # 2. Create a triplet for *EACH* relevant chunk
#         for pos_chunk_id in relevant_chunk_ids:
#             pos_chunk_text = chunked_corpus.get(pos_chunk_id)

#             if not pos_chunk_text:
#                 continue

#             # 3. Sample negatives

#             # Exclude the current positive chunk from the negative sampling pool
#             negative_pool = [cid for cid in all_chunk_ids if cid != pos_chunk_id]

#             # Simple random sampling of chunks
#             num_to_sample = min(num_negatives, len(negative_pool))
#             random_neg_ids = random.sample(negative_pool, num_to_sample)

#             neg_chunk_texts = [chunked_corpus[neg_id] for neg_id in random_neg_ids]

#             # Add the final triplet: (query, positive_chunk, [negative_chunks])
#             triplets.append((query_text, pos_chunk_text, neg_chunk_texts))

#     print(f"Created {len(triplets):,} training triplets (one for every relevant chunk).\n")
#     return triplets

# Training Loop

# Training Loop

def train_dpr(train_dataloader, q_encoder, c_encoder, optimizer, scheduler, num_epochs):
    q_encoder.train()
    c_encoder.train()

    global_step = 0
    total_steps = len(train_dataloader) * num_epochs

    print(f"Starting training for {num_epochs} epochs ({total_steps} steps total) on {DEVICE}...")

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            # Move data to device
            q_input_ids = batch["q_input_ids"].to(DEVICE)
            q_attention_mask = batch["q_attention_mask"].to(DEVICE)
            c_input_ids = batch["c_input_ids"].to(DEVICE)
            c_attention_mask = batch["c_attention_mask"].to(DEVICE)
            positive_idx = batch["positive_idx"].to(DEVICE)

            optimizer.zero_grad()

            # 1. Encode Questions
            q_outputs = q_encoder(
                input_ids=q_input_ids,
                attention_mask=q_attention_mask
            )
            q_embeddings = q_outputs.pooler_output # Shape: (batch_size, embedding_dim)

            # 2. Encode Contexts (Positive + Negatives)
            # Reshape context inputs: (batch_size, num_contexts, seq_len) -> (batch_size * num_contexts, seq_len)
            batch_size, num_contexts_per_sample, seq_len = c_input_ids.shape

            c_input_ids_flat = c_input_ids.view(-1, seq_len)
            c_attention_mask_flat = c_attention_mask.view(-1, seq_len)

            c_outputs = c_encoder(
                input_ids=c_input_ids_flat,
                attention_mask=c_attention_mask_flat
            )
            c_embeddings_flat = c_outputs.pooler_output # Shape: (batch_size * num_contexts, embedding_dim)

            # Reshape context embeddings back: (batch_size * num_contexts, embedding_dim) -> (batch_size, num_contexts, embedding_dim)
            c_embeddings = c_embeddings_flat.view(batch_size, num_contexts_per_sample, -1)

            # 3. Calculate Similarity Scores (Dot Product)
            # scores: (batch_size, num_contexts)
            # This is the core contrastive step: maximize score for the positive index (0)
            scores = torch.einsum("bd,bcd->bc", q_embeddings, c_embeddings)

            # 4. Calculate Loss (Negative Log Likelihood)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(scores, positive_idx)

            # 5. Backpropagation and Optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(q_encoder.parameters()) + list(c_encoder.parameters()), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                print(f"Step {global_step}, Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}")

    # 6. Save Fine-Tuned Models
    output_dir = "./dpr_finetuned_models"
    os.makedirs(output_dir, exist_ok=True)

    q_encoder.save_pretrained(os.path.join(output_dir, "question_encoder"))

    c_encoder.save_pretrained(os.path.join(output_dir, "context_encoder"))

    print(f"\nFine-tuning complete. Models saved to {output_dir}")
    print("To load the fine-tuned models later, use:")
    print(f"q_encoder = DPRQuestionEncoder.from_pretrained('{os.path.join(output_dir, 'question_encoder')}')")
    print(f"c_encoder = DPRContextEncoder.from_pretrained('{os.path.join(output_dir, 'context_encoder')}')")


# Main Execution

def main():
    CORPUS_PATH = Path("data/tot25/subsets/train100k-corpus.jsonl.gz")

    QUERIES_PATH = Path("data/tot25/subsets/train80/train80-queries-train.jsonl")
    QRELS_PATH = Path("data/tot25/subsets/train80/train80-qrels-train.txt")

    # 1) Data Preparation

    # Load and chunk corpus
    full_corpus = load_corpus(CORPUS_PATH)
    chunked_corpus, chunk_to_doc_id = chunk_corpus(full_corpus, chunk_size=256)

    del full_corpus

    # Load queries and relevance judgments
    queries = load_queries(QUERIES_PATH)
    doc_qrels = load_qrels(QRELS_PATH)

    # Convert document-level qrels to chunk-level qrels
    # chunk_qrels = create_chunk_qrels(doc_qrels, chunk_to_doc_id)

    chunk_qrels = create_bm25_positive_qrels(
        queries,
        doc_qrels,
        chunked_corpus,
        chunk_to_doc_id
    )

    # Create training triplets (query, positive_chunk, [negative_chunks])
    training_triplets = create_training_triplets(
        queries,
        chunked_corpus,
        chunk_qrels,
        num_negatives=NUM_NEGATIVES
    )

    if not training_triplets:
        print("Error: No training triplets created. Check your qrels and data paths.")
        return

    # 2) Model and Tokenizer Initialization
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(QUESTION_ENCODER_MODEL)
    c_tokenizer = DPRContextEncoderTokenizer.from_pretrained(CONTEXT_ENCODER_MODEL)

    q_encoder = DPRQuestionEncoder.from_pretrained(QUESTION_ENCODER_MODEL).to(DEVICE)
    c_encoder = DPRContextEncoder.from_pretrained(CONTEXT_ENCODER_MODEL).to(DEVICE)

    # 3) Training Setup
    train_dataset = DPRFineTuningDataset(training_triplets, q_tokenizer, c_tokenizer, MAX_SEQ_LENGTH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Combine parameters from both encoders for joint optimization
    optimizer = AdamW(
        list(q_encoder.parameters()) + list(c_encoder.parameters()),
        lr=LEARNING_RATE
    )

    # Calculate total steps for the scheduler
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 4) Start Training
    train_dpr(train_dataloader, q_encoder, c_encoder, optimizer, scheduler, NUM_EPOCHS)

if __name__ == "__main__":
    main()
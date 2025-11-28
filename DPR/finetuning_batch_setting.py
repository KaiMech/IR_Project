import torch
import os
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoderTokenizer,
    get_linear_schedule_with_warmup
)
import numpy as np
from utils import (
    SqliteDict,
    load_corpus,
    load_queries,
    load_qrels,
    chunk_and_store_to_db,
    create_bm25_positive_qrels,
    mine_hard_negatives_doc_level,
    DPRFineTuningDataset
)
import gc
from torch.amp import autocast, GradScaler

# DPR Configuration
QUESTION_ENCODER_MODEL = "facebook/dpr-question_encoder-single-nq-base"
CONTEXT_ENCODER_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 16
NUM_NEGATIVES = 1
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Triplet Creation
def create_training_triplets(
    queries: dict[str, str],
    chunked_corpus: dict[str, str],
    chunk_qrels: dict[str, set[str]],
    hard_negatives_map: dict[str, str],
    num_negatives: int = NUM_NEGATIVES,
) -> list[tuple[str, str, list[str]]]:

    print(f"Creating training triplets with 1 Hard Negative per positive...")
    triplets = []

    for query_id, relevant_chunk_ids in tqdm(chunk_qrels.items(), desc="Creating triplets"):
        query_text = queries.get(query_id)

        # Get the specific Hard Negative we mined earlier
        hard_neg_id = hard_negatives_map.get(query_id)

        # If we couldn't find a hard negative (rare), skip this sample
        if not query_text or not hard_neg_id:
            continue

        hard_neg_text = chunked_corpus.get(hard_neg_id)
        if not hard_neg_text:
            continue

        # Create triplet for EACH positive chunk
        for pos_chunk_id in relevant_chunk_ids:
            pos_chunk_text = chunked_corpus.get(pos_chunk_id)
            if not pos_chunk_text:
                continue

            # Append (Query, Positive, [One_Hard_Negative])
            triplets.append((query_text, pos_chunk_text, [hard_neg_text]))

    print(f"Created {len(triplets):,} training triplets.\n")
    return triplets



# Training Loop

def train_dpr(train_dataloader, q_encoder, c_encoder, optimizer, scheduler, num_epochs):
    q_encoder.train()
    c_encoder.train()

    # Scaler for Mixed Precision (FP16)
    scaler = GradScaler()

    global_step = 0
    print(f"Starting In-Batch Training (Batch Size {BATCH_SIZE}) with FP16 & Checkpointing...")

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            q_input_ids = batch["q_input_ids"].to(DEVICE)
            q_attention_mask = batch["q_attention_mask"].to(DEVICE)
            c_input_ids = batch["c_input_ids"].to(DEVICE)
            c_attention_mask = batch["c_attention_mask"].to(DEVICE)

            optimizer.zero_grad()

            with autocast():
                # 1) Encode Questions
                q_outputs = q_encoder(input_ids=q_input_ids, attention_mask=q_attention_mask)
                q_embeddings = q_outputs.pooler_output

                # 2) Encode Contexts
                batch_size, num_contexts, seq_len = c_input_ids.shape
                c_input_ids_flat = c_input_ids.view(-1, seq_len)
                c_attention_mask_flat = c_attention_mask.view(-1, seq_len)

                c_outputs = c_encoder(input_ids=c_input_ids_flat, attention_mask=c_attention_mask_flat)
                c_embeddings_flat = c_outputs.pooler_output
                c_embeddings = c_embeddings_flat.view(batch_size, num_contexts, -1)

                # 3) Global Batch Pool
                pos_embeddings = c_embeddings[:, 0, :]
                neg_embeddings = c_embeddings[:, 1, :]
                all_passages = torch.cat([pos_embeddings, neg_embeddings], dim=0)

                # 4) Matrix Multiplication
                scores = torch.matmul(q_embeddings, all_passages.transpose(0, 1))

                # 5) Loss
                target = torch.arange(batch_size, device=DEVICE)
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(scores, target)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(q_encoder.parameters()) + list(c_encoder.parameters()), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                 torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}")

    output_dir = "./dpr_finetuned_models_new_settings"
    os.makedirs(output_dir, exist_ok=True)
    q_encoder.save_pretrained(os.path.join(output_dir, "question_encoder"))
    c_encoder.save_pretrained(os.path.join(output_dir, "context_encoder"))
    print(f"\nFine-tuning complete.")

# Main Execution

def main():
    CORPUS_PATH = Path("data/tot25/subsets/train100k-corpus.jsonl.gz")
    QUERIES_PATH = Path("data/tot25/subsets/train80/train80-queries-train.jsonl")
    QRELS_PATH = Path("data/tot25/subsets/train80/train80-qrels-train.txt")

    TEMP_DB_FILE = "temp_corpus_storage.db"

    full_corpus = load_corpus(CORPUS_PATH)

    chunk_to_doc_id = chunk_and_store_to_db(
        full_corpus,
        TEMP_DB_FILE,
        chunk_size=100,
        stride=50
    )

    # Create reverse map for the miner (doc_id -> [chunk_ids])
    doc_to_chunks = defaultdict(list)
    for chunk_id, doc_id in chunk_to_doc_id.items():
        doc_to_chunks[doc_id].append(chunk_id)

    queries = load_queries(QUERIES_PATH)
    doc_qrels = load_qrels(QRELS_PATH)

    hard_negatives_map = mine_hard_negatives_doc_level(
        queries,
        full_corpus,
        doc_to_chunks,
        doc_qrels
    )

    print("Mining done. Deleting full corpus from RAM...")
    del full_corpus
    gc.collect()

    chunked_corpus = SqliteDict(TEMP_DB_FILE)

    chunk_qrels = create_bm25_positive_qrels(
        queries,
        doc_qrels,
        chunked_corpus,
        chunk_to_doc_id
    )

    training_triplets = create_training_triplets(
        queries,
        chunked_corpus,
        chunk_qrels,
        hard_negatives_map=hard_negatives_map,
        num_negatives=1
    )

    chunked_corpus.close()

    if os.path.exists(TEMP_DB_FILE):
        os.remove(TEMP_DB_FILE)

    if not training_triplets:
        print("No triplets created.")
        return

    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(QUESTION_ENCODER_MODEL)
    c_tokenizer = DPRContextEncoderTokenizer.from_pretrained(CONTEXT_ENCODER_MODEL)

    q_encoder = DPRQuestionEncoder.from_pretrained(QUESTION_ENCODER_MODEL).to(DEVICE)
    c_encoder = DPRContextEncoder.from_pretrained(CONTEXT_ENCODER_MODEL).to(DEVICE)

    q_encoder.question_encoder.bert_model.gradient_checkpointing_enable()
    c_encoder.ctx_encoder.bert_model.gradient_checkpointing_enable()

    train_dataset = DPRFineTuningDataset(training_triplets, q_tokenizer, c_tokenizer, MAX_SEQ_LENGTH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(list(q_encoder.parameters()) + list(c_encoder.parameters()), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    train_dpr(train_dataloader, q_encoder, c_encoder, optimizer, scheduler, NUM_EPOCHS)


if __name__ == "__main__":
    main()
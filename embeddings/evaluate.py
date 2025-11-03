from pathlib import Path
import re
import numpy as np
import argparse

def list_shards(root: Path):
    shard_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and re.match(r'shard\d+$', d.name)],
        key=lambda d: int(re.search(r'(\d+)$', d.name).group(1))
    )
    pairs = {}
    shard_id = 0
    for d in shard_dirs:
        shard_id += 1
        pairs[str(shard_id)] = []
        npys = sorted(d.glob('*.npy'))
        for npy in npys:
            txt = npy.with_suffix('.txt')
            if txt.exists():
                pairs[str(shard_id)].append((npy, txt))
    return pairs

def load_shard(shards, shard_id: str):
    ctx_embs = []
    ctx_ids = []
    for chunk in shards[shard_id]:
        ctx_embs.append(np.load(chunk[0]))
        # ctx_ids.append(np.array(chunk[1].read_text().splitlines()))
        ctx_ids.append(np.loadtxt(chunk[1], dtype=np.int32))
    return np.concatenate(ctx_embs, axis=0), np.concatenate(ctx_ids, axis=0)

def load_query_embeddings(root: Path):
    pairs = []
    npys = root.glob('*.npy')
    for npy in npys:
        txt = npy.with_suffix('.txt')
        if txt.exists():
            pairs.append((npy, txt))
    query_embs = []
    query_ids = []
    for chunk in pairs:
        query_embs.append(np.load(chunk[0]))
        # query_ids.append(np.array(chunk[1].read_text().splitlines()))
        query_ids.append(np.loadtxt(chunk[1], dtype=np.int32))
    return np.concatenate(query_embs, axis=0), np.concatenate(query_ids, axis=0)

def update_topk(scores_topk, indices_topk, scores_new, indices_new, topk=1000):
    Q, K = scores_topk.shape
    assert K == topk
    all_scores = np.concatenate([scores_topk, scores_new], axis=1)
    all_indices = np.concatenate([indices_topk, indices_new], axis=1)
    updated_scores = np.empty((Q, K), dtype=all_scores.dtype)
    updated_indices = np.empty((Q, K), dtype=all_indices.dtype)
    for q in range(Q):
        row_scores = all_scores[q]
        row_indices = all_indices[q]
        topk_unsorted_idx = np.argpartition(row_scores, -K)[-K:]
        order = np.argsort(row_scores[topk_unsorted_idx])[::-1]
        final_idx = topk_unsorted_idx[order]
        updated_scores[q] = row_scores[final_idx]
        updated_indices[q] = row_indices[final_idx]
    return updated_scores, updated_indices

def evaluate(shards, query_path, topk=1000):
    query_embs, query_ids = load_query_embeddings(query_path)
    scores = np.full((len(query_ids), topk), fill_value=-np.inf, dtype=np.float32)
    indices = np.full((len(query_ids), topk), fill_value=-1, dtype=np.int32)
    for shard_id in shards.keys():
        ctx_embs, ctx_ids = load_shard(shards, shard_id)
        scores_shard = np.matmul(query_embs, ctx_embs.T)
        indices_shard = np.broadcast_to(ctx_ids, (scores_shard.shape[0], scores_shard.shape[1])).copy()
        scores, indices = update_topk(scores, indices, scores_shard, indices_shard, topk)
    return scores, indices

def write_trec_run(query_ids, doc_ids_topk, scores_topk, run_tag, outfile_path):
    Q, K = doc_ids_topk.shape
    with open(outfile_path, "w") as f:
        for qi in range(Q):
            qid = query_ids[qi]
            for rank in range(K):
                doc_id = doc_ids_topk[qi, rank]
                score  = scores_topk[qi, rank]

                # skip empty/uninitialized slots if you have them
                # e.g. if score == -inf or doc_id == -1
                if score == -np.inf or doc_id == -1:
                    continue

                line = f"{qid} Q0 {doc_id} {rank+1} {score:.6f} {run_tag}\n"
                f.write(line)

def main():
    parser = argparse.ArgumentParser(description='DPR model evaluation')
    parser.add_argument('--ctx_embs_path', type=str, help='Path to context embeddings and document ids')
    parser.add_argument('--query_embs_path', type=str, help='Path to query embeddings and ids')
    parser.add_argument('--topk', type=int, default=1000, help='Number of results per query')
    parser.add_argument('--results_path', type=str, help='Path to the results')
    
    args = parser.parse_args()
    
    CTX_ROOT = Path(args.ctx_embs_path)
    QUERY_ROOT = Path(args.query_embs_path)
    RESULT_PATH = Path(args.results_path)
    topk = args.topk
    
    shards = list_shards(CTX_ROOT)
    scores, indices = evaluate(shards, QUERY_ROOT, topk)
    
    _, qids = load_query_embeddings(QUERY_ROOT)
    
    write_trec_run(qids, indices, scores, 'DPR', RESULT_PATH)
    

if __name__ == '__main__':
    main()
    
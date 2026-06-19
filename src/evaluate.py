import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def evaluate_random_metrics_at_k(model, train_edge_index, test_edge_index, eval_batch_size, k=20):
    """
    Evaluates the model and returns Recall@K, NDCG@K, MRR@K, and Precision@K.
    """
    model.eval()
    
    test_user_dict = defaultdict(list)
    for i in range(test_edge_index.shape[1]):
        u = test_edge_index[0, i].item()
        item = test_edge_index[1, i].item()
        test_user_dict[u].append(item)
        
    bipartite_edges = model.get_graph(train_edge_index)
        
    with torch.no_grad():
        user_embs, item_embs, _ = model(bipartite_edges)
        
        num_users = user_embs.shape[0]
        top_k_indices_list = []
        
        pbar = tqdm(range(0, num_users, eval_batch_size), desc="Evaluating (80-20)")
        for i in pbar:
            end_idx = min(i + eval_batch_size, num_users)
            batch_u_embs = user_embs[i:end_idx]
            
            # Scores for this batch of users against ALL items
            batch_scores = torch.matmul(batch_u_embs, item_embs.T)
            
            # Mask out training items for these users
            for batch_u_idx in range(end_idx - i):
                real_u_idx = i + batch_u_idx
                train_items = train_edge_index[1][train_edge_index[0] == real_u_idx]
                batch_scores[batch_u_idx, train_items] = -float('inf')
                
            _, batch_top_k = torch.topk(batch_scores, k, dim=1)
            top_k_indices_list.append(batch_top_k.cpu().numpy())
            
        top_k_indices = np.concatenate(top_k_indices_list, axis=0)
        
        recalls = []
        ndcgs = []
        mrrs = []
        precisions = []
        
        for u, target_items in test_user_dict.items():
            user_top_k = top_k_indices[u]
            
            hits = 0
            dcg = 0.0
            first_hit_rank = None
            
            for rank, rec_item in enumerate(user_top_k):
                if rec_item in target_items:
                    hits += 1
                    dcg += 1.0 / np.log2(rank + 2)
                    
                    # Capture the rank of the VERY FIRST hit for MRR
                    # (rank + 1) because enumerate starts at 0, but position 1 is rank 1
                    if first_hit_rank is None:
                        first_hit_rank = rank + 1
                        
            # 1. Recall@K
            recall = hits / len(target_items)
            recalls.append(recall)
            
            # 2. Precision@K
            precision = hits / k
            precisions.append(precision)
            
            # 3. MRR@K
            mrr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
            mrrs.append(mrr)
            
            # 4. NDCG@K
            idcg = 0.0
            num_ideal_hits = min(len(target_items), k)
            for i in range(num_ideal_hits):
                idcg += 1.0 / np.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
            
    return np.mean(recalls), np.mean(ndcgs), np.mean(mrrs), np.mean(precisions)

def evaluate_loo_metrics_at_k(model, train_edge_index, test_edge_index, eval_batch_size, k=10):
    """
    Evaluates the model using Leave-One-Out protocol.
    Returns Hit Ratio@K (Recall), NDCG@K, MRR@K, and Precision@K.
    """
    model.eval()
    
    test_user_dict = {}
    for i in range(test_edge_index.shape[1]):
        u = test_edge_index[0, i].item()
        item = test_edge_index[1, i].item()
        test_user_dict[u] = item
        
    bipartite_edges = model.get_graph(train_edge_index)
        
    with torch.no_grad():
        user_embs, item_embs, _ = model(bipartite_edges)
        
        num_users = user_embs.shape[0]
        top_k_indices_list = []
        
        pbar = tqdm(range(0, num_users, eval_batch_size), desc="Evaluating (LOO)")
        for i in pbar:
            end_idx = min(i + eval_batch_size, num_users)
            batch_u_embs = user_embs[i:end_idx]
            
            batch_scores = torch.matmul(batch_u_embs, item_embs.T)
            
            for batch_u_idx in range(end_idx - i):
                real_u_idx = i + batch_u_idx
                train_items = train_edge_index[1][train_edge_index[0] == real_u_idx]
                batch_scores[batch_u_idx, train_items] = -float('inf')
                
            _, batch_top_k = torch.topk(batch_scores, k, dim=1)
            top_k_indices_list.append(batch_top_k.cpu().numpy())
            
        top_k_indices = np.concatenate(top_k_indices_list, axis=0)
        
        hit_ratios = []
        ndcgs = []
        mrrs = []
        precisions = []
        
        for u, target_item in test_user_dict.items():
            user_top_k = top_k_indices[u]
            
            if target_item in user_top_k:
                rank = np.where(user_top_k == target_item)[0][0]
                
                # 1. Hit Ratio@K (Recall)
                hit_ratios.append(1.0)
                
                # 2. Precision@K
                precisions.append(1.0 / k)
                
                # 3. MRR@K (rank is 0-indexed, so we add 1)
                mrrs.append(1.0 / (rank + 1))
                
                # 4. NDCG@K
                ndcgs.append(1.0 / np.log2(rank + 2))
                
            else:
                # If the item is not in the Top-K, all metrics are 0
                hit_ratios.append(0.0)
                precisions.append(0.0)
                mrrs.append(0.0)
                ndcgs.append(0.0)
            
    return np.mean(hit_ratios), np.mean(ndcgs), np.mean(mrrs), np.mean(precisions)
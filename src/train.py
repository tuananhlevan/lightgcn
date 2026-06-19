import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data_pipeline import load_and_prep_movielens, BPRDataset
from model import SimpleLightGCN
from evaluate import evaluate_random_metrics_at_k, evaluate_loo_metrics_at_k

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 1. Load Data
    train_edge_index, test_edge_index, num_users, num_items = load_and_prep_movielens(Config.DATA_PATH, Config.SPLIT_TYPE)
    train_edge_index = train_edge_index.to(device)
    test_edge_index = test_edge_index.to(device)

    train_dataset = BPRDataset(train_edge_index.cpu(), num_items)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 2. Initialize Model, Graph and Optimizer
    model = SimpleLightGCN(num_users, num_items).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE,
    )
    bipartite_edges = model.get_graph(train_edge_index)

    # 3. Training Loop
    best_ndcg = 0.0
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Training | Epoch [{epoch}/{Config.EPOCHS}]")
        for batch in pbar:
            users, pos_items, neg_items = [x.to(device) for x in batch]
            optimizer.zero_grad()
            
            user_embs, item_embs, emb_0 = model(bipartite_edges)
            
            batch_user_embs = user_embs[users]
            batch_pos_embs = item_embs[pos_items]
            batch_neg_embs = item_embs[neg_items]
            
            pos_scores = (batch_user_embs * batch_pos_embs).sum(dim=1)
            neg_scores = (batch_user_embs * batch_neg_embs).sum(dim=1)
            
            u_emb_0 = emb_0[users]
            pos_emb_0 = emb_0[pos_items + num_users]
            neg_emb_0 = emb_0[neg_items + num_users]
            
            reg_loss = (1/2) * (u_emb_0.norm(2).pow(2) + 
                                pos_emb_0.norm(2).pow(2) + 
                                neg_emb_0.norm(2).pow(2)) / float(len(users))
            bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            loss = bpr_loss + Config.REG_WEIGHT * reg_loss
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix_str(f"Loss: {loss.cpu().detach():.4f}")
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Routine Save
        if epoch % Config.SAVE_EPOCH == 0:
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            save_path = os.path.join(Config.CKPT_DIR, f"model_epoch_{epoch}_{Config.SPLIT_TYPE}.pth")
            torch.save(checkpoint_state, save_path)
            print(f"Routine checkpoint saved at Epoch {epoch}")

        # Evaluation & Best Model Save
        if epoch % Config.EVAL_EPOCH == 0:
            if Config.SPLIT_TYPE == "random":
                recall, ndcg, mrr, precision = evaluate_random_metrics_at_k(
                    model, train_edge_index, test_edge_index, eval_batch_size=Config.BATCH_SIZE, k=Config.K
                )
            elif Config.SPLIT_TYPE == "temporal":
                recall, ndcg, mrr, precision = evaluate_loo_metrics_at_k(
                    model, train_edge_index, test_edge_index, eval_batch_size=Config.BATCH_SIZE, k=Config.K
                )
            
            print(f"Epoch {epoch:03d} | Training Loss: {avg_loss:.4f} | Recall@{Config.K}: {recall:.4f} | NDCG@{Config.K}: {ndcg:.4f} | MRR@{Config.K}: {mrr:.4f} | Precision@{Config.K}: {precision:.4f}")
            
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    f'best_ndcg@{Config.K}': best_ndcg,
                    f'recall@{Config.K}': recall,
                    f'mrr@{Config.K}': mrr,
                    f'precision@{Config.K}': precision
                }
                torch.save(best_state, os.path.join(Config.CKPT_DIR, f"best_model_{Config.SPLIT_TYPE}.pth"))
                print(f"-> New best model saved! (NDCG@{Config.K}: {best_ndcg:.4f})")

if __name__ == "__main__":
    train()
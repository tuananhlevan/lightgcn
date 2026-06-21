import torch
import torch.nn as nn
from torch_geometric.nn import LGConv
from torch_geometric.utils import to_undirected

class SimpleLightGCN(nn.Module):
    def __init__(self, num_users, num_items, num_layers=3, emb_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(num_users + num_items, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        
        self.conv = LGConv()
    
    def get_graph(self, edge_index):
        users_idx = edge_index[0]
        items_idx = edge_index[1] + self.num_users
        bipartite_edges = torch.stack([users_idx, items_idx], dim=0)
        
        # Make the graph undirected
        bipartite_edges = to_undirected(bipartite_edges)
        
        return bipartite_edges
    
    def forward(self, precomputed_bipartite_edges):
        x = self.embedding.weight
        embs = [x]
        
        for _ in range(self.num_layers):
            x = self.conv(x, precomputed_bipartite_edges)
            embs.append(x)
            
        out = torch.mean(torch.stack(embs, dim=0), dim=0)
        user_embs, item_embs = torch.split(out, [self.num_users, self.num_items])
        
        return user_embs, item_embs, self.embedding.weight
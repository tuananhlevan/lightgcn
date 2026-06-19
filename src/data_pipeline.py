import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import random

# from config import Config

class BPRDataset(Dataset):
    def __init__(self, train_edge_index, num_items):
        self.train_edge_index = train_edge_index
        self.num_items = num_items
        self.num_edges = train_edge_index.shape[1]
        
        self.user_history = {}
        for i in range(self.num_edges):
            u = train_edge_index[0, i].item()
            item = train_edge_index[1, i].item()
            if u not in self.user_history:
                self.user_history[u] = set()
            self.user_history[u].add(item)

    def __len__(self):
        return self.num_edges

    def __getitem__(self, idx):
        user = self.train_edge_index[0, idx].item()
        pos_item = self.train_edge_index[1, idx].item()

        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.user_history[user]:
                break

        return torch.tensor(user, dtype=torch.long), \
               torch.tensor(pos_item, dtype=torch.long), \
               torch.tensor(neg_item, dtype=torch.long)

def load_and_prep_movielens(data_path, split_type="random"):
    df = pd.read_csv(data_path)
    
    df = df[df['rating'] >= 3.0].copy()
    
    # Ensure every user has at least 5 positive ratings before splitting
    user_counts = df['userId'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    df = df[df['userId'].isin(valid_users)].copy()

    df['user_idx'], user_uniques = pd.factorize(df['userId'])
    df['item_idx'], item_uniques = pd.factorize(df['movieId'])
    
    num_users = len(user_uniques)
    num_items = len(item_uniques)
    
    if split_type == "temporal":
        # Sort by time
        df = df.sort_values(['user_idx', 'timestamp'])

        group_sizes = df.groupby('user_idx')['user_idx'].transform('size')
        cutoff = group_sizes - 1
        ranks = df.groupby('user_idx').cumcount()

        test_df = df[ranks >= cutoff]
        train_df = df[ranks < cutoff]
    elif split_type == "random":
        train_df, test_df = train_test_split(
            df, 
            test_size=Config.TEST_SIZE, 
            random_state=42, 
            stratify=df['user_idx']
        )
        
    train_edge_index = torch.tensor(
        np.array([train_df['user_idx'].values, train_df['item_idx'].values]), 
        dtype=torch.long
    )
    
    test_edge_index = torch.tensor(
        np.array([test_df['user_idx'].values, test_df['item_idx'].values]), 
        dtype=torch.long
    )
    
    return train_edge_index, test_edge_index, num_users, num_items
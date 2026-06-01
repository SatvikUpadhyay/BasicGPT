# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import urllib.request

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        self.tokens = tokenizer.encode(text)
        
        # Create non-overlapping chunks
        self.chunks = []
        for i in range(0, len(self.tokens) - max_seq_length, max_seq_length):
            self.chunks.append(self.tokens[i : i + max_seq_length + 1])
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, labels


def download_tinystories():
    """Download Shakespeare dataset"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "data.txt")
    print("Downloaded Shakespeare to data.txt")


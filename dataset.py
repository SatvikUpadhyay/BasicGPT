# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import urllib.request

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Tokenize entire text
        self.tokens = tokenizer.encode(text)
    
    def __len__(self):
        return len(self.tokens) - self.max_seq_length
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.max_seq_length + 1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, labels


def download_tinystories():
    """Download tinystories dataset"""
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    urllib.request.urlretrieve(url, "data.txt")
    print("Downloaded TinyStories to data.txt")
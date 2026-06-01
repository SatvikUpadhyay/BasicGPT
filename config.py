# config.py
import torch
import tiktoken

d_model = 512
d_ff = d_model * 4

enc = tiktoken.get_encoding("cl100k_base")
vocab_size = enc.n_vocab

max_seq_length = 512
batch_size = 8
learning_rate = 1e-3
num_epochs = 3
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

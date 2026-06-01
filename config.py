# config.py
import torch
import tiktoken

d_model = 1024
d_ff = d_model * 8
num_heads = 8
num_layers = 4

enc = tiktoken.get_encoding("cl100k_base")
vocab_size = enc.n_vocab

max_seq_length = 512
batch_size = 8
learning_rate = 1e-3
num_epochs = 20
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

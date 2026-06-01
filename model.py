from turtle import forward
from mpmath import residual
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cf
import tokenizer as tf

device = "cuda" if torch.cuda.is_available else "cpu"
print(f"Using device: {device}")

class TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = Attention(d_model=cf.d_model)
        self.mlp = MLP(d_model=cf.d_model, d_ff=cf.d_ff)
        self.layerNorm1 = nn.LayerNorm(cf.d_model)
        self.layerNorm2 = nn.LayerNorm(cf.d_model)

    def forward(self, x):
        # Apply norm1, then attention, then add residual
        res = x # Residual
        x = self.layerNorm1(x)
        x = self.attention(x)
        x += res
        
        # Apply norm2, then MLP, then add residual
        res = x
        x = self.layerNorm2(x)
        x = self.mlp(x)
        x += res       

        return x

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        x = F.scaled_dot_product_attention(Q, K, V)
        return x


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x


class BasicGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Token embedding layer
        self.embedding = nn.Embedding(cf.vocab_size, cf.d_model)
        # Stack multiple transformer blocks
        self.block = TransformerBlock()
        # Final layer norm
        self.layernorm = nn.LayerNorm(cf.d_model)
        # Output projection to vocabulary
        self.o_proj = nn.Linear(cf.d_model, cf.vocab_size)

    def forward(self, input_ids):
        # Embed tokens
        x = self.embedding(input_ids)
    
        # Pass through each transformer block
        x = self.block(x)
        
        # Apply final layer norm
        x = self.layernorm(x)
    
        # Project to vocab size
        x = self.o_proj(x)
    
        # Return logits
        return x

model = BasicGPT()
input_ids = torch.randint(0, cf.vocab_size, (2, 512))  # batch_size=2, seq_len=512
output = model(input_ids)
print(output.shape)  # Should be (2, 512, 10000)

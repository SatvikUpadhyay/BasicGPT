import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TextDataset
from model import BasicGPT
import config as cf
import torch.nn.functional as F
from dataset import TextDataset, download_tinystories
import tiktoken

def model_summary(model):
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print(f"{name}: {num_params:,}")
    print(f"\nTotal: {total_params:,}")

download_tinystories()

# Load tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Load text and create dataset
text = open("data.txt").read()
dataset = TextDataset(text, tokenizer, max_seq_length=cf.max_seq_length)
dataloader = DataLoader(dataset, batch_size=cf.batch_size, shuffle=True)

# Initialize model and optimizer
model = BasicGPT().to(cf.device)
optimizer = optim.Adam(model.parameters(), lr=cf.learning_rate)

model_summary(model)

# Training loop with forward pass, loss, backward pass
for epoch in range(cf.num_epochs):
    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids, labels = input_ids.to(cf.device), labels.to(cf.device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        loss = F.cross_entropy(logits.view(-1, cf.vocab_size), labels.view(-1), label_smoothing=0.1)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "model_weights.pt")
    print("Model saved to model_weights.pt")


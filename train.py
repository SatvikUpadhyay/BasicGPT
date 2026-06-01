import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import TextDataset, download_wiki
from model import BasicGPT
import config as cf
import torch.nn.functional as F
import tiktoken

def model_summary(model):
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print(f"{name}: {num_params:,}")
    print(f"\nTotal: {total_params:,}")

# Download and load data
download_wiki()

tokenizer = tiktoken.get_encoding("cl100k_base")
text = open("wikitext.txt").read()
dataset = TextDataset(text, tokenizer, max_seq_length=cf.max_seq_length)

# Split into train/val (90/10)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=cf.batch_size)

# Initialize model and optimizer
model = BasicGPT().to(cf.device)
optimizer = optim.Adam(model.parameters(), lr=cf.learning_rate)
model_summary(model)

# Training loop
for epoch in range(cf.num_epochs):
    # Training
    train_loss = 0
    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        input_ids, labels = input_ids.to(cf.device), labels.to(cf.device)
        
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, cf.vocab_size), labels.view(-1), label_smoothing=0.1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    
    # Validation
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids, labels = input_ids.to(cf.device), labels.to(cf.device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, cf.vocab_size), labels.view(-1), label_smoothing=0.1)
            val_loss += loss.item()
    model.train()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"\nEpoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save checkpoint
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")
    print(f"Model saved to model_epoch_{epoch+1}.pt\n")
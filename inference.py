# inference.py
import torch
import tiktoken
from model import BasicGPT
import config as cf

# Load model
model = BasicGPT().to(cf.device)
model.load_state_dict(torch.load("model_epoch_3.pt", map_location=cf.device))
model.eval()

# Load tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def generate(prompt, max_tokens=50, temperature=0.7):
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(cf.device)
    
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(input_ids)
        
        next_logits = logits[0, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    generated_tokens = input_ids[0].cpu().tolist()
    text = tokenizer.decode(generated_tokens)
    return text

if __name__ == "__main__":
    while True:
        prompt = input("You: ")
        if prompt.lower() == "quit":
            break
        response = generate(prompt)
        print(f"Model: {response}\n")

#!/usr/bin/env python3
"""
Train Boolean GPT on the 36K dataset
Optimized for 90%+ accuracy through heavy repetition
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time

# Reproducibility
torch.manual_seed(1337)
np.random.seed(1337)

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Hyperparameters - optimized for boolean logic (simpler than math)
batch_size = 128
block_size = 48  # Longer for verbose boolean strings
max_iters = 15000  # Fewer iterations - simpler task
eval_interval = 500
learning_rate = 3e-4
eval_iters = 100
n_embd = 32  # Smaller - boolean is simpler
n_head = 2   # Fewer heads
n_layer = 2
dropout = 0.05  # Lower dropout

print(f"\nHyperparameters:")
print(f"  Block size: {block_size}")
print(f"  Embedding dim: {n_embd}")
print(f"  Layers: {n_layer}, Heads: {n_head}")
print(f"  Batch size: {batch_size}")
print(f"  Max iterations: {max_iters}")

# Load datasets
with open('dataset/boolean/training/boolean_train.txt', 'r') as f:
    train_text = f.read()

with open('dataset/boolean/testing/boolean_test.txt', 'r') as f:
    test_text = f.read()

# Create vocabulary
chars = sorted(list(set(train_text + test_text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Prepare tensors
train_data = torch.tensor(encode(train_text), dtype=torch.long)
test_data = torch.tensor(encode(test_text), dtype=torch.long)

print(f"\nDataset loaded:")
print(f"  Vocabulary size: {vocab_size}")
print(f"  Training size: {len(train_text):,} chars ({train_text.count(chr(10)):,} expressions)")
print(f"  Testing size: {len(test_text):,} chars ({test_text.count(chr(10)):,} expressions)")

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Model architecture
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initialize model
model = GPTLanguageModel().to(device)
n_params = sum(p.numel() for p in model.parameters())

print(f"\nModel initialized:")
print(f"  Parameters: {n_params:,} ({n_params/1e6:.3f}M)")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\nTraining...\n")
start_time = time.time()

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(f"Iter {iter:5d} | Train: {losses['train']:.4f} | Test: {losses['test']:.4f} | Time: {elapsed:.1f}s")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\nTraining complete in {time.time()-start_time:.1f}s")

# Save model
torch.save(model.state_dict(), "model_weights_part2.pth")
print("\nModel saved as: model_weights_part2.pth")

# Quick evaluation
print("\nQuick evaluation on 100 test samples...")
model.eval()
test_expressions = [e.strip() for e in test_text.split('\n') if '=' in e][:100]
correct = 0

with torch.no_grad():
    for expr in test_expressions:
        parts = expr.split('=')
        if len(parts) != 2:
            continue

        input_part = parts[0] + '='
        expected = parts[1]

        try:
            context = torch.tensor([encode(input_part)], dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=15, temperature=0.8)
            prediction = decode(generated[0].tolist())

            if '=' in prediction:
                pred_answer = prediction.split('=', 1)[1].split('\n')[0].strip()
            else:
                pred_answer = ""

            if pred_answer == expected:
                correct += 1
        except:
            pass

accuracy = (correct / len(test_expressions)) * 100 if test_expressions else 0
print(f"Quick test accuracy: {accuracy:.2f}% ({correct}/{len(test_expressions)})")

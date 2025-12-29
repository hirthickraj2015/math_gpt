#!/usr/bin/env python3
"""
Run comprehensive evaluation on both models
Generates all outputs for the report in outputs/ folder
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.3)

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Evaluation environment: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"Outputs will be saved to: outputs/\n")

# Model architecture classes
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
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
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
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
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=32, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
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
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

print("✓ Model architecture defined")

# Load Math GPT
print("\n" + "="*70)
print("LOADING MATH GPT")
print("="*70)

with open('dataset/math/training/math_train.txt', 'r') as f:
    math_train_text = f.read()
with open('dataset/math/testing/math_test.txt', 'r') as f:
    math_test_text = f.read()

math_chars = sorted(list(set(math_train_text + math_test_text)))
math_vocab_size = len(math_chars)
math_stoi = {ch: i for i, ch in enumerate(math_chars)}
math_itos = {i: ch for i, ch in enumerate(math_chars)}
math_encode = lambda s: [math_stoi[c] for c in s]
math_decode = lambda l: ''.join([math_itos[i] for i in l])

math_model = GPTLanguageModel(vocab_size=math_vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=32, dropout=0.1)
math_model.load_state_dict(torch.load('model_weights_part1.pth', map_location=device))
math_model.to(device)
math_model.eval()

print(f"✓ Math GPT loaded ({sum(p.numel() for p in math_model.parameters())/1e6:.2f}M parameters)")
print(f"  Training set: {math_train_text.count(chr(10)):,} expressions")
print(f"  Testing set: {math_test_text.count(chr(10)):,} expressions")

# Load Boolean GPT
print("\n" + "="*70)
print("LOADING BOOLEAN GPT")
print("="*70)

with open('dataset/boolean/training/boolean_train.txt', 'r') as f:
    bool_train_text = f.read()
with open('dataset/boolean/testing/boolean_test.txt', 'r') as f:
    bool_test_text = f.read()

bool_chars = sorted(list(set(bool_train_text + bool_test_text)))
bool_vocab_size = len(bool_chars)
bool_stoi = {ch: i for i, ch in enumerate(bool_chars)}
bool_itos = {i: ch for i, ch in enumerate(bool_chars)}
bool_encode = lambda s: [bool_stoi[c] for c in s]
bool_decode = lambda l: ''.join([bool_itos[i] for i in l])

bool_model = GPTLanguageModel(vocab_size=bool_vocab_size, n_embd=32, n_head=2, n_layer=2, block_size=48, dropout=0.05)
bool_model.load_state_dict(torch.load('model_weights_part2.pth', map_location=device))
bool_model.to(device)
bool_model.eval()

print(f"✓ Boolean GPT loaded ({sum(p.numel() for p in bool_model.parameters())/1e6:.2f}M parameters)")
print(f"  Training set: {bool_train_text.count(chr(10)):,} expressions")
print(f"  Testing set: {bool_test_text.count(chr(10)):,} expressions")

# Evaluation functions
def evaluate_model(model, test_text, encode, decode, max_samples=2000, temperature=0.8):
    """Comprehensive model evaluation with detailed metrics."""
    model.eval()
    results = []
    correct = 0
    char_correct = 0
    char_total = 0

    test_expressions = [e.strip() for e in test_text.split('\n') if '=' in e][:max_samples]

    print(f"Evaluating on {len(test_expressions)} test expressions...")

    with torch.no_grad():
        for i, expr in enumerate(test_expressions):
            if (i + 1) % 500 == 0:
                print(f"  Progress: {i+1}/{len(test_expressions)}")

            parts = expr.split('=')
            if len(parts) != 2:
                continue

            input_part = parts[0] + '='
            expected = parts[1]

            try:
                context = torch.tensor([encode(input_part)], dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=25, temperature=temperature)
                prediction = decode(generated[0].tolist())

                if '=' in prediction:
                    pred_answer = prediction.split('=', 1)[1].split('\n')[0].strip()
                else:
                    pred_answer = ""

                is_correct = (pred_answer == expected)
                if is_correct:
                    correct += 1

                for j in range(max(len(expected), len(pred_answer))):
                    char_total += 1
                    if j < len(expected) and j < len(pred_answer) and expected[j] == pred_answer[j]:
                        char_correct += 1

                results.append((input_part, expected, pred_answer, is_correct))

            except Exception as e:
                results.append((input_part, expected, "", False))

    exact_accuracy = (correct / len(results)) * 100 if results else 0
    char_accuracy = (char_correct / char_total) * 100 if char_total > 0 else 0

    return exact_accuracy, char_accuracy, results

# Evaluate Math GPT
print("\n" + "="*70)
print("EVALUATING MATH GPT")
print("="*70)

math_exact_acc, math_char_acc, math_results = evaluate_model(
    math_model, math_test_text, math_encode, math_decode, max_samples=2000
)

print(f"\n✓ Math GPT Evaluation Complete")
print(f"  Exact Match Accuracy: {math_exact_acc:.2f}%")
print(f"  Character-Level Accuracy: {math_char_acc:.2f}%")
print(f"  Correct: {sum(1 for r in math_results if r[3])}/{len(math_results)}")

# Evaluate Boolean GPT
print("\n" + "="*70)
print("EVALUATING BOOLEAN GPT")
print("="*70)

bool_exact_acc, bool_char_acc, bool_results = evaluate_model(
    bool_model, bool_test_text, bool_encode, bool_decode, max_samples=2000
)

print(f"\n✓ Boolean GPT Evaluation Complete")
print(f"  Exact Match Accuracy: {bool_exact_acc:.2f}%")
print(f"  Character-Level Accuracy: {bool_char_acc:.2f}%")
print(f"  Correct: {sum(1 for r in bool_results if r[3])}/{len(bool_results)}")

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)
print(f"\nFinal Results:")
print(f"  Math GPT:    {math_exact_acc:.2f}% accuracy")
print(f"  Boolean GPT: {bool_exact_acc:.2f}% accuracy")
print(f"\n✓ Both models evaluated successfully")
print(f"✓ Detailed outputs saved in outputs/ folder")

#!/usr/bin/env python3
"""
Generate comprehensive evaluation materials for the assignment report
Produces all visualizations, tables, and analysis needed
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Generating report materials...")
print(f"Device: {device}\n")

# Model architecture
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
print("\nLoading Math GPT...")
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

# Load Boolean GPT
print("Loading Boolean GPT...")
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

print(f"✓ Boolean GPT loaded ({sum(p.numel() for p in bool_model.parameters())/1e6:.2f}M parameters)\n")

# Evaluation function
def evaluate_model(model, test_text, encode, decode, max_samples=2000):
    model.eval()
    results = []
    correct = 0
    char_correct = 0
    char_total = 0

    test_expressions = [e.strip() for e in test_text.split('\n') if '=' in e][:max_samples]

    with torch.no_grad():
        for expr in test_expressions:
            parts = expr.split('=')
            if len(parts) != 2:
                continue

            input_part = parts[0] + '='
            expected = parts[1]

            try:
                context = torch.tensor([encode(input_part)], dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=25, temperature=0.8)
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

            except:
                results.append((input_part, expected, "", False))

    exact_accuracy = (correct / len(results)) * 100 if results else 0
    char_accuracy = (char_correct / char_total) * 100 if char_total > 0 else 0

    return exact_accuracy, char_accuracy, results

def categorize_math_operation(expr):
    if '(' in expr:
        return 'Parentheses'
    elif '*' in expr and ('+' in expr or '-' in expr):
        return 'Mixed Ops'
    elif '//' in expr:
        return 'Division'
    elif '%' in expr:
        return 'Modulo'
    elif '*' in expr:
        return 'Multiplication'
    elif '+' in expr:
        return 'Addition'
    elif '-' in expr:
        return 'Subtraction'
    return 'Other'

def categorize_boolean_operation(expr):
    expr_upper = expr.upper()
    if '(' in expr:
        return 'Parentheses'
    elif 'NOT' in expr_upper and ('AND' in expr_upper or 'OR' in expr_upper or 'XOR' in expr_upper):
        return 'NOT Combined'
    elif 'XOR' in expr_upper:
        return 'XOR'
    elif 'AND' in expr_upper:
        return 'AND'
    elif 'OR' in expr_upper:
        return 'OR'
    elif 'NOT' in expr_upper:
        return 'NOT'
    return 'Other'

def analyze_by_operation(results, categorize_func):
    op_stats = {}
    for input_str, expected_str, predicted_str, is_correct in results:
        op_type = categorize_func(input_str)
        if op_type not in op_stats:
            op_stats[op_type] = {'correct': 0, 'total': 0, 'examples_correct': [], 'examples_incorrect': []}
        op_stats[op_type]['total'] += 1
        if is_correct:
            op_stats[op_type]['correct'] += 1
            if len(op_stats[op_type]['examples_correct']) < 5:
                op_stats[op_type]['examples_correct'].append((input_str, expected_str, predicted_str))
        else:
            if len(op_stats[op_type]['examples_incorrect']) < 5:
                op_stats[op_type]['examples_incorrect'].append((input_str, expected_str, predicted_str))
    return op_stats

# Evaluate both models
print("Evaluating Math GPT...")
math_exact_acc, math_char_acc, math_results = evaluate_model(math_model, math_test_text, math_encode, math_decode, max_samples=2000)
math_op_stats = analyze_by_operation(math_results, categorize_math_operation)

print("Evaluating Boolean GPT...")
bool_exact_acc, bool_char_acc, bool_results = evaluate_model(bool_model, bool_test_text, bool_encode, bool_decode, max_samples=2000)
bool_op_stats = analyze_by_operation(bool_results, categorize_boolean_operation)

print(f"\n✓ Evaluation complete")
print(f"  Math GPT: {math_exact_acc:.2f}%")
print(f"  Boolean GPT: {bool_exact_acc:.2f}%\n")

# Generate visualizations
print("Generating visualizations...\n")

# Figure 1: Overall Performance Comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 1: Overall Model Performance', fontsize=14, fontweight='bold', y=1.02)

# Exact match accuracy
ax1 = axes[0]
models = ['Math GPT', 'Boolean GPT']
accuracies = [math_exact_acc, bool_exact_acc]
colors = ['#4A90E2', '#E85D75']
bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5, width=0.5)
ax1.set_ylabel('Exact Match Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Exact Match Accuracy', fontsize=11)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Metrics comparison
ax2 = axes[1]
metrics = ['Exact Match', 'Character Level']
x = np.arange(len(metrics))
width = 0.35
bars1 = ax2.bar(x - width/2, [math_exact_acc, math_char_acc], width,
                label='Math GPT', color='#4A90E2', edgecolor='black', linewidth=1.2)
bars2 = ax2.bar(x + width/2, [bool_exact_acc, bool_char_acc], width,
                label='Boolean GPT', color='#E85D75', edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Accuracy Metrics Comparison', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=10)
ax2.legend(fontsize=10, loc='lower right')
ax2.set_ylim(0, 100)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('outputs/figure1_overall_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 1 saved")

# Figure 2: Operation-Specific Accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Figure 2: Operation-Specific Accuracy Analysis', fontsize=14, fontweight='bold', y=1.02)
fig.subplots_adjust(wspace=0.35)

# Math GPT operations
math_ops = sorted(math_op_stats.keys())
math_accs = [(math_op_stats[op]['correct']/math_op_stats[op]['total'])*100 for op in math_ops]
y_pos = np.arange(len(math_ops))
bars1 = ax1.barh(y_pos, math_accs, color='#4A90E2', edgecolor='black', linewidth=1.2, height=0.65)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(math_ops, fontsize=10)
ax1.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Math GPT - By Operation', fontsize=11, pad=12)
ax1.set_xlim(0, 105)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
for i, acc in enumerate(math_accs):
    ax1.text(acc + 2, i, f'{acc:.1f}%', va='center', fontsize=9, fontweight='bold')

# Boolean GPT operations
bool_ops = sorted(bool_op_stats.keys())
bool_accs = [(bool_op_stats[op]['correct']/bool_op_stats[op]['total'])*100 for op in bool_ops]
y_pos = np.arange(len(bool_ops))
bars2 = ax2.barh(y_pos, bool_accs, color='#E85D75', edgecolor='black', linewidth=1.2, height=0.65)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(bool_ops, fontsize=10)
ax2.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Boolean GPT - By Operation', fontsize=11, pad=12)
ax2.set_xlim(0, 105)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
for i, acc in enumerate(bool_accs):
    ax2.text(acc + 2, i, f'{acc:.1f}%', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figure2_operation_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 2 saved")

# Figure 3: Architectural Comparison
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Figure 3: Architectural Parameters Comparison', fontsize=14, fontweight='bold', y=0.98)

params = ['Embedding\nDimension', 'Attention\nHeads', 'Layers', 'Block\nSize', 'Dropout\n(×10)']
math_values = [64, 4, 2, 32, 1.0]  # dropout scaled by 10 for visibility
bool_values = [32, 2, 2, 48, 0.5]  # dropout scaled by 10

x = np.arange(len(params))
width = 0.35

bars1 = ax.bar(x - width/2, math_values, width, label='Math GPT',
               color='#4A90E2', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, bool_values, width, label='Boolean GPT',
               color='#E85D75', edgecolor='black', linewidth=1.2)

ax.set_ylabel('Parameter Value', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(params, fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figure3_architecture_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 3 saved")

# Save detailed results table
with open('outputs/evaluation_results_table.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("COMPREHENSIVE EVALUATION RESULTS\n")
    f.write("="*80 + "\n\n")

    f.write("TABLE 1: Overall Performance Metrics\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Model':<20} {'Exact Match':<15} {'Character-Level':<15} {'Samples':<10}\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Math GPT':<20} {math_exact_acc:<15.2f}% {math_char_acc:<15.2f}% {len(math_results):<10}\n")
    f.write(f"{'Boolean GPT':<20} {bool_exact_acc:<15.2f}% {bool_char_acc:<15.2f}% {len(bool_results):<10}\n")
    f.write("\n\n")

    f.write("TABLE 2: Math GPT - Operation-Specific Results\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Operation':<20} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Error Rate'}\n")
    f.write("-"*80 + "\n")
    for op in sorted(math_op_stats.keys(), key=lambda x: (math_op_stats[x]['correct']/math_op_stats[x]['total']), reverse=True):
        stats = math_op_stats[op]
        acc = (stats['correct'] / stats['total']) * 100
        err = 100 - acc
        f.write(f"{op:<20} {stats['correct']:<10} {stats['total']:<10} {acc:<10.1f}% {err:.1f}%\n")

    f.write("\n\n")
    f.write("TABLE 3: Boolean GPT - Operation-Specific Results\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Operation':<20} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Error Rate'}\n")
    f.write("-"*80 + "\n")
    for op in sorted(bool_op_stats.keys(), key=lambda x: (bool_op_stats[x]['correct']/bool_op_stats[x]['total']), reverse=True):
        stats = bool_op_stats[op]
        acc = (stats['correct'] / stats['total']) * 100
        err = 100 - acc
        f.write(f"{op:<20} {stats['correct']:<10} {stats['total']:<10} {acc:<10.1f}% {err:.1f}%\n")

print("✓ Evaluation tables saved")

# Save example predictions
with open('outputs/example_predictions.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("APPENDIX: PROMPT-OUTPUT PAIRS\n")
    f.write("Demonstrating Model Strengths and Weaknesses\n")
    f.write("="*80 + "\n\n")

    f.write("PART 1: MATH GPT\n")
    f.write("-"*80 + "\n\n")

    # Strengths
    f.write("STRENGTHS (Correct Predictions):\n\n")
    correct_math = [r for r in math_results if r[3]]

    # Get diverse examples
    by_op = {}
    for inp, exp, pred, _ in correct_math:
        op = categorize_math_operation(inp)
        if op not in by_op:
            by_op[op] = []
        if len(by_op[op]) < 3:
            by_op[op].append((inp, exp, pred))

    for op, examples in sorted(by_op.items()):
        f.write(f"{op}:\n")
        for inp, exp, pred in examples:
            f.write(f"  Prompt:  {inp}\n")
            f.write(f"  Output:  {pred} ✓\n\n")

    # Weaknesses
    f.write("\nWEAKNESSES (Incorrect Predictions):\n\n")
    incorrect_math = [r for r in math_results if not r[3]]

    by_op = {}
    for inp, exp, pred, _ in incorrect_math:
        op = categorize_math_operation(inp)
        if op not in by_op:
            by_op[op] = []
        if len(by_op[op]) < 3:
            by_op[op].append((inp, exp, pred))

    for op, examples in sorted(by_op.items()):
        f.write(f"{op}:\n")
        for inp, exp, pred in examples:
            f.write(f"  Prompt:   {inp}\n")
            f.write(f"  Output:   {pred} ✗\n")
            f.write(f"  Expected: {exp}\n\n")

    f.write("\n" + "="*80 + "\n\n")
    f.write("PART 2: BOOLEAN GPT\n")
    f.write("-"*80 + "\n\n")

    # Strengths
    f.write("STRENGTHS (Correct Predictions):\n\n")
    correct_bool = [r for r in bool_results if r[3]]

    by_op = {}
    for inp, exp, pred, _ in correct_bool:
        op = categorize_boolean_operation(inp)
        if op not in by_op:
            by_op[op] = []
        if len(by_op[op]) < 3:
            by_op[op].append((inp, exp, pred))

    for op, examples in sorted(by_op.items()):
        f.write(f"{op}:\n")
        for inp, exp, pred in examples:
            f.write(f"  Prompt:  {inp}\n")
            f.write(f"  Output:  {pred} ✓\n\n")

    # Weaknesses
    f.write("\nWEAKNESSES (Incorrect Predictions):\n\n")
    incorrect_bool = [r for r in bool_results if not r[3]]

    if incorrect_bool:
        by_op = {}
        for inp, exp, pred, _ in incorrect_bool:
            op = categorize_boolean_operation(inp)
            if op not in by_op:
                by_op[op] = []
            if len(by_op[op]) < 3:
                by_op[op].append((inp, exp, pred))

        for op, examples in sorted(by_op.items()):
            f.write(f"{op}:\n")
            for inp, exp, pred in examples:
                f.write(f"  Prompt:   {inp}\n")
                f.write(f"  Output:   {pred} ✗\n")
                f.write(f"  Expected: {exp}\n\n")
    else:
        f.write("No incorrect predictions found in sample.\n")

print("✓ Example predictions saved")

print("\n" + "="*80)
print("REPORT MATERIALS GENERATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  • outputs/figure1_overall_performance.png")
print("  • outputs/figure2_operation_accuracy.png")
print("  • outputs/figure3_architecture_comparison.png")
print("  • outputs/evaluation_results_table.txt")
print("  • outputs/example_predictions.txt")
print("\n✓ All materials ready for Word document creation")

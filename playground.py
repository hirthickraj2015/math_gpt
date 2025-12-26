import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict
    import json
    import os
    import time
    from tqdm import tqdm
    import wandb
    import evaluation_utils as eval_utils

    # Set random seeds for reproducibility
    torch.manual_seed(1337)
    np.random.seed(1337)

    # Configure plotting
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Device configuration
    if torch.mps.is_available():
        device = 'mps'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return F, device, eval_utils, nn, os, time, torch, wandb


@app.cell
def _(os, wandb):
    # Model hyperparameters
    batch_size = 64          # Number of independent sequences processed in parallel
    block_size = 32          # Maximum context length - reduced from 256 as math expressions are short
    max_iters = 20000        # Total training iterations
    eval_interval = 500      # Evaluate every N iterations
    learning_rate = 5e-4     # Learning rate - slightly higher than default for faster convergence
    eval_iters = 200         # Number of iterations to average for loss estimation
    n_embd = 128             # Embedding dimension - smaller than default as vocabulary is limited
    n_head = 4               # Number of attention heads
    n_layer = 4              # Number of transformer blocks
    dropout = 0.1            # Dropout rate for regularization
    eval_samples = 100       # Number of samples for accuracy evaluation during training

    # Checkpoint configuration
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Block size: {block_size}")
    print(f"  Embedding dim: {n_embd}")
    print(f"  Layers: {n_layer}, Heads: {n_head}")
    print(f"  Dropout: {dropout}")
    print(f"  Learning rate: {learning_rate}")

    # Initialize wandb
    wandb.init(
        project="math-gpt",
        name="part1-arithmetic-solver",
        config={
            "batch_size": batch_size,
            "block_size": block_size,
            "max_iters": max_iters,
            "eval_interval": eval_interval,
            "learning_rate": learning_rate,
            "eval_iters": eval_iters,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "dropout": dropout,
            "architecture": "GPT",
            "task": "arithmetic-solver"
        }
    )
    return (
        batch_size,
        block_size,
        checkpoint_dir,
        dropout,
        eval_interval,
        eval_iters,
        eval_samples,
        learning_rate,
        max_iters,
        n_embd,
        n_head,
        n_layer,
    )


@app.cell
def _():
    # Load datasets
    train_path = 'dataset/math_v2/training/math_train.txt'
    test_path = 'dataset/math_v2/testing/math_test.txt'

    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = f.read()

    with open(test_path, 'r', encoding='utf-8') as f:
        test_text = f.read()

    # Combine for vocabulary creation
    full_text = train_text + test_text

    print(f"Dataset Statistics:")
    print(f"  Training set size: {len(train_text):,} characters")
    print(f"  Testing set size: {len(test_text):,} characters")
    print(f"  Training examples: {train_text.count(chr(10)):,}")  # newline count
    print(f"  Testing examples: {test_text.count(chr(10)):,}")

    # Show sample expressions
    print(f"\nSample expressions from training set:")
    for i, line in enumerate(train_text.split('\n')[:10]):
        print(f"  {line}")

    # Prepare list of test expressions for exact match evaluation
    test_expressions = [line for line in test_text.split('\n') if line.strip()]
    return full_text, test_expressions, test_text, train_text


@app.cell
def _(full_text):
    # Create character-level vocabulary
    chars = sorted(list(set(full_text)))
    vocab_size = len(chars)

    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encoder and decoder functions
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    print(f"Vocabulary:")
    print(f"  Size: {vocab_size}")
    print(f"  Characters: {' '.join(chars)}")
    print(f"\nExample encoding:")
    test_expr = "5+3=8"
    encoded = encode(test_expr)
    print(f"  Original: {test_expr}")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: {decode(encoded)}")
    return chars, decode, encode, itos, stoi, vocab_size


@app.cell
def _(batch_size, block_size, device, encode, test_text, torch, train_text):
    # Encode datasets
    train_data = torch.tensor(encode(train_text), dtype=torch.long)
    test_data = torch.tensor(encode(test_text), dtype=torch.long)

    print(f"Encoded datasets:")
    print(f"  Train tensor shape: {train_data.shape}")
    print(f"  Test tensor shape: {test_data.shape}")

    # Data loading function
    def get_batch(split):
        """
        Generate a batch of data for training or testing.

        Args:
            split: 'train' or 'test'

        Returns:
            x: input sequences (batch_size, block_size)
            y: target sequences (batch_size, block_size)
        """
        data = train_data if split == 'train' else test_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
    return (get_batch,)


@app.cell
def _(
    F,
    block_size,
    device,
    dropout,
    n_embd,
    n_head,
    n_layer,
    nn,
    torch,
    vocab_size,
):
    class Head(nn.Module):
        """Single head of self-attention."""

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
            # Compute attention scores
            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            # Weighted aggregation
            v = self.value(x)
            out = wei @ v
            return out


    class MultiHeadAttention(nn.Module):
        """Multiple heads of self-attention in parallel."""

        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, n_embd)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out


    class FeedForward(nn.Module):
        """Feed-forward network with ReLU activation."""

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
        """Transformer block: communication (attention) followed by computation (FFN)."""

        def __init__(self, n_embd, n_head):
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedForward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            # Residual connections with layer normalization
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x


    class GPTLanguageModel(nn.Module):
        """GPT Language Model for arithmetic expressions."""

        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)

            # Initialize weights
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

            # Embeddings
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = tok_emb + pos_emb

            # Transformer blocks
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

        def generate(self, idx, max_new_tokens, temperature=1.0):
            """
            Generate new tokens given a context.

            Args:
                idx: context tokens (B, T)
                max_new_tokens: number of tokens to generate
                temperature: sampling temperature (higher = more random)

            Returns:
                Generated sequence (B, T+max_new_tokens)
            """
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, loss = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

    # Initialize model
    model = GPTLanguageModel()
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Architecture:")
    print(f"  Total parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension: {n_embd}")
    print(f"  Number of layers: {n_layer}")
    print(f"  Number of heads: {n_head}")
    print(f"\nModel loaded on: {device}")
    return (model,)


@app.cell
def _(
    block_size,
    chars,
    checkpoint_dir,
    dropout,
    eval_iters,
    get_batch,
    itos,
    model,
    n_embd,
    n_head,
    n_layer,
    os,
    stoi,
    torch,
    vocab_size,
):
    @torch.no_grad()
    def estimate_loss():
        """
        Estimate loss on train and test sets.
        Averages loss over multiple batches for stability.
        """
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


    def save_checkpoint(model, optimizer, iteration, losses, is_best=False):
        """
        Save model checkpoint with training state.

        Args:
            model: the model to save
            optimizer: optimizer state
            iteration: current training iteration
            losses: dictionary of losses
            is_best: whether this is the best model so far
        """
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'vocab_size': vocab_size,
            'chars': chars,
            'stoi': stoi,
            'itos': itos,
            'hyperparameters': {
                'n_embd': n_embd,
                'n_head': n_head,
                'n_layer': n_layer,
                'block_size': block_size,
                'dropout': dropout,
            }
        }

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}.pt')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  New best model saved! (test loss: {losses['test']:.4f})")


    def load_checkpoint(checkpoint_path, model, optimizer=None):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: path to checkpoint file
            model: model to load weights into
            optimizer: optimizer to load state into (optional)

        Returns:
            iteration: training iteration of checkpoint
            losses: loss history
        """
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['iteration'], checkpoint['losses']
    return estimate_loss, save_checkpoint


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 Training Loop with Progress Tracking
    """)
    return


@app.cell
def _(
    decode,
    device,
    encode,
    estimate_loss,
    eval_interval,
    eval_samples,
    eval_utils,
    get_batch,
    learning_rate,
    max_iters,
    model,
    save_checkpoint,
    test_expressions,
    time,
    torch,
    wandb,
):
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

    # Training history
    train_losses = []
    test_losses = []
    iterations = []
    best_test_loss = float('inf')

    print("Starting training...")
    print(f"Total iterations: {max_iters}")
    print(f"Evaluation interval: {eval_interval}")
    print("-" * 60)

    start_time = time.time()

    for iter in range(max_iters):
        # Evaluate periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()

            # Calculate accuracy using generation
            print(f"Evaluating accuracy at iteration {iter}...")
            accuracy, results = eval_utils.evaluate_exact_match(
                model, 
                test_expressions, 
                encode, 
                decode, 
                device=device, 
                max_samples=eval_samples,
                temperature=0.8
            )

            # Calculate digit accuracy
            digit_acc = eval_utils.calculate_digit_accuracy(results)

            train_losses.append(losses['train'])
            test_losses.append(losses['test'])
            iterations.append(iter)

            elapsed = time.time() - start_time
            print(f"Iter {iter:5d} | Train loss: {losses['train']:.4f} | Test loss: {losses['test']:.4f} | Accuracy: {accuracy:.1f}% | Digit Acc: {digit_acc:.1f}% | Time: {elapsed:.1f}s")

            # Log to wandb
            wandb.log({
                "train/loss": losses['train'],
                "test/loss": losses['test'],
                "test/accuracy": accuracy,
                "test/digit_accuracy": digit_acc,
                "iter": iter,
                "elapsed_time": elapsed
            }, step=iter)

            # Save checkpoint
            is_best = losses['test'] < best_test_loss
            if is_best:
                best_test_loss = losses['test']

            if iter % (eval_interval * 2) == 0:  # Save every 1000 iterations
                save_checkpoint(model, optimizer, iter, losses, is_best)

        # Training step
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log training loss to wandb
        wandb.log({"train/step_loss": loss.item()}, step=iter)

        # Log evaluation metrics to wandb
        if iter % eval_interval == 0 or iter == max_iters - 1:
            # The code already calls estimate_loss() above in the if block
            # But 'losses' is defined inside that block. 
            # I will move the log call inside that block.
            pass


    print("-" * 60)
    print(f"Training complete!")
    print(f"Best test loss: {best_test_loss:.4f}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final test loss: {test_losses[-1]:.4f}")
    print(f"Total training time: {time.time() - start_time:.1f}s")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Save Final Model and Generate Report Materials

    Save the trained model weights for submission and generate comprehensive evaluation materials.
    """)
    return


@app.cell
def _(model, torch, wandb):
    # Save model weights as required by assignment
    torch.save(model.state_dict(), "model_weights_part1.pth")

    print("Model saved as: model_weights_part1.pth")
    print("\nTo load this model:")
    print("  model = GPTLanguageModel()")
    print("  model.load_state_dict(torch.load('model_weights_part1.pth'))")
    print("  model.eval()")

    # Log model weights to wandb as an artifact
    if wandb.run is not None:
        artifact = wandb.Artifact('math-gpt-model', type='model')
        artifact.add_file('model_weights_part1.pth')
        wandb.log_artifact(artifact)
    return


@app.cell
def _(wandb):
    # This cell signals the end of the notebook and finishes the wandb run
    if wandb.run is not None:
        wandb.finish()
    return


@app.cell
def _(test_text):
    test_list = []
    for index, l in enumerate(test_text.split('\n')[:100]):
        test_list.append(l)

    print(test_list)
    return (test_list,)


@app.cell
def _(decode, device, encode, model, torch):
    context = torch.tensor(encode("1-2-4="), dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(context, max_new_tokens=10, temperature=0.01)
    prediction = decode(generated[0].tolist())
    print("prediction",prediction)
    prediction = prediction.split('=')[1].split('\n')[0]
    print("prediction after",prediction)
    return


@app.cell
def _(decode, device, encode, model, test_list, torch):
    from evaluation_utils import evaluate_exact_match

    weights = torch.load('model_weights_part1.pth', map_location=device)
    model.load_state_dict(weights)

    final_accuracy, final_results = evaluate_exact_match(model, test_list, encode, decode, device=device)

    print(f"Total Accuracy: {final_accuracy:.2f}%")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

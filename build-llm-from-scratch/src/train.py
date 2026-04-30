"""
train.py — Training loop for TinyGPT
======================================
Teaches the model to predict the next token by minimising cross-entropy loss.

Concepts covered:
  - Cross-entropy loss for language modelling
  - Mini-batch gradient descent
  - AdamW optimiser and weight decay
  - Gradient clipping (prevents exploding gradients)
  - Cosine learning rate schedule
  - Checkpointing — saving the best model automatically

Run:
  python src/train.py
"""

import json, time, torch, torch.nn as nn
from pathlib import Path
from tokenizer import build_from_dataset, CharTokenizer
from model import TinyGPT

# ── Hyperparameters ────────────────────────────────────────────────── #
# Tune these to improve your model

DATASET   = "data/train.jsonl"
TOK_SAVE  = "tokenizer.json"
MODEL_SAVE= "model.pt"

D_MODEL   = 128    # embedding width — increase for better quality
N_HEADS   = 4      # attention heads
N_LAYERS  = 4      # transformer depth
CTX_LEN   = 256    # context window
DROPOUT   = 0.1

BATCH     = 32     # examples per gradient step
LR        = 3e-4   # learning rate
MAX_STEPS = 5_000  # training steps (increase to 10K for better quality)
EVAL_EVERY= 500    # print loss every N steps
# ────────────────────────────────────────────────────────────────────── #


def load_data(path: str, tok: CharTokenizer) -> torch.Tensor:
    """Encode entire dataset as one long token sequence."""
    sep = tok.sep_id
    tokens = []
    with open(path) as f:
        for line in f:
            obj  = json.loads(line.strip())
            p_ids= tok.encode(obj["prompt"])
            c_ids= tok.encode(obj["code"])
            tokens += p_ids + [sep] + c_ids + [sep]
    data = torch.tensor(tokens, dtype=torch.long)
    print(f"Dataset: {len(data):,} tokens")
    return data


def get_batch(data: torch.Tensor, batch: int, ctx: int, device: str):
    """Sample random (input, target) pairs from the data."""
    starts = torch.randint(0, len(data) - ctx - 1, (batch,))
    x = torch.stack([data[i:i+ctx]   for i in starts])
    y = torch.stack([data[i+1:i+ctx+1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, data, batch, ctx, device, n=20) -> float:
    model.eval()
    losses = []
    for _ in range(n):
        x, y   = get_batch(data, batch, ctx, device)
        logits = model(x)
        loss   = nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print("─" * 50)

    # Build tokenizer
    tok  = build_from_dataset(DATASET)
    tok.save(TOK_SAVE)

    # Load and encode dataset
    data = load_data(DATASET, tok)

    # Create model
    model = TinyGPT(
        vocab_size=tok.vocab_size, d_model=D_MODEL,
        n_heads=N_HEADS, n_layers=N_LAYERS,
        ctx_len=CTX_LEN, dropout=DROPOUT,
    ).to(device)
    print(f"Model  : {model.count_params():,} parameters")
    print("─" * 50)

    # Optimiser and learning rate schedule
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_STEPS)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")
    t0 = time.time()

    for step in range(1, MAX_STEPS + 1):
        # Forward pass
        x, y   = get_batch(data, BATCH, CTX_LEN, device)
        logits = model(x)
        loss   = loss_fn(logits.view(-1, tok.vocab_size), y.view(-1))

        # Backward pass
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        opt.step()
        sched.step()

        if step % EVAL_EVERY == 0:
            avg_loss = eval_loss(model, data, BATCH, CTX_LEN, device)
            elapsed  = time.time() - t0
            print(f"Step {step:5d} | Loss: {avg_loss:.4f} | "
                  f"LR: {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "model_state": model.state_dict(),
                    "config": dict(vocab_size=tok.vocab_size, d_model=D_MODEL,
                                   n_heads=N_HEADS, n_layers=N_LAYERS,
                                   ctx_len=CTX_LEN),
                    "step": step, "loss": best_loss,
                }, MODEL_SAVE)
                print(f"         ✓ Saved best model  (loss={best_loss:.4f})")

    print("─" * 50)
    print(f"Done! Best loss: {best_loss:.4f} · Saved → {MODEL_SAVE}")


if __name__ == "__main__":
    train()

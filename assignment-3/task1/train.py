"""
train.py — Task 1: Text Generation
Trains all four model variants and saves checkpoints + a results CSV.

Variants trained:
  lstm_learned   — LSTM  + learned embeddings
  lstm_glove     — LSTM  + frozen GloVe-100
  gru_learned    — GRU   + learned embeddings
  gru_onehot     — GRU   + one-hot encoding  (baseline)

Usage:
    python train.py                  # train all variants
    python train.py --model lstm_glove --epochs 10
"""

import argparse
import csv
import math
import os
import time

import torch
import torch.nn as nn

from data_loader import load_data, get_batch, SEQ_LEN, BATCH_SIZE
from models import LSTMModel, GRUModel, build_glove_embedding, build_onehot_embedding


# ── Config ───────────────────────────────────────────────────────────────────
EPOCHS      = 15
LR          = 20.0          # SGD learning rate (standard for AWD-LSTM style)
CLIP        = 0.25          # gradient clipping
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.5
EMBED_DIM   = 256           # for learned embeddings
GLOVE_DIM   = 100
SAVE_DIR    = "checkpoints"
RESULTS_CSV = "results_task1.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Training Loop ────────────────────────────────────────────────────────────

def train_epoch(model, train_data, criterion, optimizer, vocab_size):
    model.train()
    total_loss = 0.0
    hidden = model.init_hidden(BATCH_SIZE, device)

    for i in range(0, train_data.size(0) - 1, SEQ_LEN):
        x, y = get_batch(train_data, i)
        x, y = x.to(device), y.to(device)

        # Detach hidden so we don't backprop through the entire corpus
        hidden = model.detach_hidden(hidden)

        optimizer.zero_grad()
        log_probs, hidden = model(x, hidden)

        # y must be flattened to match log_probs shape (seq*batch,)
        loss = criterion(log_probs, y.reshape(-1))
        loss.backward()

        # Clip gradients — critical for RNNs
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        total_loss += loss.item()

    num_batches = train_data.size(0) // SEQ_LEN
    return total_loss / num_batches


def evaluate(model, data, criterion):
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(BATCH_SIZE, device)

    with torch.no_grad():
        for i in range(0, data.size(0) - 1, SEQ_LEN):
            x, y = get_batch(data, i)
            x, y = x.to(device), y.to(device)
            hidden = model.detach_hidden(hidden)
            log_probs, hidden = model(x, hidden)
            loss = criterion(log_probs, y.reshape(-1))
            total_loss += loss.item()

    num_batches = data.size(0) // SEQ_LEN
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


# ── Model Factory ────────────────────────────────────────────────────────────

def build_model(name: str, vocab, vocab_size: int):
    """Return an untrained model for the given variant name."""
    if name == "lstm_learned":
        return LSTMModel(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)

    elif name == "lstm_glove":
        emb = build_glove_embedding(vocab, glove_dim=GLOVE_DIM, freeze=True)
        return LSTMModel(vocab_size, GLOVE_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, embedding=emb)

    elif name == "gru_learned":
        return GRUModel(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)

    elif name == "gru_onehot":
        emb = build_onehot_embedding(vocab_size)
        # one-hot is huge → reduce hidden dim to keep memory reasonable
        return GRUModel(vocab_size, vocab_size, hidden_dim=128, num_layers=1,
                        dropout=0.0, embedding=emb)

    else:
        raise ValueError(f"Unknown model name: {name}")


# ── Main Training Run ────────────────────────────────────────────────────────

def run(model_name: str, train_data, val_data, test_data, vocab):
    vocab_size = len(vocab)
    model = build_model(model_name, vocab, vocab_size).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  Model : {model_name}")
    print(f"  Params: {total_params:,}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_ppl = float("inf")
    os.makedirs(SAVE_DIR, exist_ok=True)
    ckpt_path = os.path.join(SAVE_DIR, f"{model_name}.pt")

    epoch_results = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_data, criterion, optimizer, vocab_size)
        val_loss, val_ppl = evaluate(model, val_data, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:>2}/{EPOCHS} | "
              f"train_loss={train_loss:.3f} | "
              f"val_loss={val_loss:.3f} | "
              f"val_ppl={val_ppl:.2f} | "
              f"{elapsed:.1f}s")

        epoch_results.append({
            "model": model_name,
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
        })

        # Save best checkpoint
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best model (val_ppl={val_ppl:.2f})")

    # Load best and run test set
    model.load_state_dict(torch.load(ckpt_path))
    test_loss, test_ppl = evaluate(model, test_data, criterion)
    print(f"\n  Test perplexity: {test_ppl:.2f}")

    return epoch_results, test_ppl


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    global EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["all", "lstm_learned", "lstm_glove", "gru_learned", "gru_onehot"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    EPOCHS = args.epochs

    print("Loading data…")
    train_data, val_data, test_data, vocab = load_data(BATCH_SIZE)

    variants = (["lstm_learned", "lstm_glove", "gru_learned", "gru_onehot"]
                if args.model == "all" else [args.model])

    all_results = []
    summary = []

    for name in variants:
        epoch_rows, test_ppl = run(name, train_data, val_data, test_data, vocab)
        all_results.extend(epoch_rows)
        summary.append({"model": name, "test_ppl": round(test_ppl, 2)})

    # Write results CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "epoch", "train_loss", "val_loss", "val_ppl"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved → {RESULTS_CSV}")
    print("\n── Final Test Perplexities ──")
    for row in summary:
        print(f"  {row['model']:20s}: {row['test_ppl']}")


if __name__ == "__main__":
    main()
"""
train.py — Task 2: Machine Translation
Trains LSTM and GRU seq2seq models on Multi30K (EN→DE).
Saves checkpoints and a results CSV.

Usage:
    python train.py                   # train both models
    python train.py --model gru --epochs 15
"""

import argparse
import csv
import math
import os
import time

import torch
import torch.nn as nn

from data_loader import load_data, PAD_IDX, BATCH_SIZE
from models import build_lstm_seq2seq, build_gru_seq2seq


# ── Config ───────────────────────────────────────────────────────────────────
EPOCHS     = 20
LR         = 0.001
CLIP       = 1.0
EMBED_DIM  = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT    = 0.5
SAVE_DIR   = "checkpoints"
RESULTS_CSV = "results_task2.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Train / Eval ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, clip, teacher_forcing=0.5):
    model.train()
    total_loss = 0.0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()

        output = model(src, tgt, teacher_forcing_ratio=teacher_forcing)
        # output: (tgt_len, batch, vocab) — skip <bos> at position 0
        output_flat = output[1:].reshape(-1, output.shape[-1])
        tgt_flat    = tgt[1:].reshape(-1)

        loss = criterion(output_flat, tgt_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            output_flat = output[1:].reshape(-1, output.shape[-1])
            tgt_flat    = tgt[1:].reshape(-1)
            loss = criterion(output_flat, tgt_flat)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(avg_loss)


# ── Main ─────────────────────────────────────────────────────────────────────

def run(model_name, model, train_loader, val_loader):
    """Train one model for EPOCHS and save best checkpoint."""
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    ckpt_path = os.path.join(SAVE_DIR, f"{model_name}.pt")
    best_val_loss = float("inf")
    results = []

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  Model : {model_name}  ({total_params:,} params)")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Gradually reduce teacher forcing over training
        tf_ratio = max(0.3, 0.5 - (epoch - 1) * 0.015)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP, tf_ratio)
        val_loss, val_ppl = evaluate_loss(model, val_loader, criterion)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:>2}/{EPOCHS} | "
              f"train_loss={train_loss:.3f} | "
              f"val_loss={val_loss:.3f} | ppl={val_ppl:.1f} | "
              f"tf={tf_ratio:.2f} | {elapsed:.1f}s")

        results.append({
            "model": model_name,
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.3f})")

    return results


def main():
    global EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=["all", "lstm", "gru"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    EPOCHS = args.epochs

    print("Loading Multi30K data…")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = load_data(BATCH_SIZE)
    src_vs = len(src_vocab)
    tgt_vs = len(tgt_vocab)

    all_results = []

    models_to_train = []
    if args.model in ("all", "lstm"):
        models_to_train.append(("lstm_seq2seq",
            build_lstm_seq2seq(src_vs, tgt_vs, PAD_IDX, device,
                               EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)))
    if args.model in ("all", "gru"):
        models_to_train.append(("gru_seq2seq",
            build_gru_seq2seq(src_vs, tgt_vs, PAD_IDX, device,
                              EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)))

    for name, model in models_to_train:
        results = run(name, model, train_loader, val_loader)
        all_results.extend(results)

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model","epoch","train_loss","val_loss","val_ppl"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved → {RESULTS_CSV}")
    print("Run evaluate.py to compute BLEU scores on the test set.")


if __name__ == "__main__":
    main()
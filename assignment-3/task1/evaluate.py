"""
evaluate.py — Task 1: Text Generation
- Loads a saved checkpoint and reports test perplexity
- Generates sample text completions (qualitative evaluation)

Usage:
    python evaluate.py --model lstm_glove
    python evaluate.py --model gru_learned --prompt "the king of"
    python evaluate.py --all
"""

import argparse
import math
import torch
import torch.nn.functional as F

from data_loader import load_data, get_batch, BATCH_SIZE, SEQ_LEN, tokenizer
from models import LSTMModel, GRUModel, build_glove_embedding, build_onehot_embedding
from train import build_model, HIDDEN_DIM, NUM_LAYERS, DROPOUT, EMBED_DIM, GLOVE_DIM, SAVE_DIR

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Perplexity ───────────────────────────────────────────────────────────────

def compute_perplexity(model, data, criterion):
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

    avg_loss = total_loss / (data.size(0) // SEQ_LEN)
    return math.exp(avg_loss)


# ── Text Generation ──────────────────────────────────────────────────────────

def generate_text(model, vocab, prompt: str, num_words: int = 50, temperature: float = 0.8):
    """
    Auto-regressively generate words following a prompt.

    Temperature:
        < 1.0  → more conservative / repetitive
        = 1.0  → sample directly from model distribution
        > 1.0  → more creative / random
    """
    model.eval()
    itos = vocab.get_itos()   # index → token

    # Encode prompt
    tokens = vocab(tokenizer(prompt))
    if not tokens:
        tokens = [vocab["<unk>"]]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)

    # Warm up hidden state on the prompt
    hidden = model.init_hidden(1, device)
    with torch.no_grad():
        _, hidden = model(input_tensor, hidden)

    # Generate
    generated = list(prompt.split())
    current = torch.tensor([[tokens[-1]]], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(num_words):
            log_probs, hidden = model(current, hidden)
            probs = F.softmax(log_probs[-1] / temperature, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            word = itos[next_idx]
            if word == "<eos>":
                break
            generated.append(word)
            current = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return " ".join(generated)


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm_glove",
                        choices=["lstm_learned", "lstm_glove", "gru_learned", "gru_onehot"])
    parser.add_argument("--all", action="store_true", help="Evaluate all variants")
    parser.add_argument("--prompt", default="the president of the united states",
                        help="Seed text for generation")
    parser.add_argument("--words", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    import torch.nn as nn
    criterion = nn.NLLLoss()

    print("Loading data…")
    train_data, val_data, test_data, vocab = load_data(BATCH_SIZE)

    variants = (["lstm_learned", "lstm_glove", "gru_learned", "gru_onehot"]
                if args.all else [args.model])

    print(f"\n{'='*60}")
    for name in variants:
        ckpt = os.path.join(SAVE_DIR, f"{name}.pt")
        if not os.path.exists(ckpt):
            print(f"  [{name}] No checkpoint found at {ckpt} — skipping.")
            continue

        model = build_model(name, vocab, len(vocab)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        ppl = compute_perplexity(model, test_data, criterion)
        print(f"\n  {name}")
        print(f"    Test Perplexity : {ppl:.2f}")

        print(f"    Prompt          : \"{args.prompt}\"")
        generated = generate_text(model, vocab, args.prompt, args.words, args.temperature)
        print(f"    Generated text  :\n")
        # Word-wrap at 70 chars for readability
        words = generated.split()
        line = []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 70:
                print("      " + " ".join(line))
                line = []
        if line:
            print("      " + " ".join(line))

    print(f"\n{'='*60}")
    print("\nInterpretation guide:")
    print("  Perplexity = exp(cross-entropy loss)")
    print("  Lower is better. A perfect model would score 1.0.")
    print("  A random baseline over V vocab tokens ≈ V (e.g. 28,000).")


if __name__ == "__main__":
    main()
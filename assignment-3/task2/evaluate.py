"""
evaluate.py - Task 2: Machine Translation
Computes corpus BLEU on the test set and shows qualitative translation examples.

Usage:
    python evaluate.py                       # evaluate both models
    python evaluate.py --model lstm_seq2seq
"""

import argparse
import os
import torch

from data_loader import load_data, PAD_IDX, BOS_IDX, EOS_IDX, BATCH_SIZE, de_tokenizer, en_tokenizer
from models import build_lstm_seq2seq, build_gru_seq2seq
from train import EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, SAVE_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_OUTPUT_LEN = 50


# ── BLEU (manual, no torchtext dependency) ────────────────────────────────────

def _ngrams(sequence, n):
    return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

def _count_clip(candidate, reference, n):
    cand_ngrams = {}
    for ng in _ngrams(candidate, n):
        cand_ngrams[ng] = cand_ngrams.get(ng, 0) + 1
    ref_ngrams = {}
    for ng in _ngrams(reference, n):
        ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1
    clipped = sum(min(cnt, ref_ngrams.get(ng, 0)) for ng, cnt in cand_ngrams.items())
    total = max(len(candidate) - n + 1, 0)
    return clipped, total

def corpus_bleu(hypotheses, references, max_n=4):
    """
    Compute corpus-level BLEU-4.
    hypotheses : list of token lists
    references : list of token lists (one ref per hypothesis)
    """
    import math
    clipped_counts = [0] * max_n
    total_counts   = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            c, t = _count_clip(hyp, ref, n)
            clipped_counts[n-1] += c
            total_counts[n-1]   += t

    # Brevity penalty
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)

    # Log sum of precisions
    log_avg = 0.0
    for n in range(max_n):
        if clipped_counts[n] == 0 or total_counts[n] == 0:
            return 0.0
        log_avg += math.log(clipped_counts[n] / total_counts[n])

    return bp * math.exp(log_avg / max_n) * 100


# ── Greedy Decode ────────────────────────────────────────────────────────────

def greedy_decode(model, src, tgt_vocab):
    model.eval()
    with torch.no_grad():
        src = src.to(device)
        is_lstm = hasattr(model.encoder, "lstm")

        if is_lstm:
            enc_out, hidden, cell = model.encoder(src)
        else:
            enc_out, hidden = model.encoder(src)
            cell = None

        dec_input = torch.tensor([BOS_IDX], device=device)
        tgt_itos = tgt_vocab.get_itos()
        tokens = []

        for _ in range(MAX_OUTPUT_LEN):
            if cell is not None:
                logits, hidden, cell = model.decoder.forward_step(dec_input, hidden, cell, enc_out)
            else:
                logits, hidden = model.decoder.forward_step(dec_input, hidden, enc_out)

            pred_idx = logits.argmax(dim=1)
            if pred_idx.item() == EOS_IDX:
                break
            tokens.append(tgt_itos[pred_idx.item()])
            dec_input = pred_idx

    return tokens


# ── Evaluation ───────────────────────────────────────────────────────────────

def compute_bleu(model, test_loader, src_vocab, tgt_vocab):
    tgt_itos = tgt_vocab.get_itos()
    special = {"<pad>", "<bos>", "<eos>", "<unk>"}
    hypotheses = []
    references = []

    for src_batch, tgt_batch in test_loader:
        for i in range(src_batch.size(1)):
            src_single = src_batch[:, i].unsqueeze(1)
            hyp = greedy_decode(model, src_single, tgt_vocab)
            ref = [tgt_itos[idx.item()] for idx in tgt_batch[:, i]
                   if tgt_itos[idx.item()] not in special]
            hypotheses.append(hyp)
            references.append(ref)

    return corpus_bleu(hypotheses, references), hypotheses, references


def show_examples(model, test_loader, src_vocab, tgt_vocab, n=5):
    src_itos = src_vocab.get_itos()
    tgt_itos = tgt_vocab.get_itos()
    special  = {"<pad>", "<bos>", "<eos>", "<unk>"}
    shown = 0

    for src_batch, tgt_batch in test_loader:
        for i in range(src_batch.size(1)):
            if shown >= n:
                return
            src_single = src_batch[:, i].unsqueeze(1)
            src_text = " ".join(src_itos[idx.item()] for idx in src_batch[:, i]
                                if src_itos[idx.item()] not in special)
            ref_text = " ".join(tgt_itos[idx.item()] for idx in tgt_batch[:, i]
                                if tgt_itos[idx.item()] not in special)
            hyp_text = " ".join(greedy_decode(model, src_single, tgt_vocab))
            print(f"\n  Source    : {src_text}")
            print(f"  Reference : {ref_text}")
            print(f"  Hypothesis: {hyp_text}")
            shown += 1


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["all", "lstm_seq2seq", "gru_seq2seq"])
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    print("Loading data...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = load_data(BATCH_SIZE)
    src_vs = len(src_vocab)
    tgt_vs = len(tgt_vocab)

    model_map = {
        "lstm_seq2seq": build_lstm_seq2seq(src_vs, tgt_vs, PAD_IDX, device,
                                           EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT),
        "gru_seq2seq":  build_gru_seq2seq(src_vs, tgt_vs, PAD_IDX, device,
                                          EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT),
    }
    variants = list(model_map.keys()) if args.model == "all" else [args.model]

    for name in variants:
        ckpt = os.path.join(SAVE_DIR, f"{name}.pt")
        if not os.path.exists(ckpt):
            print(f"  [{name}] Checkpoint not found at {ckpt} -- skipping.")
            continue

        model = model_map[name]
        model.load_state_dict(torch.load(ckpt, map_location=device))

        print(f"\n{'='*60}")
        print(f"  Model: {name}")
        print(f"{'='*60}")

        bleu, hyps, refs = compute_bleu(model, test_loader, src_vocab, tgt_vocab)
        print(f"\n  Corpus BLEU: {bleu:.2f}")
        print("\n  Interpretation:")
        print("    < 10  : almost unusable")
        print("    10-19 : gist is understandable")
        print("    20-29 : clear but with errors  <- typical for small datasets")
        print("    30-40 : good, approaching human quality")
        print("    > 40  : very high quality")

        print(f"\n  -- Qualitative Examples ({args.examples} sentences) --")
        show_examples(model, test_loader, src_vocab, tgt_vocab, n=args.examples)


if __name__ == "__main__":
    main()
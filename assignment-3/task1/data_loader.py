"""
data_loader.py - Task 1: Text Generation
Reads WikiText-2 from HuggingFace parquet files.
"""

import os
from collections import Counter
import torch
import pandas as pd
from torchtext.data.utils import get_tokenizer

BATCH_SIZE = 32
SEQ_LEN    = 35
MIN_FREQ   = 3

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wikitext-2")

FILES = {
    "train": "train-00000-of-00001.parquet",
    "valid": "validation-00000-of-00001.parquet",
    "test":  "test-00000-of-00001.parquet",
}

TEXT_COL = "text"
tokenizer = get_tokenizer("basic_english")


# Read 

def _read(split: str):
    fpath = os.path.join(DATA_DIR, FILES[split])
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Missing: {fpath}\nCopy parquet files into: {DATA_DIR}")
    df = pd.read_parquet(fpath)
    return df[TEXT_COL].dropna().tolist()


# Vocab (manual, compatible with torchtext 0.6)

class Vocab:
    """Minimal vocab class that mimics torchtext Vocab interface."""

    def __init__(self, counter, min_freq=1, specials=("<unk>", "<eos>")):
        self.stoi = {}
        self.itos = []

        # Add specials first
        for s in specials:
            self.stoi[s] = len(self.itos)
            self.itos.append(s)

        # Add tokens meeting min_freq threshold
        for token, count in counter.most_common():
            if count >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

        self.default_index = self.stoi["<unk>"]

    def __len__(self):
        return len(self.itos)

    def __call__(self, tokens):
        return [self.stoi.get(t, self.default_index) for t in tokens]

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)

    def get_itos(self):
        return self.itos

    def get_stoi(self):
        return self.stoi


def build_vocab(train_lines):
    counter = Counter()
    for line in train_lines:
        if isinstance(line, str) and line.strip():
            counter.update(tokenizer(line))
    vocab = Vocab(counter, min_freq=MIN_FREQ, specials=("<unk>", "<eos>"))
    return vocab


# Processing 

def data_process(lines, vocab):
    tokens = []
    eos = vocab["<eos>"]
    for line in lines:
        if not isinstance(line, str) or not line.strip():
            continue
        ids = vocab(tokenizer(line))
        if ids:
            tokens += ids + [eos]
    return torch.tensor(tokens, dtype=torch.long)


def batchify(data: torch.Tensor, batch_size: int):
    n = data.size(0) // batch_size
    data = data[: n * batch_size]
    return data.view(batch_size, -1).t().contiguous()


def get_batch(source: torch.Tensor, i: int, seq_len: int = SEQ_LEN):
    length = min(seq_len, source.size(0) - 1 - i)
    x = source[i : i + length]
    y = source[i + 1 : i + 1 + length]
    return x, y


#  Public API 

def load_data(batch_size: int = BATCH_SIZE):
    print("Loading WikiText-2 from parquet files...")

    train_lines = _read("train")
    val_lines   = _read("valid")
    test_lines  = _read("test")

    print(f"  Train rows: {len(train_lines):,}")
    print(f"  Val rows  : {len(val_lines):,}")
    print(f"  Test rows : {len(test_lines):,}")

    vocab = build_vocab(train_lines)

    train_data = batchify(data_process(train_lines, vocab), batch_size)
    val_data   = batchify(data_process(val_lines,   vocab), batch_size)
    test_data  = batchify(data_process(test_lines,  vocab), batch_size)

    print(f"Vocab size   : {len(vocab):,}")
    print(f"Train tokens : {train_data.numel():,}")
    print(f"Val tokens   : {val_data.numel():,}")
    print(f"Test tokens  : {test_data.numel():,}")

    return train_data, val_data, test_data, vocab


if __name__ == "__main__":
    train, val, test, vocab = load_data()
    x, y = get_batch(train, 0)
    print(f"\nSample -- input: {x.shape}, target: {y.shape}")

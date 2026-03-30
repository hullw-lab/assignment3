"""
data_loader.py - Task 2: Machine Translation (English -> German)
Reads Multi30K from parquet files saved via HuggingFace datasets.
Columns: 'en' (source), 'de' (target)
Compatible with torchtext 0.6.0 (no torchtext dataset API used).
"""

import os
from collections import Counter
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer

# Constants
BATCH_SIZE = 128
MIN_FREQ   = 2

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "multi30k")

FILES = {
    "train": "train.parquet",
    "valid": "validation.parquet",
    "test":  "test.parquet",
}

# Special token indices
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
SPECIAL_SYMBOLS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# Tokenizers 
try:
    import spacy
    spacy_en = spacy.load("en_core_web_sm")
    spacy_de = spacy.load("de_core_news_sm")
    en_tokenizer = lambda text: [tok.text.lower() for tok in spacy_en.tokenizer(text)]
    de_tokenizer = lambda text: [tok.text.lower() for tok in spacy_de.tokenizer(text)]
    print("Using spaCy tokenizers.")
except OSError:
    print("spaCy models not found -- falling back to basic_english tokenizer.")
    _basic = get_tokenizer("basic_english")
    en_tokenizer = lambda text: _basic(text)
    de_tokenizer = lambda text: _basic(text)


# Vocab (manual, torchtext 0.6 compatible)

class Vocab:
    def __init__(self, counter, min_freq=1, specials=None):
        self.stoi = {}
        self.itos = []

        for s in (specials or []):
            self.stoi[s] = len(self.itos)
            self.itos.append(s)

        for token, count in counter.most_common():
            if count >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

        self.default_index = self.stoi.get("<unk>", 0)

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


# Read & Build Vocab 

def _read(split: str):
    fpath = os.path.join(DATA_DIR, FILES[split])
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Missing: {fpath}\n"
            f"Run download_data.py first to fetch Multi30K."
        )
    df = pd.read_parquet(fpath)
    return df["en"].tolist(), df["de"].tolist()


def build_vocabs(train_en, train_de):
    src_counter = Counter()
    tgt_counter = Counter()
    for en, de in zip(train_en, train_de):
        src_counter.update(en_tokenizer(en))
        tgt_counter.update(de_tokenizer(de))

    src_vocab = Vocab(src_counter, min_freq=MIN_FREQ, specials=SPECIAL_SYMBOLS)
    tgt_vocab = Vocab(tgt_counter, min_freq=MIN_FREQ, specials=SPECIAL_SYMBOLS)

    print(f"Source (EN) vocab size: {len(src_vocab):,}")
    print(f"Target (DE) vocab size: {len(tgt_vocab):,}")
    return src_vocab, tgt_vocab


# Collate 

def make_collate_fn(src_vocab, tgt_vocab):
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_text, tgt_text in batch:
            src_ids = [BOS_IDX] + src_vocab(en_tokenizer(src_text)) + [EOS_IDX]
            tgt_ids = [BOS_IDX] + tgt_vocab(de_tokenizer(tgt_text)) + [EOS_IDX]
            src_batch.append(torch.tensor(src_ids, dtype=torch.long))
            tgt_batch.append(torch.tensor(tgt_ids, dtype=torch.long))

        # Pad to longest in batch -> (max_len, batch)
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch
    return collate_fn


#  Public API 

def load_data(batch_size: int = BATCH_SIZE):
    """Returns (train_loader, val_loader, test_loader, src_vocab, tgt_vocab)."""
    print("Loading Multi30K from parquet files...")

    train_en, train_de = _read("train")
    val_en,   val_de   = _read("valid")
    test_en,  test_de  = _read("test")

    print(f"  Train pairs: {len(train_en):,}")
    print(f"  Val pairs  : {len(val_en):,}")
    print(f"  Test pairs : {len(test_en):,}")

    src_vocab, tgt_vocab = build_vocabs(train_en, train_de)
    collate_fn = make_collate_fn(src_vocab, tgt_vocab)

    def make_loader(en_list, de_list, shuffle):
        pairs = list(zip(en_list, de_list))
        return DataLoader(pairs, batch_size=batch_size,
                          shuffle=shuffle, collate_fn=collate_fn)

    train_loader = make_loader(train_en, train_de, shuffle=True)
    val_loader   = make_loader(val_en,   val_de,   shuffle=False)
    test_loader  = make_loader(test_en,  test_de,  shuffle=False)

    print(f"  Train batches: {len(train_loader)} | "
          f"Val: {len(val_loader)} | Test: {len(test_loader)}")

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


if __name__ == "__main__":
    train, val, test, sv, tv = load_data(batch_size=4)
    src_batch, tgt_batch = next(iter(train))
    print(f"\nSample batch -- src: {src_batch.shape}, tgt: {tgt_batch.shape}")

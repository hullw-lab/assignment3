"""
download_data.py - Task 2: Machine Translation
Downloads Multi30K (EN/DE) from HuggingFace and saves as parquet files.

Usage:
    python download_data.py
"""

import os
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "multi30k")
os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading bentrevett/multi30k from HuggingFace...")
ds = load_dataset("bentrevett/multi30k")

print(ds)
print("\nSample:", ds["train"][0])

ds["train"].to_parquet(os.path.join(DATA_DIR, "train.parquet"))
ds["validation"].to_parquet(os.path.join(DATA_DIR, "validation.parquet"))
ds["test"].to_parquet(os.path.join(DATA_DIR, "test.parquet"))

print(f"\nSaved to {DATA_DIR}/")
print("  train.parquet")
print("  validation.parquet")
print("  test.parquet")
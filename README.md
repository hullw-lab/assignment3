# Assignment 3: Natural Language Processing

## Project Overview

For this assignment I implemented deep learning models for two different NLP tasks using PyTorch.

- **Task 1 - Text Generation**: I trained LSTM and GRU language models on the WikiText-2 dataset to predict the next word in a sequence. I tested four different combinations of model architecture and word embeddings to see how they compare.
- **Task 2 - Machine Translation**: I built a seq2seq model that translates English sentences to German using the Multi30K dataset. Both LSTM and GRU versions were implemented with an attention mechanism.

## Dataset Description

### Task 1 - WikiText-2
- A dataset of Wikipedia articles commonly used for language modeling tasks
- After preprocessing there were about 2 million training tokens
- Vocabulary size ended up being 29,473 tokens (filtered out anything appearing less than 3 times)
- Split into 36,718 train / 3,760 validation / 4,358 test rows
- Downloaded from HuggingFace as parquet files

### Task 2 - Multi30K (English to German)
- A dataset of image captions in English paired with German translations
- 29,000 training pairs, 1,014 validation, 1,000 test
- English vocab: 5,921 tokens | German vocab: 7,820 tokens
- Also downloaded from HuggingFace (`bentrevett/multi30k`)

## Model Architectures Used

### Task 1 - Language Models

Both models take in a sequence of words and try to predict the next word at each step. Training uses backpropagation through time (BPTT) with a window of 35 tokens.

**LSTM Model**

Embedding -> Dropout(0.5) -> LSTM(2 layers, 512 hidden units) -> Dropout(0.5) -> Linear -> LogSoftmax

**GRU Model**

Embedding -> Dropout(0.5) -> GRU(2 layers, 512 hidden units) -> Dropout(0.5) -> Linear -> LogSoftmax

### Task 2 - Seq2Seq with Attention

Both translation models use an encoder-decoder design with Bahdanau attention. The attention mechanism lets the decoder look back at different parts of the input sentence when generating each output word, which helps a lot compared to just passing a single context vector.

Encoder: Embedding -> Dropout -> RNN -> encoder outputs + hidden state
Attention: score = v * tanh(W1 * decoder_hidden + W2 * encoder_output)
Decoder: Embedding + attention context -> Dropout -> RNN -> Linear -> word probabilities

- **LSTM Seq2Seq**: uses LSTM cells in both encoder and decoder (has both hidden state and cell state)
- **GRU Seq2Seq**: uses GRU cells (just one hidden state, simpler than LSTM)
- Teacher forcing starts at 0.5 and slowly decreases to 0.3 over training so the model learns to use its own predictions

## Word Embedding Methods

| Method | Dimension | Trainable | Where Used |
|---|---|---|---|
| Learned embeddings | 256 | Yes | Task 1 LSTM and GRU (main baseline) |
| GloVe-100 (pretrained, frozen) | 100 | No | Task 1 LSTM |
| One-hot encoding | vocab_size | No | Task 1 GRU (simple baseline) |
| Learned embeddings | 256 | Yes | Task 2 LSTM and GRU |

**GloVe**: embeddings come from Stanford's pretrained vectors trained on 6 billion tokens. The idea is that words with similar meanings are already close together in the embedding space before training even starts. Coverage on our vocabulary was 96.4% - words not found in GloVe were just set to zero vectors.

**One-hot encoding**: just represents each word as a giant vector of zeros with a single 1 at that word's index. There's no information about word similarity at all, so it acts as a lower bound baseline to compare against.

## Experimental Results

### Task 1 - Test Perplexity (lower is better)

| Model | Embedding | Test Perplexity |
|---|---|---|
| LSTM | Learned (256d) | 120.93 |
| LSTM | GloVe-100 (frozen) | 231.36 |
| GRU | Learned (256d) | 130.05 |
| GRU | One-hot (baseline) | 212.51 |

### Task 2 - BLEU Score on Test Set (higher is better)

| Model | BLEU Score |
|---|---|
| LSTM Seq2Seq | 25.78 |
| GRU Seq2Seq | 25.88 |

### Sample Generated Text (Task 1, LSTM + Learned embeddings)

Prompt: *"the president of the united states"*

the president of the united states that it was serving to be found to be the case of the two @-@ piece . the effect was made for 5 @ . @ 9 % of the game , and a year in which the game housed a series of the household ' s

### Sample Translations (Task 2)

| Source (EN) | Reference (DE) | LSTM Output | GRU Output |
|---|---|---|---|
| a man in an orange hat starring at something . | ein mann mit einem orangefarbenen hut , der etwas . | ein mann mit orangefarbenem hut betrachtet etwas etwas . | ein mann mit orangefarbener mütze meißelt etwas etwas . |
| people are fixing the roof of a house . | leute reparieren das dach eines hauses . | menschen personen die das eines eines hauses . | leute starren den dach eines hauses . |
| five people wearing winter jackets and helmets stand in the snow , with in the background . | fünf leute in winterjacken und mit helmen stehen im schnee mit im hintergrund . | fünf personen in rettungswesten und helmen stehen im schnee und im hintergrund im im hintergrund . | fünf personen mit jacken und helmen stehen im schnee im hintergrund . 

## Comparison of Models

### LSTM vs GRU - Task 1

LSTM did slightly better than GRU on text generation (120.93 vs 130.05 perplexity). This makes sense because LSTMs have both a hidden state and a cell state, giving them more memory to work with across the 35-token training window. GRUs only have one hidden state so they're a bit less expressive, though they're also faster to train.

### LSTM vs GRU - Task 2

For translation the two models basically tied (25.78 vs 25.88 BLEU). Multi30K sentences are pretty short on average so the extra capacity of LSTM doesn't really help - GRU handles short sequences just as well.

### Embedding Comparison

The biggest surprise was that learned embeddings beat GloVe by a lot (120.93 vs 231.36 perplexity). I expected GloVe to do better since it was pretrained on a huge corpus, but the issue is that the GloVe weights were frozen during training. The model couldn't adjust them to fit the WikiText-2 domain, so the LSTM was stuck working with embeddings that weren't optimized for this task. If GloVe had been fine-tuned during training it probably would have done better.

One-hot performed similarly badly to frozen GloVe (212.51 perplexity), which makes sense since both are fixed representations that the model can't learn from.

## Challenges Faced During Implementation

- **torchtext version issues**: The version installed (0.6.0) was really old and didn't support the modern API at all. WikiText2(split="train") didn't work, and build_vocab_from_iterator didn't accept min_freq as an argument. I had to completely rewrite the data loading code and build a custom Vocab class from scratch using collections.Counter.
- **Dataset downloading**: The WikiText-2 raw files kept returning 404 errors from different URLs. Eventually I just downloaded the data from HuggingFace as parquet files and read them with pandas, which bypassed the torchtext dataset API entirely.
- **GPU not being used**: PyTorch was defaulting to CPU even though my machine has a GPU. Turned out the installed version was the CPU-only build. Had to uninstall and reinstall with --index-url https://download.pytorch.org/whl/cu130 to get CUDA working.
- **One-hot memory**: A 29,473 x 29,473 identity matrix takes up a lot of memory. I had to reduce the hidden size to 128 and use only one layer for the one-hot GRU model to avoid running out of VRAM.
- **API deprecations**: ReduceLROnPlateau no longer accepts verbose=True in newer PyTorch, and global declarations in Python 3.13 have to come before any use of the variable in the same function. Both caused errors that had to be fixed.
- **spaCy models**: The German spaCy model (`de_core_news_sm`) couldn't be installed with the normal one-liner command on Windows. Had to install it directly from the GitHub release URL using pip.

##  Limitations of the Considered Models

- **These aren't Transformers**: LSTM and GRU seq2seq models were state of the art around 2017 but have been completely replaced by Transformer-based models like BERT and GPT. The results here are much worse than what you'd get with a modern approach.
- **Frozen GloVe hurt performance**: Keeping GloVe weights frozen prevented the model from adapting to the dataset. This was a design choice made to compare pure embedding types, but in practice you'd want to fine-tune them.
- **Word-level tokenization causes OOV problems**: German has a lot of compound words that don't appear in the training vocabulary, which shows up as `<unk>` tokens in the translations. Subword tokenization (like BPE) would fix this.
- **Greedy decoding**: The translation evaluation just picks the highest probability word at each step. Beam search would give better results by exploring multiple possible translations at once.
- **Multi30K is tiny**: 29,000 sentence pairs is a very small dataset for machine translation. Real MT systems train on hundreds of millions of pairs.
- **Unidirectional encoder**: The encoder only reads the source sentence left to right. A bidirectional encoder would give the decoder more context to work with.

---

## Possible Future Improvements

- **Use a Transformer**: This is the obvious next step. Even a small Transformer would likely outperform these RNN models significantly.
- **Fine-tune GloVe**: Unfreeze the GloVe weights after a few warm-up epochs to let them adapt to the domain.
- **BPE tokenization**: Use byte pair encoding so the model can handle rare and compound German words without falling back to '<unk>'.
- **Bidirectional encoder**: A bidirectional LSTM/GRU encoder would give better representations of the source sentence.
- **More data**: Training on a larger dataset like WMT14 would dramatically improve translation quality.

## Setup

### Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install torchtext pandas pyarrow datasets spacy
python -m spacy download en_core_web_sm
pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl
```

### Task 1 - Text Generation

```bash
cd task1

# Train all four model variants
python train.py

# Evaluate + generate text
python evaluate.py --all
python evaluate.py --model lstm_learned --prompt "the history of science"
```

### Task 2 - Machine Translation

```bash
cd task2

# Download the dataset first
python download_data.py

# Train
python train.py

# Evaluate BLEU 
python evaluate.py
```

---

## File Structure

```
assignment3-nlp/
├── README.md
├── requirements.txt
├── task1/
│   ├── data_loader.py   # loads WikiText-2 parquet files, builds vocab, BPTT batching
│   ├── models.py        # LSTM and GRU models, GloVe and one-hot embedding setup
│   ├── train.py         # training loop for all 4 variants
│   └── evaluate.py      # perplexity + text generation examples
└── task2/
    ├── download_data.py # downloads Multi30K from HuggingFace
    ├── data_loader.py   # loads parquet files, builds vocab, pads batches
    ├── models.py        # LSTM and GRU seq2seq with Bahdanau attention
    ├── train.py         # training with teacher forcing
    └── evaluate.py      # BLEU score + translation examples
```

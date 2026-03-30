"""
models.py — Task 1: Text Generation
Defines four model variants:
  - LSTMModel  (with learned embeddings OR pre-trained GloVe)
  - GRUModel   (with learned embeddings OR pre-trained GloVe)
One-hot encoding is handled as a special embedding (vocab_size → vocab_size matrix).
"""

import torch
import torch.nn as nn
from torchtext.vocab import GloVe


# Embedding Helpers 

def build_glove_embedding(vocab, glove_dim: int = 100, freeze: bool = True):
    """
    Load GloVe vectors and align them to our vocabulary.
    Words not found in GloVe are initialised to zero.

    Args:
        vocab     : torchtext Vocab object
        glove_dim : 50 | 100 | 200 | 300
        freeze    : if True, GloVe weights are not updated during training

    Returns:
        nn.Embedding with pre-filled weights
    """
    glove = GloVe(name="6B", dim=glove_dim)
    vocab_size = len(vocab)
    weight = torch.zeros(vocab_size, glove_dim)

    hits = 0
    for token, idx in vocab.get_stoi().items():
        if token in glove.stoi:
            weight[idx] = glove[token]
            hits += 1

    coverage = 100 * hits / vocab_size
    print(f"GloVe coverage: {hits}/{vocab_size} tokens ({coverage:.1f}%)")

    embedding = nn.Embedding(vocab_size, glove_dim)
    embedding.weight = nn.Parameter(weight, requires_grad=not freeze)
    return embedding


def build_onehot_embedding(vocab_size: int):
    """
    One-hot 'embedding': identity matrix — each token maps to a
    vocab_size-dimensional sparse vector. Not trainable.
    Because this blows up memory for large vocabs, we keep it frozen.
    """
    weight = torch.eye(vocab_size)
    embedding = nn.Embedding(vocab_size, vocab_size)
    embedding.weight = nn.Parameter(weight, requires_grad=False)
    return embedding


# LSTM Model 

class LSTMModel(nn.Module):
    """
    Word-level language model using an LSTM.

    Architecture:
        Embedding → Dropout -> LSTM (stacked) -> Dropout -> Linear -> LogSoftmax

    Args:
        vocab_size    : number of tokens in vocabulary
        embed_dim     : embedding dimension (ignored when embedding is provided)
        hidden_dim    : LSTM hidden state size
        num_layers    : number of stacked LSTM layers
        dropout       : dropout probability (applied between layers & on output)
        embedding     : optional pre-built nn.Embedding (GloVe / one-hot)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.5,
        embedding: nn.Embedding = None,
    ):
        super().__init__()

        if embedding is not None:
            self.embedding = embedding
            embed_dim = embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.dropout   = nn.Dropout(dropout)
        self.lstm      = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,   # expects (seq_len, batch, features)
        )
        self.fc        = nn.Linear(hidden_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)

    def forward(self, x, hidden=None):
        """
        x      : (seq_len, batch)
        hidden : optional (h_n, c_n) carried across BPTT chunks
        Returns:
            log_probs : (seq_len * batch, vocab_size)
            hidden    : new (h_n, c_n) for next chunk
        """
        emb = self.dropout(self.embedding(x))          # (seq, batch, embed)
        out, hidden = self.lstm(emb, hidden)            # (seq, batch, hidden)
        out = self.dropout(out)
        logits = self.fc(out.reshape(-1, out.size(2))) # (seq*batch, vocab)
        return self.log_softmax(logits), hidden

    def init_hidden(self, batch_size: int, device):
        """Zero initial hidden state."""
        h = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        c = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        return (h, c)

    def detach_hidden(self, hidden):
        """Detach hidden from graph to prevent backprop through full history."""
        return tuple(h.detach() for h in hidden)


# GRU Model 

class GRUModel(nn.Module):
    """
    Word-level language model using a GRU.
    Identical architecture to LSTMModel but uses GRU (single hidden state).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.5,
        embedding: nn.Embedding = None,
    ):
        super().__init__()

        if embedding is not None:
            self.embedding = embedding
            embed_dim = embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.gru     = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )
        self.fc      = nn.Linear(hidden_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.gru(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out.reshape(-1, out.size(2)))
        return self.log_softmax(logits), hidden

    def init_hidden(self, batch_size: int, device):
        """GRU only has h, no cell state."""
        return torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size).to(device)

    def detach_hidden(self, hidden):
        return hidden.detach()

"""
models.py — Task 2: Machine Translation
Encoder-Decoder seq2seq with attention, implemented for both LSTM and GRU.

Architecture overview:
    Encoder : embeds + encodes source sentence → context vectors
    Attention: computes alignment weights over encoder outputs
    Decoder : generates target tokens one at a time using attention context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# ────────────────────────────────────────────────────────────────────────────
# Shared Attention Module (Bahdanau / additive attention)
# ────────────────────────────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    """
    Computes attention weights given decoder hidden state and encoder outputs.

    attn_score(h_dec, h_enc) = v · tanh(W1·h_dec + W2·h_enc)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v  = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden  : (batch, hidden_dim)
        encoder_outputs : (src_len, batch, hidden_dim)

        Returns:
            context     : (batch, hidden_dim) — weighted sum of encoder outputs
            attn_weights: (batch, src_len)    — for visualisation / analysis
        """
        src_len = encoder_outputs.size(0)
        # Repeat decoder hidden across src_len for element-wise addition
        dec = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch, src_len, h)
        enc = encoder_outputs.permute(1, 0, 2)                   # (batch, src_len, h)

        energy = self.v(torch.tanh(self.W1(dec) + self.W2(enc))).squeeze(2)  # (batch, src_len)
        attn_weights = F.softmax(energy, dim=1)                              # (batch, src_len)

        context = torch.bmm(attn_weights.unsqueeze(1), enc).squeeze(1)       # (batch, h)
        return context, attn_weights


# ────────────────────────────────────────────────────────────────────────────
# LSTM Seq2Seq
# ────────────────────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """src: (src_len, batch)"""
        emb = self.dropout(self.embedding(src))          # (src_len, batch, embed)
        outputs, (hidden, cell) = self.lstm(emb)         # outputs: (src_len, batch, h)
        return outputs, hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention  = BahdanauAttention(hidden_dim)
        # Input to LSTM = embed + context vector
        self.lstm       = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers,
                                  dropout=dropout if num_layers > 1 else 0.0,
                                  batch_first=False)
        self.fc_out     = nn.Linear(hidden_dim * 2, vocab_size)  # hidden + context → vocab
        self.dropout    = nn.Dropout(dropout)

    def forward_step(self, tgt_token, hidden, cell, encoder_outputs):
        """
        Single decoding step.
        tgt_token      : (batch,) — current input token
        Returns logits (batch, vocab_size), new hidden, new cell
        """
        emb = self.dropout(self.embedding(tgt_token.unsqueeze(0)))  # (1, batch, embed)

        # Use top layer of hidden for attention
        context, _ = self.attention(hidden[-1], encoder_outputs)    # (batch, h)
        context_expanded = context.unsqueeze(0)                     # (1, batch, h)

        lstm_input = torch.cat([emb, context_expanded], dim=2)      # (1, batch, embed+h)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        output  = output.squeeze(0)                                  # (batch, h)
        logits  = self.fc_out(torch.cat([output, context], dim=1))  # (batch, vocab)
        return logits, hidden, cell


class LSTMSeq2Seq(nn.Module):
    """LSTM encoder–decoder with Bahdanau attention."""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    def forward(self, src, tgt, teacher_forcing_ratio: float = 0.5):
        """
        src : (src_len, batch)
        tgt : (tgt_len, batch)
        Returns:
            outputs : (tgt_len, batch, tgt_vocab_size)
        """
        tgt_len, batch_size = tgt.shape
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)
        enc_out, hidden, cell = self.encoder(src)

        # First decoder input is <bos>
        dec_input = tgt[0]

        for t in range(1, tgt_len):
            logits, hidden, cell = self.decoder.forward_step(
                dec_input, hidden, cell, enc_out
            )
            outputs[t] = logits

            # Teacher forcing: feed ground truth or model prediction
            use_teacher = random.random() < teacher_forcing_ratio
            dec_input = tgt[t] if use_teacher else logits.argmax(dim=1)

        return outputs


# ────────────────────────────────────────────────────────────────────────────
# GRU Seq2Seq
# ────────────────────────────────────────────────────────────────────────────

class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers,
                          dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        emb = self.dropout(self.embedding(src))
        outputs, hidden = self.gru(emb)
        return outputs, hidden


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hidden_dim)
        self.gru       = nn.GRU(embed_dim + hidden_dim, hidden_dim, num_layers,
                                dropout=dropout if num_layers > 1 else 0.0,
                                batch_first=False)
        self.fc_out    = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout   = nn.Dropout(dropout)

    def forward_step(self, tgt_token, hidden, encoder_outputs):
        emb = self.dropout(self.embedding(tgt_token.unsqueeze(0)))
        context, _ = self.attention(hidden[-1], encoder_outputs)
        context_expanded = context.unsqueeze(0)
        gru_input = torch.cat([emb, context_expanded], dim=2)
        output, hidden = self.gru(gru_input, hidden)
        output = output.squeeze(0)
        logits = self.fc_out(torch.cat([output, context], dim=1))
        return logits, hidden


class GRUSeq2Seq(nn.Module):
    """GRU encoder–decoder with Bahdanau attention."""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    def forward(self, src, tgt, teacher_forcing_ratio: float = 0.5):
        tgt_len, batch_size = tgt.shape
        tgt_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)

        enc_out, hidden = self.encoder(src)
        dec_input = tgt[0]

        for t in range(1, tgt_len):
            logits, hidden = self.decoder.forward_step(dec_input, hidden, enc_out)
            outputs[t] = logits
            use_teacher = random.random() < teacher_forcing_ratio
            dec_input = tgt[t] if use_teacher else logits.argmax(dim=1)

        return outputs


# ────────────────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────────────────

def build_lstm_seq2seq(src_vocab_size, tgt_vocab_size, pad_idx, device,
                       embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.5):
    encoder = LSTMEncoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx)
    decoder = LSTMDecoder(tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx)
    model   = LSTMSeq2Seq(encoder, decoder, device).to(device)
    return model


def build_gru_seq2seq(src_vocab_size, tgt_vocab_size, pad_idx, device,
                      embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.5):
    encoder = GRUEncoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx)
    decoder = GRUDecoder(tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx)
    model   = GRUSeq2Seq(encoder, decoder, device).to(device)
    return model
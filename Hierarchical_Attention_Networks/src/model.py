"""
Hierarchical Attention Network (HAN) for Document Classification
Re-implementation of:
  Yang et al. (2016) - "Hierarchical Attention Networks for Document Classification"
  https://aclanthology.org/N16-1174/

Architecture:
  Word Encoder      -> Bidirectional GRU over word embeddings
  Word Attention    -> Soft attention with a learned context vector u_w
  Sentence Encoder  -> Bidirectional GRU over sentence vectors
  Sentence Attention-> Soft attention with a learned context vector u_s
  Classifier        -> Linear + Softmax over the final document vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  Word-level encoder + attention
# ---------------------------------------------------------------------------

class WordAttentionEncoder(nn.Module):
    """
    Encodes a single sentence.

    Input : (batch, T, embed_dim)   — a batch of sentences (padded word embeddings)
    Output: (batch, 2*hidden_dim)   — one sentence vector per example
            (batch, T)              — word-level attention weights (for visualisation)
    """

    def __init__(self, embed_dim: int, hidden_dim: int, context_dim: int):
        """
        Args:
            embed_dim:   dimensionality of the input word embeddings
            hidden_dim:  number of units in each direction of the BiGRU
                         (final annotation has 2*hidden_dim dims)
            context_dim: dimensionality of the word-level context vector u_w
        """
        super().__init__()

        # Bidirectional GRU — produces h_it = [h_it→, h_it←]
        self.bigru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # MLP: u_it = tanh(W_w * h_it + b_w)   (Eq. 5)
        self.projection = nn.Linear(2 * hidden_dim, context_dim)

        # Learned word-level context vector u_w                (Eq. 6)
        self.context_vector = nn.Parameter(torch.randn(context_dim))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """
        Args:
            x       : (batch, T, embed_dim)
            lengths : (batch,) actual number of words in each sentence (optional,
                      used to pack padded sequences for efficiency)
        Returns:
            s       : (batch, 2*hidden_dim) — sentence vector
            alpha   : (batch, T)            — word attention weights
        """
        # --- BiGRU encoding ---
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            h_packed, _ = self.bigru(x_packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h_packed, batch_first=True)
        else:
            h, _ = self.bigru(x)  # (batch, T, 2*hidden_dim)

        # --- Word attention (Eq. 5-7) ---
        # u_it = tanh(W_w h_it + b_w)
        u = torch.tanh(self.projection(h))                      # (batch, T, context_dim)

        # similarity with context vector
        score = torch.matmul(u, self.context_vector)            # (batch, T)

        # mask padding positions (set to -inf before softmax)
        if lengths is not None:
            mask = torch.arange(h.size(1), device=h.device).unsqueeze(0) >= lengths.unsqueeze(1)
            score = score.masked_fill(mask, float("-inf"))

        alpha = F.softmax(score, dim=1)                         # (batch, T)

        # sentence vector = weighted sum of annotations
        s = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)        # (batch, 2*hidden_dim)

        return s, alpha


# ---------------------------------------------------------------------------
# 2.  Sentence-level encoder + attention
# ---------------------------------------------------------------------------

class SentenceAttentionEncoder(nn.Module):
    """
    Encodes a full document given its sentence vectors.

    Input : (batch, L, 2*word_hidden_dim) — sentence vectors
    Output: (batch, 2*sent_hidden_dim)    — document vector
            (batch, L)                    — sentence-level attention weights
    """

    def __init__(self, sent_input_dim: int, hidden_dim: int, context_dim: int):
        """
        Args:
            sent_input_dim: input dim of each sentence vector (= 2*word_hidden_dim)
            hidden_dim:     BiGRU hidden size per direction
            context_dim:    dimensionality of the sentence-level context vector u_s
        """
        super().__init__()

        self.bigru = nn.GRU(
            input_size=sent_input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # u_i = tanh(W_s h_i + b_s)         (Eq. 8)
        self.projection = nn.Linear(2 * hidden_dim, context_dim)

        # Learned sentence-level context vector u_s            (Eq. 9)
        self.context_vector = nn.Parameter(torch.randn(context_dim))

    def forward(self, s: torch.Tensor, lengths: torch.Tensor | None = None):
        """
        Args:
            s       : (batch, L, sent_input_dim)
            lengths : (batch,) actual number of sentences (optional)
        Returns:
            v       : (batch, 2*hidden_dim)  — document vector
            alpha   : (batch, L)             — sentence attention weights
        """
        if lengths is not None:
            s_packed = nn.utils.rnn.pack_padded_sequence(
                s, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            h_packed, _ = self.bigru(s_packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h_packed, batch_first=True)
        else:
            h, _ = self.bigru(s)                                # (batch, L, 2*hidden_dim)

        # Sentence attention  (Eq. 8-10)
        u = torch.tanh(self.projection(h))                      # (batch, L, context_dim)
        score = torch.matmul(u, self.context_vector)            # (batch, L)

        if lengths is not None:
            mask = torch.arange(h.size(1), device=h.device).unsqueeze(0) >= lengths.unsqueeze(1)
            score = score.masked_fill(mask, float("-inf"))

        alpha = F.softmax(score, dim=1)                         # (batch, L)
        v = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)        # (batch, 2*hidden_dim)

        return v, alpha


# ---------------------------------------------------------------------------
# 3.  Full HAN model
# ---------------------------------------------------------------------------

class HAN(nn.Module):
    """
    Hierarchical Attention Network for document classification.

    The model expects documents to be pre-tokenised and encoded as integer
    indices.  A typical forward pass:

        logits, word_attn, sent_attn = model(doc_tensor, sent_lengths, doc_lengths)

    Parameters
    ----------
    vocab_size     : int   — vocabulary size (number of distinct word ids)
    embed_dim      : int   — word embedding dimensionality          (paper: 200)
    word_hidden    : int   — BiGRU hidden size at word level        (paper: 50)
    sent_hidden    : int   — BiGRU hidden size at sentence level    (paper: 50)
    word_context   : int   — word-level attention context dim       (paper: 100)
    sent_context   : int   — sentence-level attention context dim   (paper: 100)
    num_classes    : int   — number of output classes
    dropout        : float — dropout probability (applied before classification)
    padding_idx    : int   — embedding index used for padding (default 0)
    pretrained_emb : optional Tensor of shape (vocab_size, embed_dim)
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 200,
        word_hidden: int = 50,
        sent_hidden: int = 50,
        word_context: int = 100,
        sent_context: int = 100,
        dropout: float = 0.1,
        padding_idx: int = 0,
        pretrained_emb: torch.Tensor | None = None,
    ):
        super().__init__()

        # --- Word embedding ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)

        # --- Word encoder + attention ---
        self.word_encoder = WordAttentionEncoder(
            embed_dim=embed_dim,
            hidden_dim=word_hidden,
            context_dim=word_context,
        )

        # --- Sentence encoder + attention ---
        # input to sentence encoder = 2*word_hidden (bidirectional word GRU output)
        self.sentence_encoder = SentenceAttentionEncoder(
            sent_input_dim=2 * word_hidden,
            hidden_dim=sent_hidden,
            context_dim=sent_context,
        )

        self.dropout = nn.Dropout(dropout)

        # --- Classifier  (Eq. 11) ---
        # document vector dimension = 2*sent_hidden (bidirectional)
        self.classifier = nn.Linear(2 * sent_hidden, num_classes)

    # ------------------------------------------------------------------
    def forward(
        self,
        doc: torch.Tensor,
        sent_lengths: torch.Tensor | None = None,
        doc_lengths: torch.Tensor | None = None,
    ):
        """
        Args
        ----
        doc          : LongTensor of shape (batch, L, T)
                       batch of documents; each document is L sentences of T words
        sent_lengths : LongTensor (batch, L) — actual word counts per sentence
        doc_lengths  : LongTensor (batch,)   — actual sentence counts per document

        Returns
        -------
        logits       : (batch, num_classes)
        word_alphas  : (batch, L, T)   — word-level attention weights
        sent_alphas  : (batch, L)      — sentence-level attention weights
        """
        batch_size, L, T = doc.shape

        # ---- Word level ----
        # Flatten to (batch*L, T) so we can run all sentences through the
        # word encoder in a single batch.
        doc_flat = doc.view(batch_size * L, T)              # (B*L, T)

        x = self.embedding(doc_flat)                        # (B*L, T, E)

        flat_sent_lengths = None
        if sent_lengths is not None:
            flat_sent_lengths = sent_lengths.view(batch_size * L)

        sent_vecs, word_alphas = self.word_encoder(x, flat_sent_lengths)
        # sent_vecs  : (B*L, 2*word_hidden)
        # word_alphas: (B*L, T)

        sent_vecs   = sent_vecs.view(batch_size, L, -1)     # (B, L, 2*word_hidden)
        word_alphas = word_alphas.view(batch_size, L, T)    # (B, L, T)

        # ---- Sentence level ----
        doc_vec, sent_alphas = self.sentence_encoder(sent_vecs, doc_lengths)
        # doc_vec    : (B, 2*sent_hidden)
        # sent_alphas: (B, L)

        # ---- Classification ----
        out = self.dropout(doc_vec)
        logits = self.classifier(out)                       # (B, num_classes)

        return logits, word_alphas, sent_alphas

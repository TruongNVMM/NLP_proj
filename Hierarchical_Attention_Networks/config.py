"""
config.py
---------
Central configuration for the HAN training pipeline.

All hyper-parameters live here as a single dataclass so they can be
inspected, serialised, and passed between modules without hidden globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class TrainConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    train_path: str = "data/train.csv"   # path to training CSV
    val_path:   str = "data/val.csv"     # path to validation CSV
    label_col:  str = "label"            # column name for integer class label
    text_col:   str = "text"             # column name for raw document text

    # ── Vocabulary ────────────────────────────────────────────────────────────
    max_vocab:     int = 50_000   # keep the N most frequent tokens
    min_freq:      int = 5        # drop tokens appearing fewer times than this
    max_sent_len:  int = 100      # truncate sentences longer than this (words)
    max_doc_sents: int = 30       # truncate documents longer than this (sentences)

    # ── Model (paper defaults) ────────────────────────────────────────────────
    embed_dim:    int   = 200    # word embedding dimensionality
    word_hidden:  int   = 50     # BiGRU hidden size at word level (per direction)
    sent_hidden:  int   = 50     # BiGRU hidden size at sentence level (per direction)
    word_context: int   = 100    # word-level attention context vector size
    sent_context: int   = 100    # sentence-level attention context vector size
    dropout:      float = 0.1    # dropout before the final linear classifier
    num_classes:  int   = 5      # number of output classes

    # ── Optimiser (paper uses SGD + momentum 0.9) ─────────────────────────────
    optimiser:    str   = "sgd"   # "sgd" | "adam"
    lr:           float = 0.01
    momentum:     float = 0.9     # SGD only
    weight_decay: float = 1e-5
    grad_clip:    float = 5.0     # max-norm gradient clipping

    # ── Training schedule ─────────────────────────────────────────────────────
    epochs:      int = 20
    batch_size:  int = 64
    seed:        int = 42
    num_workers: int = 0          # DataLoader worker processes

    # ── I/O ───────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"   # directory for saved checkpoints
    resume:         str = ""              # path to a checkpoint to resume from
    log_every:      int = 50              # log progress every N training batches

    # ── Device (auto-detected) ────────────────────────────────────────────────
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ))

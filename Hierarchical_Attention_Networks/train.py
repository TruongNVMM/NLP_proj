from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── sibling modules ───────────────────────────────────────────────────────────
from config import TrainConfig
from src.dataset import HANDataset, make_collate
from src.trainer import run_epoch, save_checkpoint, load_checkpoint
from src.vocabulary import Vocabulary

# ── model (lives one level up, alongside han_training/) ───────────────────────
# Adjust the import path if you keep han.py in a different location.
try:
    from src.model import HAN
except ModuleNotFoundError:
    # Allow running as a standalone package; callers must ensure han.py is on
    # sys.path or installed as a package.
    raise ImportError(
        "Cannot import HAN from han.py. "
        "Make sure han.py is in the same directory as the han_training/ package "
        "or is otherwise importable."
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def train(cfg: TrainConfig) -> tuple[HAN, Vocabulary]:
    """
    Build data loaders, model, and optimiser; run the full training loop.

    Parameters
    ----------
    cfg : fully-populated :class:`TrainConfig`.

    Returns
    -------
    (model, vocab)  — trained model and vocabulary (useful for inference).
    """
    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    log.info("Device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = HANDataset(cfg.train_path, vocab=None, cfg=cfg, build_vocab=True)
    vocab    = train_ds.vocab
    val_ds   = HANDataset(cfg.val_path,   vocab=vocab, cfg=cfg, build_vocab=False)

    collate_fn = make_collate(vocab.pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = (cfg.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.batch_size * 2,   # no gradients → can fit more
        shuffle     = False,
        num_workers = cfg.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = (cfg.device == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HAN(
        vocab_size   = len(vocab),
        num_classes  = cfg.num_classes,
        embed_dim    = cfg.embed_dim,
        word_hidden  = cfg.word_hidden,
        sent_hidden  = cfg.sent_hidden,
        word_context = cfg.word_context,
        sent_context = cfg.sent_context,
        dropout      = cfg.dropout,
        padding_idx  = vocab.pad_id,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    if cfg.optimiser.lower() == "sgd":
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr           = cfg.lr,
            momentum     = cfg.momentum,
            weight_decay = cfg.weight_decay,
        )
    else:
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr           = cfg.lr,
            weight_decay = cfg.weight_decay,
        )

    # Halve LR after 3 epochs without val-loss improvement (mirrors paper's
    # grid-search strategy).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3,
    )

    criterion = nn.CrossEntropyLoss()

    # ── Optional resume ───────────────────────────────────────────────────────
    start_epoch  = 1
    best_val_acc = 0.0

    if cfg.resume:
        start_epoch, _, best_val_acc = load_checkpoint(cfg.resume, model, optimiser)
        start_epoch += 1

    # ── Training loop ─────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg.epochs + 1):
        current_lr = optimiser.param_groups[0]["lr"]
        log.info("── Epoch %d / %d  (lr=%.2e) ──", epoch, cfg.epochs, current_lr)

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimiser,
            cfg, device, epoch, phase="train",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None,
            cfg, device, epoch, phase="val",
        )

        scheduler.step(val_loss)

        # Always keep the most recent checkpoint.
        save_checkpoint(
            str(ckpt_dir / "latest.pt"),
            epoch, model, optimiser, val_loss, val_acc, cfg, vocab,
        )

        # Overwrite best checkpoint when val accuracy improves.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                str(ckpt_dir / "best.pt"),
                epoch, model, optimiser, val_loss, val_acc, cfg, vocab,
            )
            log.info("★ New best val_acc=%.2f%%", best_val_acc)

        log.info(
            "Summary  train_loss=%.4f  train_acc=%.2f%%  "
            "val_loss=%.4f  val_acc=%.2f%%  best=%.2f%%",
            train_loss, train_acc, val_loss, val_acc, best_val_acc,
        )

    log.info("Training complete. Best val accuracy: %.2f%%", best_val_acc)
    return model, vocab


def parse_args() -> TrainConfig:
    """Parse sys.argv and return a populated :class:`TrainConfig`."""
    cfg = TrainConfig()
    p   = argparse.ArgumentParser(
        description="Train a Hierarchical Attention Network (HAN) "
                    "for document classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--train",         default=cfg.train_path,    help="Training CSV")
    g.add_argument("--val",           default=cfg.val_path,      help="Validation CSV")
    g.add_argument("--label_col",     default=cfg.label_col,     help="Label column name")
    g.add_argument("--text_col",      default=cfg.text_col,      help="Text column name")

    # Vocabulary
    g = p.add_argument_group("Vocabulary")
    g.add_argument("--max_vocab",     type=int, default=cfg.max_vocab)
    g.add_argument("--min_freq",      type=int, default=cfg.min_freq)
    g.add_argument("--max_sent_len",  type=int, default=cfg.max_sent_len)
    g.add_argument("--max_doc_sents", type=int, default=cfg.max_doc_sents)

    # Model
    g = p.add_argument_group("Model")
    g.add_argument("--embed_dim",    type=int,   default=cfg.embed_dim)
    g.add_argument("--word_hidden",  type=int,   default=cfg.word_hidden)
    g.add_argument("--sent_hidden",  type=int,   default=cfg.sent_hidden)
    g.add_argument("--word_context", type=int,   default=cfg.word_context)
    g.add_argument("--sent_context", type=int,   default=cfg.sent_context)
    g.add_argument("--dropout",      type=float, default=cfg.dropout)
    g.add_argument("--num_classes",  type=int,   default=cfg.num_classes)

    # Optimiser
    g = p.add_argument_group("Optimiser")
    g.add_argument("--optimiser",    default=cfg.optimiser, choices=["sgd", "adam"])
    g.add_argument("--lr",           type=float, default=cfg.lr)
    g.add_argument("--momentum",     type=float, default=cfg.momentum)
    g.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    g.add_argument("--grad_clip",    type=float, default=cfg.grad_clip)

    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--epochs",       type=int, default=cfg.epochs)
    g.add_argument("--batch_size",   type=int, default=cfg.batch_size)
    g.add_argument("--seed",         type=int, default=cfg.seed)
    g.add_argument("--num_workers",  type=int, default=cfg.num_workers)

    # I/O
    g = p.add_argument_group("I/O")
    g.add_argument("--checkpoint_dir", default=cfg.checkpoint_dir)
    g.add_argument("--resume",         default=cfg.resume,
                   help="Path to checkpoint to resume from")
    g.add_argument("--log_every",      type=int, default=cfg.log_every)

    # Device
    p.add_argument("--device", default=cfg.device)

    args = p.parse_args()

    # Write parsed values back onto the config dataclass.
    for key, val in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    # argparse uses "train" / "val" as flag names; dataclass uses *_path.
    cfg.train_path = args.train
    cfg.val_path   = args.val

    return cfg


if __name__ == "__main__":
    trained_cfg = parse_args()
    train(trained_cfg)

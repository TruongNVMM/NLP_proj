from __future__ import annotations

import logging
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainConfig
from .metrics import RunningMetrics
from .vocabulary import Vocabulary

log = logging.getLogger(__name__)

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimiser: torch.optim.Optimizer | None,
    cfg:       TrainConfig,
    device:    torch.device,
    epoch:     int,
    phase:     str = "train",
) -> tuple[float, float]:
    """
    Run one full pass over *loader*.

    Pass ``optimiser=None`` to run in evaluation mode (no gradient updates,
    model set to ``eval()``).

    Parameters
    ----------
    model     : HAN (or any nn.Module with the same output signature).
    loader    : DataLoader yielding (docs, sent_lengths, doc_lengths, labels).
    criterion : loss function.
    optimiser : if ``None`` the function runs in inference mode.
    cfg       : training config (used for ``grad_clip`` and ``log_every``).
    device    : target device.
    epoch     : current epoch number (for log messages only).
    phase     : "train" or "val" (for log messages only).

    Returns
    -------
    (avg_loss, accuracy)  — epoch-level averages.
    """
    training = optimiser is not None
    model.train(training)
    metrics  = RunningMetrics()
    t0       = time.time()

    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for step, (docs, sent_lengths, doc_lengths, labels) in enumerate(loader, 1):
            docs         = docs.to(device)
            sent_lengths = sent_lengths.to(device)
            doc_lengths  = doc_lengths.to(device)
            labels       = labels.to(device)

            logits, _, _ = model(docs, sent_lengths, doc_lengths)
            loss         = criterion(logits, labels)

            if training:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimiser.step()

            metrics.update(loss.item(), logits.detach(), labels)

            if training and step % cfg.log_every == 0:
                log.info(
                    "Epoch %d  [%s]  step %4d/%d  loss=%.4f  acc=%.2f%%  (%.1fs)",
                    epoch, phase, step, len(loader),
                    metrics.avg_loss, metrics.accuracy,
                    time.time() - t0,
                )

    log.info(
        "Epoch %d  [%s]  DONE  loss=%.4f  acc=%.2f%%  (%.1fs)",
        epoch, phase, metrics.avg_loss, metrics.accuracy,
        time.time() - t0,
    )
    return metrics.avg_loss, metrics.accuracy


def save_checkpoint(
    path:      str,
    epoch:     int,
    model:     nn.Module,
    optimiser: torch.optim.Optimizer,
    val_loss:  float,
    val_acc:   float,
    cfg:       TrainConfig,
    vocab:     Vocabulary,
):
    """
    Save a training checkpoint to *path*.

    The checkpoint contains everything needed to resume training or run
    inference: model weights, optimiser state, config, vocabulary, and the
    epoch/metric values at the time of saving.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "val_loss":  val_loss,
            "val_acc":   val_acc,
            "cfg":       cfg,
            "vocab":     vocab,
        },
        path,
    )
    log.info("Checkpoint saved → %s", path)


def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimiser: torch.optim.Optimizer,
) -> tuple[int, float, float]:
    """
    Load a checkpoint saved by :func:`save_checkpoint`.

    Restores weights into *model* and state into *optimiser* **in-place**.

    Returns
    -------
    (epoch, val_loss, val_acc)  — values recorded when the checkpoint was saved.
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimiser.load_state_dict(ckpt["optimiser"])
    log.info(
        "Resumed from '%s'  (epoch %d, val_acc=%.2f%%)",
        path, ckpt["epoch"], ckpt["val_acc"],
    )
    return ckpt["epoch"], ckpt["val_loss"], ckpt["val_acc"]

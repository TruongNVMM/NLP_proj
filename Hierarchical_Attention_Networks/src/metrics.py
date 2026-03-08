from __future__ import annotations

import torch


class RunningMetrics:
    """
    Accumulates loss and accuracy over a sequence of mini-batches.

    Typical usage inside a training loop::

        metrics = RunningMetrics()
        for docs, labels in loader:
            logits = model(docs)
            loss   = criterion(logits, labels)
            metrics.update(loss.item(), logits.detach(), labels)

        print(metrics.avg_loss, metrics.accuracy)
        metrics.reset()   # ready for the next epoch
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Zero all accumulators."""
        self._loss:    float = 0.0
        self._correct: int   = 0
        self._total:   int   = 0
        self._batches: int   = 0

    def update(
        self,
        loss:   float,
        logits: torch.Tensor,   # (batch, num_classes)  — detached
        labels: torch.Tensor,   # (batch,)
    ):
        """
        Record the results of one mini-batch.

        Parameters
        ----------
        loss   : scalar loss value for the batch (already `.item()`).
        logits : raw model output — argmax is taken to determine predictions.
        labels : ground-truth class indices.
        """
        self._loss    += loss
        self._batches += 1

        preds          = logits.argmax(dim=1)
        self._correct += int((preds == labels).sum().item())
        self._total   += labels.size(0)

    # ── read-only summaries ───────────────────────────────────────────────────

    @property
    def avg_loss(self) -> float:
        """Mean loss per batch since the last :meth:`reset`."""
        return self._loss / max(self._batches, 1)

    @property
    def accuracy(self) -> float:
        """Classification accuracy in percent (0–100) since the last :meth:`reset`."""
        return self._correct / max(self._total, 1) * 100

    def __repr__(self) -> str:
        return (
            f"RunningMetrics(loss={self.avg_loss:.4f}, "
            f"acc={self.accuracy:.2f}%, "
            f"batches={self._batches}, samples={self._total})"
        )

from __future__ import annotations
import csv
import logging

import torch
from torch.utils.data import Dataset

from config import TrainConfig
from .tokenizer import tokenise_doc
from .vocabulary import Vocabulary

log = logging.getLogger(__name__)


class HANDataset(Dataset):
    """
    Document classification dataset for HAN.

    Each item returned by ``__getitem__`` is a 4-tuple::

        (doc_ids, sent_lengths, doc_length, label)

    where *doc_ids* is a **list of lists** of integer word ids (not yet padded).
    Padding to a uniform rectangular shape is handled by :func:`han_collate`.

    Parameters
    ----------
    csv_path    : path to a CSV file with at minimum a label column and a text
                  column (column names are taken from *cfg*).
    vocab       : a pre-built :class:`Vocabulary`.  Pass ``None`` together with
                  ``build_vocab=True`` to build the vocabulary from this split.
    cfg         : :class:`TrainConfig` controlling truncation limits and column
                  names.
    build_vocab : if ``True`` a new vocabulary is built from this split and
                  stored in ``self.vocab``.
    """

    def __init__(
        self,
        csv_path:    str,
        vocab:       Vocabulary | None,
        cfg:         TrainConfig,
        build_vocab: bool = False,
    ):
        self.cfg   = cfg
        self.items: list[tuple[list[list[int]], list[int], int, int]] = []

        raw_docs: list[list[list[str]]] = []
        labels:   list[int]             = []

        # ── 1. Read and tokenise ──────────────────────────────────────────────
        self.label2id = {}
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                label_str = row[cfg.label_col]
                if label_str not in self.label2id:
                    self.label2id[label_str] = len(self.label2id)
                label = self.label2id[label_str]
                
                text  = row[cfg.text_col]
                sents = tokenise_doc(text)
                if not sents:
                    continue
                raw_docs.append(sents)
                labels.append(label)

        # ── 2. Build or validate vocabulary ──────────────────────────────────
        if build_vocab:
            vocab = Vocabulary.from_corpus(raw_docs, cfg.max_vocab, cfg.min_freq)
        if vocab is None:
            raise ValueError("Provide a Vocabulary or pass build_vocab=True.")
        self.vocab = vocab

        # ── 3. Encode and truncate ────────────────────────────────────────────
        for doc_tokens, label in zip(raw_docs, labels):
            doc_tokens = doc_tokens[: cfg.max_doc_sents]   # sentence truncation

            doc_ids:      list[list[int]] = []
            sent_lengths: list[int]       = []

            for sent in doc_tokens:
                ids = vocab.encode(sent[: cfg.max_sent_len])   # word truncation
                if not ids:
                    continue
                doc_ids.append(ids)
                sent_lengths.append(len(ids))

            if not doc_ids:
                continue   # skip empty documents

            self.items.append((doc_ids, sent_lengths, len(doc_ids), label))

        log.info("Dataset loaded: %d documents from '%s'", len(self.items), csv_path)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


# Collate
def han_collate(
    batch:  list[tuple[list[list[int]], list[int], int, int]],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a list of dataset items into padded batch tensors.

    Pads documents to the shape ``(B, L_max, T_max)`` where *L_max* is the
    longest document (in sentences) and *T_max* the longest sentence (in words)
    within the batch.

    Returns
    -------
    docs         : LongTensor  (B, L_max, T_max)  — padded word ids
    sent_lengths : LongTensor  (B, L_max)          — real word count per sentence
                   (clamped to ≥1 so pack_padded_sequence never sees a 0-length)
    doc_lengths  : LongTensor  (B,)                — real sentence count per doc
    labels       : LongTensor  (B,)                — class labels
    """
    doc_ids_list, sent_len_list, doc_len_list, label_list = zip(*batch)

    L_max = max(doc_len_list)
    T_max = max(max(len(s) for s in doc) for doc in doc_ids_list)
    B     = len(batch)

    docs         = torch.full((B, L_max, T_max), pad_id, dtype=torch.long)
    sent_lengths = torch.ones(B, L_max, dtype=torch.long)   # default 1 (not 0)

    for b, (doc_ids, sent_lens, _, _) in enumerate(batch):
        for s, (ids, sl) in enumerate(zip(doc_ids, sent_lens)):
            docs[b, s, :sl] = torch.tensor(ids, dtype=torch.long)
            sent_lengths[b, s] = sl

    doc_lengths = torch.tensor(doc_len_list, dtype=torch.long)
    labels      = torch.tensor(label_list,   dtype=torch.long)

    return docs, sent_lengths, doc_lengths, labels


def make_collate(pad_id: int):
    """
    Return a collate function with *pad_id* baked in, ready for
    ``DataLoader(collate_fn=make_collate(vocab.pad_id))``.
    """
    def _collate(batch):
        return han_collate(batch, pad_id=pad_id)
    return _collate

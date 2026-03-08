from __future__ import annotations

import logging
from collections import Counter

log = logging.getLogger(__name__)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class Vocabulary:
    """
    Bidirectional token ↔ integer-id mapping.

    Build from a corpus with the class method :meth:`from_corpus`, then
    use :meth:`encode` to convert token lists to id lists.
    """

    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: list[str]      = []
        self._add(PAD_TOKEN)   # always id 0
        self._add(UNK_TOKEN)   # always id 1

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_corpus(
        cls,
        docs:      list[list[list[str]]],   # docs → sentences → tokens
        max_vocab: int = 50_000,
        min_freq:  int = 5,
    ) -> "Vocabulary":
        """
        Build a Vocabulary from a nested list of token lists.

        Parameters
        ----------
        docs      : training corpus — a list of documents, each a list of
                    sentences, each a list of string tokens.
        max_vocab : maximum number of real tokens to keep (excluding specials).
        min_freq  : tokens appearing fewer times than this are discarded.
        """
        counter: Counter = Counter()
        for doc in docs:
            for sentence in doc:
                counter.update(sentence)

        vocab = cls()
        for token, freq in counter.most_common(max_vocab):
            if freq >= min_freq:
                vocab._add(token)

        log.info(
            "Vocabulary built: %d tokens  (max_vocab=%d, min_freq=%d)",
            len(vocab), max_vocab, min_freq,
        )
        return vocab

    def _add(self, token: str) -> int:
        """Add *token* if absent; return its id."""
        if token not in self.token2id:
            self.token2id[token] = len(self.id2token)
            self.id2token.append(token)
        return self.token2id[token]

    # ── encoding ──────────────────────────────────────────────────────────────

    def encode(self, tokens: list[str]) -> list[int]:
        """
        Map a list of string tokens to a list of integer ids.
        Unknown tokens are mapped to the UNK id.
        """
        unk = self.token2id[UNK_TOKEN]
        return [self.token2id.get(t, unk) for t in tokens]

    # ── helpers ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.id2token)

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)})"

    @property
    def pad_id(self) -> int:
        """Integer id of the padding token."""
        return self.token2id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        """Integer id of the unknown token."""
        return self.token2id[UNK_TOKEN]

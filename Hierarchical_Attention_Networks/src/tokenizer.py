from __future__ import annotations

import re

# Sentence boundary: whitespace that follows a terminal punctuation mark.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Word token: any contiguous sequence of word characters.
_TOKEN = re.compile(r"\b\w+\b")


def split_sentences(text: str) -> list[str]:
    """
    Split *text* into a list of sentence strings.

    Uses a simple look-behind on  . ! ?  — adequate for English reviews.
    Swap this function for a proper sentence tokeniser if needed.
    """
    sentences = _SENT_BOUNDARY.split(text.strip())
    return [s for s in sentences if s.strip()]


def tokenise(sentence: str) -> list[str]:
    """
    Lowercase and tokenise a single sentence into a list of word strings.

    Example
    -------
    >>> tokenise("Pork belly = delicious!")
    ['pork', 'belly', 'delicious']
    """
    return _TOKEN.findall(sentence.lower())


def tokenise_doc(text: str) -> list[list[str]]:
    """
    Convert a raw document string into a list of token lists, one per sentence.

    Empty sentences (no tokens after tokenisation) are silently dropped.

    Example
    -------
    >>> tokenise_doc("Food was great. Service was slow.")
    [['food', 'was', 'great'], ['service', 'was', 'slow']]
    """
    return [tokenise(s) for s in split_sentences(text) if tokenise(s)]

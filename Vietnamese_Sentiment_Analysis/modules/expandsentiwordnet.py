from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple
import numpy as np

@dataclass
class SentimentEntry:
    """Stores sentiment scores for a single Vietnamese word."""
    word: str
    pos_score: float          # ∈ [0, 1]
    neg_score: float          # ∈ [0, 1]  (= 1 – pos_score for extended words)
    source: str = "original"  # "original" | "expanded"

    @property
    def obj_score(self) -> float:
        """Objectivity score (neutral degree)."""
        return max(0.0, 1.0 - self.pos_score - self.neg_score)

class ViSentiWordNetExtender:
    def __init__(
        self,
        threshold: float = 0.5,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        vn_synonym_dict: Optional[Dict[str, List[str]]] = None,
        vn_antonym_dict: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.threshold = threshold
        self.embedding_fn = embedding_fn
        self.vn_synonym_dict: Dict[str, List[str]] = vn_synonym_dict or {}
        self.vn_antonym_dict: Dict[str, List[str]] = vn_antonym_dict or {}

        # The extended dictionary: word -> SentimentEntry
        self._dictionary: Dict[str, SentimentEntry] = {}

        # Seed sets (populated during build)
        self._positive_set: Set[str] = set()
        self._negative_set: Set[str] = set()

        # Cached embedding matrix for fast distance queries
        self._pos_vectors: Optional[np.ndarray] = None
        self._neg_vectors: Optional[np.ndarray] = None

        self._logger = logging.getLogger(self.__class__.__name__)

    def build(
        self,
        vi_sentiwordnet: Dict[str, Tuple[float, float]],
        new_vietnamese_words: List[str],
    ) -> Dict[str, SentimentEntry]:
        # Seed the dictionary with original entries
        for word, (pos, neg) in vi_sentiwordnet.items():
            self._dictionary[word] = SentimentEntry(
                word=word, pos_score=pos, neg_score=neg, source="original"
            )

        # Step 1 – Extract seed sets P and N from English SentiWordNet
        en_P, en_N = self._extract_seed_sets_from_swn()

        # Merge with Vietnamese seeds already present in ViSentiWordNet
        vi_P = {w for w, (p, n) in vi_sentiwordnet.items() if p > self.threshold and n == 0}
        vi_N = {w for w, (p, n) in vi_sentiwordnet.items() if n > self.threshold and p == 0}

        self._positive_set = en_P | vi_P
        self._negative_set = en_N | vi_N

        self._logger.info(
            "Seed sets: |P|=%d  |N|=%d", len(self._positive_set), len(self._negative_set)
        )

        # Step 2 – Expand P and N via synonyms / antonyms
        self._expand_seed_sets()

        self._logger.info(
            "After expansion: |P|=%d  |N|=%d",
            len(self._positive_set),
            len(self._negative_set),
        )

        # Pre-compute embedding matrices for Step 3
        if self.embedding_fn is not None:
            self._build_embedding_matrices()

        # Step 3 – Score new Vietnamese words and add to dictionary
        added = 0
        for word in new_vietnamese_words:
            if word in self._dictionary:
                continue  # already present with original score
            entry = self._score_word(word)
            self._dictionary[word] = entry
            added += 1

        return self._dictionary

    def score_word(self, word: str) -> Optional[SentimentEntry]:
        """Look up a word; returns *None* if not in the extended dictionary."""
        return self._dictionary.get(word)

    def get_extended_dictionary(self) -> Dict[str, SentimentEntry]:
        """Return the full extended dictionary."""
        return dict(self._dictionary)

    @property
    def positive_set(self) -> Set[str]:
        """Seed + expanded positive word set (P)."""
        return set(self._positive_set)

    @property
    def negative_set(self) -> Set[str]:
        """Seed + expanded negative word set (N)."""
        return set(self._negative_set)

    def _extract_seed_sets_from_swn(self) -> Tuple[Set[str], Set[str]]:
        try:
            _ensure_nltk_resources()
            from nltk.corpus import sentiwordnet as swn
        except Exception as exc:
            return set(), set()

        positive: Set[str] = set()
        negative: Set[str] = set()

        for synset in swn.all_senti_synsets():
            pos_score = synset.pos_score()
            neg_score = synset.neg_score()

            is_pure_positive = pos_score > self.threshold and neg_score == 0.0
            is_pure_negative = neg_score > self.threshold and pos_score == 0.0

            # Extract lemma names and add to the appropriate set
            for lemma in synset.synset.lemma_names():
                lemma = lemma.lower().replace("_", " ")
                if is_pure_positive:
                    positive.add(lemma)
                if is_pure_negative:
                    negative.add(lemma)

        return positive, negative

    def _expand_seed_sets(self) -> None:
        # Snapshot current sets to avoid infinite growth in one pass
        p_snapshot = set(self._positive_set)
        n_snapshot = set(self._negative_set)

        # --- WordNet relations (English) ---
        wn_p_syns, wn_p_ants, wn_n_syns, wn_n_ants = self._wordnet_relations(
            p_snapshot, n_snapshot
        )

        # --- Vietnamese dictionary relations ---
        vi_p_syns: Set[str] = set()
        vi_p_ants: Set[str] = set()
        vi_n_syns: Set[str] = set()
        vi_n_ants: Set[str] = set()

        for word in p_snapshot:
            vi_p_syns.update(self.vn_synonym_dict.get(word, []))
            vi_p_ants.update(self.vn_antonym_dict.get(word, []))
        for word in n_snapshot:
            vi_n_syns.update(self.vn_synonym_dict.get(word, []))
            vi_n_ants.update(self.vn_antonym_dict.get(word, []))

        # Apply expansion rules (paper, Section 4.1 Step 2)
        self._positive_set |= wn_p_syns | wn_n_ants | vi_p_syns | vi_n_ants
        self._negative_set |= wn_n_syns | wn_p_ants | vi_n_syns | vi_p_ants

        # Ensure P ∩ N = ∅  (conflict resolution: remove from both)
        conflict = self._positive_set & self._negative_set
        if conflict:
            self._positive_set -= conflict
            self._negative_set -= conflict

    def _wordnet_relations(
        self,
        p_words: Set[str],
        n_words: Set[str],
    ) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
        """Return (p_synonyms, p_antonyms, n_synonyms, n_antonyms) from WordNet."""
        try:
            _ensure_nltk_resources()
            from nltk.corpus import wordnet as wn
        except Exception as exc:
            return set(), set(), set(), set()

        def _get_synset_relations(words: Set[str]):
            synonyms: Set[str] = set()
            antonyms: Set[str] = set()
            for word in words:
                for synset in wn.synsets(word):
                    for lemma in synset.lemmas():
                        synonyms.add(lemma.name().lower().replace("_", " "))
                        for ant in lemma.antonyms():
                            antonyms.add(ant.name().lower().replace("_", " "))
            return synonyms, antonyms

        p_syns, p_ants = _get_synset_relations(p_words)
        n_syns, n_ants = _get_synset_relations(n_words)
        return p_syns, p_ants, n_syns, n_ants

    def _build_embedding_matrices(self) -> None:
        """Pre-compute stacked embedding matrices for P and N sets."""
        assert self.embedding_fn is not None

        def _stack(words: Set[str]) -> Optional[np.ndarray]:
            vectors = []
            for w in words:
                try:
                    vec = self.embedding_fn(w)
                    if vec is not None and np.any(vec):
                        vectors.append(vec)
                except Exception:
                    pass
            return np.vstack(vectors) if vectors else None

        self._pos_vectors = _stack(self._positive_set)
        self._neg_vectors = _stack(self._negative_set)

    def _average_distance(
        self, word_vec: np.ndarray, set_matrix: Optional[np.ndarray]
    ) -> float:
        if set_matrix is None or set_matrix.shape[0] == 0:
            return 1.0

        # Cosine similarity → distance = 1 – similarity
        norm_word = np.linalg.norm(word_vec)
        if norm_word == 0:
            return 1.0

        norms = np.linalg.norm(set_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)          # avoid /0
        similarities = set_matrix.dot(word_vec) / (norms.ravel() * norm_word)
        distances = 1.0 - similarities
        return float(np.mean(distances))

    def _score_word(self, word: str) -> SentimentEntry:
        if self.embedding_fn is None:
            return SentimentEntry(
                word=word, pos_score=0.5, neg_score=0.5, source="expanded"
            )

        try:
            word_vec = self.embedding_fn(word)
        except Exception as exc:
            return SentimentEntry(
                word=word, pos_score=0.5, neg_score=0.5, source="expanded"
            )

        d_pos = self._average_distance(word_vec, self._pos_vectors)   # dist to P
        d_neg = self._average_distance(word_vec, self._neg_vectors)   # dist to N

        denom = d_pos + d_neg
        pos_score = d_pos / denom if denom > 0 else 0.5  # Equation 5
        neg_score = 1.0 - pos_score                       # Equation 6

        return SentimentEntry(
            word=word,
            pos_score=round(pos_score, 6),
            neg_score=round(neg_score, 6),
            source="expanded",
        )

class SentiVectorExtractor:
    DEFAULT_NEGATION_WORDS: Set[str] = {
        "vô", "bất", "chẳng", "không", "kém",
        "chẳng hề", "không bao giờ", "chẳng bao giờ",
    }

    def __init__(
        self,
        dictionary: Dict[str, SentimentEntry],
        vector_size: int = 128,
        negation_words: Optional[Set[str]] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.dictionary = dictionary
        self.vector_size = vector_size
        self.negation_words = negation_words or self.DEFAULT_NEGATION_WORDS
        self.tokenizer = tokenizer or (lambda text: text.lower().split())

    def extract(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        tokens = self.tokenizer(text)
        pos_vals: List[float] = []
        neg_vals: List[float] = []

        # Detect negation context with a simple bigram window
        negation_flags = self._detect_negation(tokens)

        for idx, token in enumerate(tokens):
            entry = self.dictionary.get(token)
            if entry is None:
                continue

            if negation_flags[idx]:
                # Reversed: positive word in negation context → negative
                pos_vals.append(entry.neg_score)
                neg_vals.append(entry.pos_score)
            else:
                pos_vals.append(entry.pos_score)
                neg_vals.append(entry.neg_score)

        pos_vec = self._vsno(pos_vals)
        neg_vec = self._vsno(neg_vals)
        return pos_vec, neg_vec

    def _detect_negation(self, tokens: List[str]) -> List[bool]:
        flags = [False] * len(tokens)
        for i, token in enumerate(tokens):
            if token in self.negation_words and i + 1 < len(tokens):
                flags[i + 1] = True
        return flags

    def _vsno(self, values: List[float]) -> np.ndarray:
        arr = np.zeros(self.vector_size, dtype=np.float32)
        n = min(len(values), self.vector_size)
        arr[:n] = values[:n]
        return arr

"""Hybrid text retrieval: BM25 (lexical) + TF-IDF cosine (vector), fused with Reciprocal Rank Fusion.

The vector side uses TF-IDF so the project runs offline; swap ``VectorIndex`` for a sentence-transformer
or BGE embedder without touching the rest of the code.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {"the", "a", "an", "of", "to", "in", "on", "and", "is", "it", "for", "by", "which", "what",
             "who", "does", "do", "that", "this", "are", "with", "from", "data"}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.avgdl = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        df = Counter(t for doc in self.tokens for t in set(doc))
        n = len(self.tokens)
        # BM25+ style IDF that stays positive for very common terms
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.tokens))
        for i, doc in enumerate(self.tokens):
            tf = Counter(doc)
            norm = self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
            out[i] = sum(self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf)
        return out


class VectorIndex:
    def __init__(self, docs: Sequence[str]) -> None:
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(docs)

    def scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        return (self.matrix @ q.T).toarray().ravel()  # rows are L2-normalised, so this is cosine


@dataclass
class Hit:
    index: int
    score: float          # fused RRF score
    bm25: float
    cosine: float


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> Dict[int, float]:
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(self, docs: Sequence[str]) -> None:
        self.docs = list(docs)
        self.bm25 = BM25(self.docs)
        self.vec = VectorIndex(self.docs)

    def search(self, query: str, top_k: int = 5) -> List[Hit]:
        b, c = self.bm25.scores(query), self.vec.scores(query)
        rank_b = [i for i in np.argsort(-b) if b[i] > 0]
        rank_c = [i for i in np.argsort(-c) if c[i] > 0]
        fused = rrf([rank_b, rank_c])
        best: List[Tuple[int, float]] = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [Hit(i, s, float(b[i]), float(c[i])) for i, s in best]

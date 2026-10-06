"""BM25 sparse retrieval."""

from __future__ import annotations

import pickle
import re
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from rag_forge.retrieval.base import RetrievalResult, RetrieverBase

if TYPE_CHECKING:
    from rag_forge.chunking import Chunk


class BM25Retriever(RetrieverBase):
    """BM25-based sparse retrieval."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]

    def index(self, chunks: list[Chunk]) -> None:
        """Index chunks using BM25."""
        self._chunks = chunks
        self._tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve chunks using BM25 scoring."""
        if self._bm25 is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        scored_chunks = list(zip(self._chunks, scores, strict=False))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in scored_chunks[:top_k]:
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    retrieval_method="bm25",
                )
            )

        return results

    def save(self, path: str) -> None:
        """Save BM25 index to disk."""
        data = {
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str, chunks: list[Chunk]) -> None:
        """Load BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._chunks = chunks
        self._tokenized_corpus = data["tokenized_corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

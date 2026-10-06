"""Dense embedding-based retrieval using FAISS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import faiss

from rag_forge.retrieval.base import RetrievalResult, RetrieverBase

if TYPE_CHECKING:
    from rag_forge.chunking import Chunk
    from rag_forge.providers.base import EmbeddingProviderBase


class DenseRetriever(RetrieverBase):
    """Dense retrieval using embeddings and FAISS."""

    def __init__(self, embedding_provider: EmbeddingProviderBase) -> None:
        self._embedding_provider = embedding_provider
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        """Index chunks using dense embeddings."""
        self._chunks = chunks

        if not chunks:
            self._index = None
            return

        texts = [c.text for c in chunks]
        embeddings = self._embedding_provider.embed(texts)

        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve chunks using dense similarity."""
        if self._index is None or not self._chunks:
            return []

        query_embedding = self._embedding_provider.embed_single(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        top_k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append(
                RetrievalResult(
                    chunk=self._chunks[idx],
                    score=float(score),
                    retrieval_method="dense",
                )
            )

        return results

    def save(self, path: str) -> None:
        """Save FAISS index to disk."""
        if self._index is not None:
            faiss.write_index(self._index, path)

    def load(self, path: str, chunks: list[Chunk]) -> None:
        """Load FAISS index from disk."""
        self._chunks = chunks
        self._index = faiss.read_index(path)

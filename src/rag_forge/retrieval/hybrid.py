"""Hybrid retrieval combining BM25 and dense methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_forge.retrieval.base import RetrievalResult, RetrieverBase
from rag_forge.retrieval.bm25 import BM25Retriever
from rag_forge.retrieval.dense import DenseRetriever

if TYPE_CHECKING:
    from rag_forge.chunking import Chunk
    from rag_forge.providers.base import EmbeddingProviderBase


class HybridRetriever(RetrieverBase):
    """Hybrid retrieval combining BM25 and dense embeddings.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings from both methods.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProviderBase,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        rrf_k: int = 60,
    ) -> None:
        self._bm25_retriever = BM25Retriever()
        self._dense_retriever = DenseRetriever(embedding_provider)
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight
        self._rrf_k = rrf_k
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        """Index chunks with both BM25 and dense methods."""
        self._chunks = chunks
        self._bm25_retriever.index(chunks)
        self._dense_retriever.index(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve using hybrid approach with RRF fusion."""
        if not self._chunks:
            return []

        fetch_k = min(top_k * 3, len(self._chunks))

        bm25_results = self._bm25_retriever.retrieve(query, fetch_k)
        dense_results = self._dense_retriever.retrieve(query, fetch_k)

        chunk_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievalResult] = {}

        for rank, result in enumerate(bm25_results):
            chunk_id = result.chunk.id
            rrf_score = 1.0 / (self._rrf_k + rank + 1)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + self._bm25_weight * rrf_score
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result

        for rank, result in enumerate(dense_results):
            chunk_id = result.chunk.id
            rrf_score = 1.0 / (self._rrf_k + rank + 1)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + self._dense_weight * rrf_score
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            original_result = chunk_map[chunk_id]
            results.append(
                RetrievalResult(
                    chunk=original_result.chunk,
                    score=score,
                    retrieval_method="hybrid",
                )
            )

        return results

    def retrieve_with_breakdown(
        self, query: str, top_k: int = 5
    ) -> tuple[list[RetrievalResult], list[RetrievalResult], list[RetrievalResult]]:
        """Retrieve and return individual results from each method.

        Returns:
            Tuple of (hybrid_results, bm25_results, dense_results)
        """
        hybrid_results = self.retrieve(query, top_k)
        bm25_results = self._bm25_retriever.retrieve(query, top_k)
        dense_results = self._dense_retriever.retrieve(query, top_k)
        return hybrid_results, bm25_results, dense_results

    def save(self, path: str) -> None:
        """Save both indices."""
        self._bm25_retriever.save(f"{path}.bm25.pkl")
        self._dense_retriever.save(f"{path}.faiss")

    def load(self, path: str, chunks: list[Chunk]) -> None:
        """Load both indices."""
        self._chunks = chunks
        self._bm25_retriever.load(f"{path}.bm25.pkl", chunks)
        self._dense_retriever.load(f"{path}.faiss", chunks)

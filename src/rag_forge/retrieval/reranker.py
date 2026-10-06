"""Reranking components for improving retrieval quality."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_forge.retrieval.base import RetrievalResult


class RerankerBase(ABC):
    """Base class for rerankers."""

    @abstractmethod
    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 3
    ) -> list[RetrievalResult]:
        """Rerank retrieval results.

        Args:
            query: Original query.
            results: Retrieved results to rerank.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked list of results.
        """
        ...


class CrossEncoderReranker(RerankerBase):
    """Cross-encoder based reranker using sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model: object | None = None

    def _get_model(self) -> object:
        """Lazy load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 3
    ) -> list[RetrievalResult]:
        """Rerank results using cross-encoder."""
        if not results:
            return []

        model = self._get_model()

        pairs = [[query, r.chunk.text] for r in results]
        scores = model.predict(pairs)  # type: ignore[union-attr]

        scored_results = list(zip(results, scores, strict=False))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for result, score in scored_results[:top_k]:
            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=float(score),
                    retrieval_method=f"{result.retrieval_method}+rerank",
                )
            )

        return reranked


class NoOpReranker(RerankerBase):
    """No-op reranker that just truncates results."""

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 3
    ) -> list[RetrievalResult]:
        """Return top-k results without reranking."""
        return results[:top_k]

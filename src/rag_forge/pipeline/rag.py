"""Main RAG pipeline implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rag_forge.chunking import Chunk, get_chunker
from rag_forge.config import RetrievalStrategy, Settings
from rag_forge.providers import get_embedding_provider, get_llm_provider
from rag_forge.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    RetrievalResult,
)
from rag_forge.retrieval.reranker import NoOpReranker, RerankerBase

if TYPE_CHECKING:
    from rag_forge.providers.base import EmbeddingProviderBase, LLMProviderBase
    from rag_forge.retrieval.base import RetrieverBase


@dataclass
class Citation:
    """A citation to a source document."""

    source_number: int
    document_title: str
    document_id: str
    chunk_id: str
    text_snippet: str
    relevance_score: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Convert to dictionary."""
        return {
            "source_number": self.source_number,
            "document_title": self.document_title,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "text_snippet": self.text_snippet,
            "relevance_score": self.relevance_score,
        }


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""

    answer: str
    citations: list[Citation]
    retrieval_results: list[RetrievalResult]
    model: str
    total_tokens: int
    query: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "retrieval_results": [r.to_dict() for r in self.retrieval_results],
            "model": self.model,
            "total_tokens": self.total_tokens,
            "query": self.query,
            "metadata": self.metadata,
        }


class RAGPipeline:
    """End-to-end RAG pipeline with document ingestion, retrieval, and generation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._chunks: list[Chunk] = []
        self._embedding_provider: EmbeddingProviderBase | None = None
        self._llm_provider: LLMProviderBase | None = None
        self._retriever: RetrieverBase | None = None
        self._reranker: RerankerBase | None = None
        self._is_indexed = False

    @property
    def embedding_provider(self) -> EmbeddingProviderBase:
        """Get or create embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider(self.settings)
        return self._embedding_provider

    @property
    def llm_provider(self) -> LLMProviderBase:
        """Get or create LLM provider."""
        if self._llm_provider is None:
            self._llm_provider = get_llm_provider(self.settings)
        return self._llm_provider

    @property
    def retriever(self) -> RetrieverBase:
        """Get or create retriever."""
        if self._retriever is None:
            self._retriever = self._create_retriever()
        return self._retriever

    @property
    def reranker(self) -> RerankerBase:
        """Get or create reranker."""
        if self._reranker is None:
            if self.settings.use_reranking:
                self._reranker = CrossEncoderReranker(self.settings.rerank_model)
            else:
                self._reranker = NoOpReranker()
        return self._reranker

    def _create_retriever(self) -> RetrieverBase:
        """Create the appropriate retriever based on settings."""
        if self.settings.retrieval_strategy == RetrievalStrategy.BM25:
            return BM25Retriever()
        elif self.settings.retrieval_strategy == RetrievalStrategy.DENSE:
            return DenseRetriever(self.embedding_provider)
        else:
            return HybridRetriever(
                self.embedding_provider,
                bm25_weight=self.settings.bm25_weight,
                dense_weight=self.settings.dense_weight,
            )

    def ingest_documents(
        self, documents: list[dict[str, str]], show_progress: bool = True
    ) -> list[Chunk]:
        """Ingest documents and create chunks.

        Args:
            documents: List of dicts with 'id', 'title', and 'text' keys.
            show_progress: Whether to show progress bar.

        Returns:
            List of created chunks.
        """
        from tqdm import tqdm

        from rag_forge.gutenberg import clean_gutenberg_text

        chunker = get_chunker(self.settings)
        all_chunks: list[Chunk] = []

        doc_iter = tqdm(documents, desc="Chunking documents") if show_progress else documents
        for doc in doc_iter:
            chunks = chunker.chunk(
                text=clean_gutenberg_text(doc["text"]),
                document_id=doc["id"],
                document_title=doc["title"],
            )
            if chunks:
                # Location queries ("opening line of X") do not overlap the
                # first paragraph. Mark the first chunk so BM25/dense can
                # find the start of the work.
                opening = f"Opening of {doc['title']}.\n\n{chunks[0].text}"
                chunks[0].text = opening
                chunks[0].token_count = chunker.count_tokens(opening)
            all_chunks.extend(chunks)

        self._chunks = all_chunks
        self._is_indexed = False

        return all_chunks

    def build_index(self, show_progress: bool = True) -> None:
        """Build the retrieval index from ingested chunks."""
        if not self._chunks:
            raise ValueError("No chunks to index. Call ingest_documents first.")

        if show_progress:
            from rich.console import Console

            console = Console()
            with console.status("[bold green]Building index..."):
                self.retriever.index(self._chunks)
        else:
            self.retriever.index(self._chunks)

        self._is_indexed = True

    def save_index(self) -> None:
        """Save index and chunks to disk."""
        if not self._is_indexed:
            raise ValueError("No index to save. Call build_index first.")

        chunks_data = [c.to_dict() for c in self._chunks]
        with open(self.settings.get_chunks_path(), "w") as f:
            json.dump(chunks_data, f)

        if self.settings.retrieval_strategy == RetrievalStrategy.BM25:
            self.retriever.save(str(self.settings.get_bm25_path()))
        elif self.settings.retrieval_strategy == RetrievalStrategy.DENSE:
            self.retriever.save(str(self.settings.get_index_path()))
        else:
            self.retriever.save(str(self.settings.cache_dir / "hybrid"))

    def _bm25_index_path(self) -> Path | None:
        """Resolve a BM25 index, including one written by a hybrid ingest."""
        for path in (
            self.settings.get_bm25_path(),
            self.settings.cache_dir / "hybrid.bm25.pkl",
        ):
            if path.exists():
                return path
        return None

    def _dense_index_path(self) -> Path | None:
        """Resolve a FAISS index, including one written by a hybrid ingest."""
        for path in (
            self.settings.get_index_path(),
            self.settings.cache_dir / "hybrid.faiss",
        ):
            if path.exists():
                return path
        return None

    def load_index(self) -> None:
        """Load index and chunks from disk."""
        chunks_path = self.settings.get_chunks_path()
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        with open(chunks_path) as f:
            chunks_data = json.load(f)

        self._chunks = [Chunk.from_dict(c) for c in chunks_data]

        if self.settings.retrieval_strategy == RetrievalStrategy.BM25:
            path = self._bm25_index_path()
            if path is None:
                raise FileNotFoundError(
                    f"BM25 index not found in {self.settings.cache_dir}. "
                    "Run 'rag-forge ingest' first."
                )
            self.retriever.load(str(path), self._chunks)
        elif self.settings.retrieval_strategy == RetrievalStrategy.DENSE:
            path = self._dense_index_path()
            if path is None:
                raise FileNotFoundError(
                    f"Dense index not found in {self.settings.cache_dir}. "
                    "Run 'rag-forge ingest' first."
                )
            self.retriever.load(str(path), self._chunks)
        else:
            hybrid_prefix = self.settings.cache_dir / "hybrid"
            if (
                Path(f"{hybrid_prefix}.bm25.pkl").exists()
                and Path(f"{hybrid_prefix}.faiss").exists()
            ):
                self.retriever.load(str(hybrid_prefix), self._chunks)
            else:
                raise FileNotFoundError(
                    f"Hybrid index not found in {self.settings.cache_dir}. "
                    "Run 'rag-forge ingest --retrieval hybrid' first."
                )

        self._is_indexed = True

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Query string.
            top_k: Number of results (defaults to settings.retrieval_top_k).

        Returns:
            List of retrieval results.
        """
        if not self._is_indexed:
            raise ValueError("Index not built. Call build_index first.")

        top_k = top_k or self.settings.retrieval_top_k
        results = self.retriever.retrieve(query, top_k)

        if self.settings.use_reranking:
            results = self.reranker.rerank(query, results, self.settings.rerank_top_k)

        return results

    def query(
        self,
        question: str,
        top_k: int | None = None,
        system_prompt: str | None = None,
    ) -> RAGResponse:
        """Answer a question using RAG.

        Args:
            question: Question to answer.
            top_k: Number of chunks to retrieve.
            system_prompt: Optional custom system prompt.

        Returns:
            RAGResponse with answer and citations.
        """
        retrieval_results = self.retrieve(question, top_k)

        context_chunks = [r.chunk.text for r in retrieval_results]

        llm_response = self.llm_provider.generate_with_context(
            query=question,
            context_chunks=context_chunks,
            system_prompt=system_prompt,
        )

        citations = []
        for i, result in enumerate(retrieval_results):
            snippet = (
                result.chunk.text[:200] + "..."
                if len(result.chunk.text) > 200
                else result.chunk.text
            )
            citations.append(
                Citation(
                    source_number=i + 1,
                    document_title=result.chunk.document_title,
                    document_id=result.chunk.document_id,
                    chunk_id=result.chunk.id,
                    text_snippet=snippet,
                    relevance_score=result.score,
                )
            )

        return RAGResponse(
            answer=llm_response.content,
            citations=citations,
            retrieval_results=retrieval_results,
            model=llm_response.model,
            total_tokens=llm_response.total_tokens,
            query=question,
            metadata={
                "retrieval_strategy": self.settings.retrieval_strategy.value,
                "use_reranking": self.settings.use_reranking,
            },
        )

    def index_exists(self) -> bool:
        """Check if a saved index exists."""
        chunks_path = self.settings.get_chunks_path()
        if not chunks_path.exists():
            return False

        if self.settings.retrieval_strategy == RetrievalStrategy.BM25:
            return self._bm25_index_path() is not None
        elif self.settings.retrieval_strategy == RetrievalStrategy.DENSE:
            return self._dense_index_path() is not None
        else:
            return self._bm25_index_path() is not None and self._dense_index_path() is not None

    @property
    def chunk_count(self) -> int:
        """Get the number of indexed chunks."""
        return len(self._chunks)

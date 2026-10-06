"""Tests for retrieval components."""

import pytest

from rag_forge.chunking import Chunk
from rag_forge.config import EmbeddingProvider, Settings
from rag_forge.providers import get_embedding_provider
from rag_forge.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


def create_test_chunks() -> list[Chunk]:
    """Create test chunks for retrieval tests."""
    return [
        Chunk(
            id="chunk1",
            text="Python is a programming language known for its simplicity and readability.",
            document_id="doc1",
            document_title="Python Guide",
            start_char=0,
            end_char=73,
            token_count=12,
        ),
        Chunk(
            id="chunk2",
            text="Machine learning is a subset of artificial intelligence.",
            document_id="doc2",
            document_title="ML Basics",
            start_char=0,
            end_char=56,
            token_count=9,
        ),
        Chunk(
            id="chunk3",
            text="Data structures are fundamental to computer science and programming.",
            document_id="doc3",
            document_title="CS Fundamentals",
            start_char=0,
            end_char=67,
            token_count=10,
        ),
        Chunk(
            id="chunk4",
            text="Natural language processing helps computers understand human language.",
            document_id="doc4",
            document_title="NLP Guide",
            start_char=0,
            end_char=70,
            token_count=10,
        ),
    ]


class TestBM25Retriever:
    """Tests for BM25 retrieval."""

    def test_index_and_retrieve(self) -> None:
        retriever = BM25Retriever()
        chunks = create_test_chunks()

        retriever.index(chunks)
        results = retriever.retrieve("Python programming", top_k=2)

        assert len(results) == 2
        assert results[0].chunk.id == "chunk1"
        assert results[0].retrieval_method == "bm25"
        assert results[0].score > 0

    def test_empty_query(self) -> None:
        retriever = BM25Retriever()
        chunks = create_test_chunks()
        retriever.index(chunks)

        results = retriever.retrieve("", top_k=2)

        assert len(results) == 2

    def test_no_index(self) -> None:
        retriever = BM25Retriever()

        results = retriever.retrieve("test query", top_k=2)

        assert len(results) == 0


class TestDenseRetriever:
    """Tests for dense embedding retrieval."""

    @pytest.fixture
    def embedding_provider(self):
        settings = Settings(embedding_provider=EmbeddingProvider.OFFLINE)
        return get_embedding_provider(settings)

    def test_index_and_retrieve(self, embedding_provider) -> None:
        retriever = DenseRetriever(embedding_provider)
        chunks = create_test_chunks()

        retriever.index(chunks)
        results = retriever.retrieve("Python programming", top_k=2)

        assert len(results) == 2
        assert results[0].retrieval_method == "dense"

    def test_empty_chunks(self, embedding_provider) -> None:
        retriever = DenseRetriever(embedding_provider)

        retriever.index([])
        results = retriever.retrieve("test", top_k=2)

        assert len(results) == 0


class TestHybridRetriever:
    """Tests for hybrid retrieval."""

    @pytest.fixture
    def embedding_provider(self):
        settings = Settings(embedding_provider=EmbeddingProvider.OFFLINE)
        return get_embedding_provider(settings)

    def test_hybrid_retrieval(self, embedding_provider) -> None:
        retriever = HybridRetriever(
            embedding_provider,
            bm25_weight=0.3,
            dense_weight=0.7,
        )
        chunks = create_test_chunks()

        retriever.index(chunks)
        results = retriever.retrieve("machine learning AI", top_k=3)

        assert len(results) == 3
        assert results[0].retrieval_method == "hybrid"

    def test_retrieval_with_breakdown(self, embedding_provider) -> None:
        retriever = HybridRetriever(embedding_provider)
        chunks = create_test_chunks()
        retriever.index(chunks)

        hybrid, bm25, dense = retriever.retrieve_with_breakdown("Python", top_k=2)

        assert len(hybrid) == 2
        assert len(bm25) == 2
        assert len(dense) == 2

    def test_weight_configuration(self, embedding_provider) -> None:
        retriever_bm25_heavy = HybridRetriever(
            embedding_provider,
            bm25_weight=0.9,
            dense_weight=0.1,
        )
        retriever_dense_heavy = HybridRetriever(
            embedding_provider,
            bm25_weight=0.1,
            dense_weight=0.9,
        )
        chunks = create_test_chunks()
        retriever_bm25_heavy.index(chunks)
        retriever_dense_heavy.index(chunks)

        results_bm25 = retriever_bm25_heavy.retrieve("Python", top_k=2)
        results_dense = retriever_dense_heavy.retrieve("Python", top_k=2)

        assert len(results_bm25) == 2
        assert len(results_dense) == 2

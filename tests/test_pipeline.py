"""Tests for the RAG pipeline."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag_forge.config import EmbeddingProvider, LLMProvider, RetrievalStrategy, Settings
from rag_forge.pipeline.rag import RAGPipeline, RAGResponse
from rag_forge.providers.base import LLMResponse


@pytest.fixture
def test_documents() -> list[dict[str, str]]:
    """Create test documents."""
    return [
        {
            "id": "doc1",
            "title": "Python Programming",
            "text": "Python is a high-level programming language. It is known for its simplicity and readability. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        },
        {
            "id": "doc2",
            "title": "Machine Learning Basics",
            "text": "Machine learning is a subset of artificial intelligence. It enables computers to learn from data without being explicitly programmed. Common ML techniques include supervised learning, unsupervised learning, and reinforcement learning.",
        },
    ]


@pytest.fixture
def offline_settings(tmp_path: Path) -> Settings:
    """Create settings for offline testing."""
    return Settings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        embedding_provider=EmbeddingProvider.OFFLINE,
        llm_provider=LLMProvider.OFFLINE,
        retrieval_strategy=RetrievalStrategy.HYBRID,
    )


class TestRAGPipeline:
    """Tests for the RAG pipeline."""

    def test_ingest_documents(
        self, test_documents: list[dict[str, str]], offline_settings: Settings
    ) -> None:
        pipeline = RAGPipeline(offline_settings)

        chunks = pipeline.ingest_documents(test_documents, show_progress=False)

        assert len(chunks) > 0
        assert len(chunks) == len(pipeline._chunks)
        first_by_doc = {}
        for chunk in chunks:
            first_by_doc.setdefault(chunk.document_id, chunk)
        for doc in test_documents:
            assert first_by_doc[doc["id"]].text.startswith(f"Opening of {doc['title']}.")

    def test_build_index(
        self, test_documents: list[dict[str, str]], offline_settings: Settings
    ) -> None:
        pipeline = RAGPipeline(offline_settings)
        pipeline.ingest_documents(test_documents, show_progress=False)

        pipeline.build_index(show_progress=False)

        assert pipeline.chunk_count > 0

    def test_query(self, test_documents: list[dict[str, str]], offline_settings: Settings) -> None:
        pipeline = RAGPipeline(offline_settings)
        pipeline.ingest_documents(test_documents, show_progress=False)
        pipeline.build_index(show_progress=False)

        response = pipeline.query("What is Python?")

        assert isinstance(response, RAGResponse)
        assert response.answer is not None
        assert len(response.citations) > 0
        assert response.query == "What is Python?"

    def test_query_returns_citations(
        self, test_documents: list[dict[str, str]], offline_settings: Settings
    ) -> None:
        pipeline = RAGPipeline(offline_settings)
        pipeline.ingest_documents(test_documents, show_progress=False)
        pipeline.build_index(show_progress=False)

        response = pipeline.query("What is Python?", top_k=3)

        assert len(response.citations) <= 3
        for citation in response.citations:
            assert citation.source_number > 0
            assert citation.document_title is not None
            assert citation.text_snippet is not None

    def test_save_and_load_index(
        self, test_documents: list[dict[str, str]], offline_settings: Settings
    ) -> None:
        pipeline1 = RAGPipeline(offline_settings)
        pipeline1.ingest_documents(test_documents, show_progress=False)
        pipeline1.build_index(show_progress=False)
        pipeline1.save_index()

        pipeline2 = RAGPipeline(offline_settings)
        pipeline2.load_index()

        assert pipeline2.chunk_count == pipeline1.chunk_count

        response = pipeline2.query("What is Python?")
        assert response.answer is not None

    def test_query_can_reuse_hybrid_index_for_bm25(
        self, test_documents: list[dict[str, str]], offline_settings: Settings
    ) -> None:
        """After a hybrid ingest, --retrieval bm25 should load hybrid.bm25.pkl."""
        hybrid = RAGPipeline(offline_settings)
        hybrid.ingest_documents(test_documents, show_progress=False)
        hybrid.build_index(show_progress=False)
        hybrid.save_index()

        bm25_settings = Settings(
            data_dir=offline_settings.data_dir,
            cache_dir=offline_settings.cache_dir,
            embedding_provider=EmbeddingProvider.OFFLINE,
            llm_provider=LLMProvider.OFFLINE,
            retrieval_strategy=RetrievalStrategy.BM25,
        )
        bm25_pipeline = RAGPipeline(bm25_settings)
        assert bm25_pipeline.index_exists()
        bm25_pipeline.load_index()
        response = bm25_pipeline.query("What is Python?")
        assert response.answer

    def test_index_exists(
        self, test_documents: list[dict[str, str]], offline_settings: Settings
    ) -> None:
        pipeline = RAGPipeline(offline_settings)

        assert not pipeline.index_exists()

        pipeline.ingest_documents(test_documents, show_progress=False)
        pipeline.build_index(show_progress=False)
        pipeline.save_index()

        assert pipeline.index_exists()

    def test_retrieve_without_index_raises(self, offline_settings: Settings) -> None:
        pipeline = RAGPipeline(offline_settings)

        with pytest.raises(ValueError, match="Index not built"):
            pipeline.retrieve("test query")

    def test_query_with_mocked_local_llm(
        self, test_documents: list[dict[str, str]], offline_settings: Settings
    ) -> None:
        """Retrieve + generate path with a mocked local provider (no model download)."""
        offline_settings.llm_provider = LLMProvider.LOCAL
        pipeline = RAGPipeline(offline_settings)
        pipeline.ingest_documents(test_documents, show_progress=False)
        pipeline.build_index(show_progress=False)

        mock_llm = MagicMock()
        mock_llm.generate_with_context.return_value = LLMResponse(
            content="Python is a high-level language. [Source 1]",
            model="llama3.2:1b",
            prompt_tokens=20,
            completion_tokens=8,
            total_tokens=28,
        )
        pipeline._llm_provider = mock_llm

        response = pipeline.query("What is Python?")

        assert "Python" in response.answer
        assert response.model == "llama3.2:1b"
        mock_llm.generate_with_context.assert_called_once()
        call_kwargs = mock_llm.generate_with_context.call_args.kwargs
        assert call_kwargs["query"] == "What is Python?"
        assert call_kwargs["context_chunks"]

    def test_build_index_without_documents_raises(self, offline_settings: Settings) -> None:
        pipeline = RAGPipeline(offline_settings)

        with pytest.raises(ValueError, match="No chunks"):
            pipeline.build_index(show_progress=False)


class TestRAGResponse:
    """Tests for RAG response dataclass."""

    def test_response_to_dict(self) -> None:
        from rag_forge.pipeline.rag import Citation

        response = RAGResponse(
            answer="Test answer",
            citations=[
                Citation(
                    source_number=1,
                    document_title="Test Doc",
                    document_id="doc1",
                    chunk_id="chunk1",
                    text_snippet="Test snippet",
                    relevance_score=0.95,
                )
            ],
            retrieval_results=[],
            model="test-model",
            total_tokens=100,
            query="Test query",
        )

        d = response.to_dict()

        assert d["answer"] == "Test answer"
        assert len(d["citations"]) == 1
        assert d["model"] == "test-model"
        assert d["query"] == "Test query"

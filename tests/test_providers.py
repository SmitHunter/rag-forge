"""Tests for LLM and embedding providers."""

import numpy as np
import pytest

from rag_forge.config import EmbeddingProvider, LLMProvider, Settings
from rag_forge.providers import get_embedding_provider, get_llm_provider
from rag_forge.providers.local_llm_provider import LocalLLMProvider
from rag_forge.providers.offline_provider import OfflineEmbeddingProvider, OfflineLLMProvider


class TestOfflineEmbeddingProvider:
    """Tests for offline embedding provider."""

    def test_embed_single(self) -> None:
        settings = Settings(embedding_provider=EmbeddingProvider.OFFLINE)
        provider = OfflineEmbeddingProvider(settings)

        embedding = provider.embed_single("Test text")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (provider.dimension,)

    def test_embed_multiple(self) -> None:
        settings = Settings(embedding_provider=EmbeddingProvider.OFFLINE)
        provider = OfflineEmbeddingProvider(settings)
        texts = ["First text", "Second text", "Third text"]

        embeddings = provider.embed(texts)

        assert embeddings.shape == (3, provider.dimension)

    def test_deterministic_embeddings(self) -> None:
        settings = Settings(embedding_provider=EmbeddingProvider.OFFLINE)
        provider = OfflineEmbeddingProvider(settings)

        emb1 = provider.embed_single("Same text")
        emb2 = provider.embed_single("Same text")

        np.testing.assert_array_equal(emb1, emb2)

    def test_different_texts_different_embeddings(self) -> None:
        settings = Settings(embedding_provider=EmbeddingProvider.OFFLINE)
        provider = OfflineEmbeddingProvider(settings)

        emb1 = provider.embed_single("First text")
        emb2 = provider.embed_single("Second text")

        assert not np.allclose(emb1, emb2)

    def test_empty_list(self) -> None:
        settings = Settings(embedding_provider=EmbeddingProvider.OFFLINE)
        provider = OfflineEmbeddingProvider(settings)

        embeddings = provider.embed([])

        assert embeddings.shape == (0, provider.dimension)


class TestOfflineLLMProvider:
    """Tests for offline LLM provider."""

    def test_generate_without_context(self) -> None:
        settings = Settings(llm_provider=LLMProvider.OFFLINE)
        provider = OfflineLLMProvider(settings)

        response = provider.generate("What is Python?")

        assert response.content is not None
        assert len(response.content) > 0
        assert response.model == "offline-template"
        assert response.total_tokens > 0

    def test_generate_with_context(self) -> None:
        settings = Settings(llm_provider=LLMProvider.OFFLINE)
        provider = OfflineLLMProvider(settings)
        context = [
            "Python is a high-level programming language.",
            "Python emphasizes code readability.",
        ]

        response = provider.generate_with_context("What is Python?", context)

        assert response.content is not None
        assert "[Source" in response.content or "context" in response.content.lower()

    def test_empty_context(self) -> None:
        settings = Settings(llm_provider=LLMProvider.OFFLINE)
        provider = OfflineLLMProvider(settings)

        response = provider.generate_with_context("What is Python?", [])

        assert "context" in response.content.lower() or "documents" in response.content.lower()


class TestProviderFactory:
    """Tests for provider factory functions."""

    def test_get_offline_embedding_provider(self) -> None:
        settings = Settings(embedding_provider=EmbeddingProvider.OFFLINE)

        provider = get_embedding_provider(settings)

        assert isinstance(provider, OfflineEmbeddingProvider)

    def test_get_offline_llm_provider(self) -> None:
        settings = Settings(llm_provider=LLMProvider.OFFLINE)

        provider = get_llm_provider(settings)

        assert isinstance(provider, OfflineLLMProvider)

    def test_get_local_llm_provider(self) -> None:
        settings = Settings(llm_provider=LLMProvider.LOCAL, llm_model="llama3.2:1b")

        provider = get_llm_provider(settings)

        assert isinstance(provider, LocalLLMProvider)

    def test_openai_provider_requires_key(self) -> None:
        settings = Settings(
            embedding_provider=EmbeddingProvider.OPENAI,
            openai_api_key=None,
        )

        with pytest.raises(ValueError, match="API key required"):
            get_embedding_provider(settings)

    def test_anthropic_provider_requires_key(self) -> None:
        settings = Settings(
            llm_provider=LLMProvider.ANTHROPIC,
            anthropic_api_key=None,
        )

        with pytest.raises(ValueError, match="API key required"):
            get_llm_provider(settings)

"""Factory functions for creating providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_forge.config import EmbeddingProvider, LLMProvider
from rag_forge.providers.base import EmbeddingProviderBase, LLMProviderBase

if TYPE_CHECKING:
    from rag_forge.config import Settings


def get_embedding_provider(settings: Settings) -> EmbeddingProviderBase:
    """Get the appropriate embedding provider based on settings."""
    if settings.embedding_provider == EmbeddingProvider.OPENAI:
        from rag_forge.providers.openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(settings)

    elif settings.embedding_provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
        from rag_forge.providers.sentence_transformers_provider import SentenceTransformersProvider

        return SentenceTransformersProvider(settings)

    elif settings.embedding_provider == EmbeddingProvider.OFFLINE:
        from rag_forge.providers.offline_provider import OfflineEmbeddingProvider

        return OfflineEmbeddingProvider(settings)

    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def get_llm_provider(settings: Settings) -> LLMProviderBase:
    """Get the appropriate LLM provider based on settings."""
    if settings.llm_provider == LLMProvider.OPENAI:
        from rag_forge.providers.openai_provider import OpenAILLMProvider

        return OpenAILLMProvider(settings)

    elif settings.llm_provider == LLMProvider.ANTHROPIC:
        from rag_forge.providers.anthropic_provider import AnthropicLLMProvider

        return AnthropicLLMProvider(settings)

    elif settings.llm_provider == LLMProvider.LOCAL:
        from rag_forge.providers.local_llm_provider import LocalLLMProvider

        return LocalLLMProvider(settings)

    elif settings.llm_provider == LLMProvider.OFFLINE:
        from rag_forge.providers.offline_provider import OfflineLLMProvider

        return OfflineLLMProvider(settings)

    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

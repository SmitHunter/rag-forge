"""Base classes for LLM and embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from rag_forge.config import Settings


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def usage_summary(self) -> str:
        """Get a summary of token usage."""
        return f"Tokens: {self.prompt_tokens} prompt + {self.completion_tokens} completion = {self.total_tokens} total"


class EmbeddingProviderBase(ABC):
    """Base class for embedding providers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model_name = settings.embedding_model
        self._dimension = settings.embedding_dimension

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    @abstractmethod
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Array of shape (len(texts), dimension) with embeddings.
        """
        ...

    def embed_single(self, text: str) -> NDArray[np.float32]:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            Array of shape (dimension,) with embedding.
        """
        return self.embed([text])[0]


class LLMProviderBase(ABC):
    """Base class for LLM providers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model_name = settings.llm_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Returns:
            LLMResponse with the generated content.
        """
        ...

    @abstractmethod
    def generate_with_context(
        self,
        query: str,
        context_chunks: list[str],
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response with context from retrieved chunks.

        Args:
            query: User query.
            context_chunks: List of relevant context chunks.
            system_prompt: Optional system prompt.

        Returns:
            LLMResponse with the generated answer.
        """
        ...

"""Offline providers for testing and demo mode without API keys."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from rag_forge.providers.base import EmbeddingProviderBase, LLMProviderBase, LLMResponse

if TYPE_CHECKING:
    from rag_forge.config import Settings


class OfflineEmbeddingProvider(EmbeddingProviderBase):
    """Deterministic offline embedding provider for testing.

    Generates consistent embeddings based on text hashing.
    Not suitable for production - use for testing and demos only.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._dimension = settings.embedding_dimension or 384
        self._model_name = "offline-deterministic"

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate deterministic embeddings from text hashes."""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)

        embeddings = []
        for text in texts:
            text_hash = hashlib.sha256(text.lower().encode()).digest()

            rng = np.random.Generator(np.random.PCG64(int.from_bytes(text_hash[:8], "big")))
            embedding = rng.standard_normal(self._dimension).astype(np.float32)

            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)


class OfflineLLMProvider(LLMProviderBase):
    """Offline LLM provider for testing and demos.

    Generates template-based responses using simple heuristics.
    Not suitable for production - use for testing and demos only.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model_name = "offline-template"

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate a template-based response."""
        content = self._generate_response(prompt, [])

        prompt_tokens = len(prompt.split()) + (len(system_prompt.split()) if system_prompt else 0)
        completion_tokens = len(content.split())

        return LLMResponse(
            content=content,
            model=self._model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def generate_with_context(
        self,
        query: str,
        context_chunks: list[str],
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response based on context chunks."""
        content = self._generate_response(query, context_chunks)

        prompt_tokens = len(query.split()) + sum(len(c.split()) for c in context_chunks)
        if system_prompt:
            prompt_tokens += len(system_prompt.split())
        completion_tokens = len(content.split())

        return LLMResponse(
            content=content,
            model=self._model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _generate_response(self, query: str, context_chunks: list[str]) -> str:
        """Generate a simple response based on query and context."""
        if not context_chunks:
            return (
                f"I don't have specific context to answer your question about: {query}. "
                "Please provide relevant documents for me to search through."
            )

        query_words = set(re.findall(r"\b\w+\b", query.lower()))
        query_words -= {"what", "is", "are", "the", "a", "an", "how", "why", "when", "where", "who"}

        relevant_sentences: list[tuple[int, str]] = []
        for i, chunk in enumerate(context_chunks):
            sentences = re.split(r"[.!?]+", chunk)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 20:
                    continue

                sentence_words = set(re.findall(r"\b\w+\b", sentence.lower()))
                overlap = len(query_words & sentence_words)
                if overlap > 0:
                    relevant_sentences.append((i + 1, sentence))

        if not relevant_sentences:
            first_sentences = []
            for i, chunk in enumerate(context_chunks[:2]):
                sentences = re.split(r"[.!?]+", chunk)
                for sentence in sentences[:2]:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 20:
                        first_sentences.append((i + 1, sentence))

            if first_sentences:
                sources_used = sorted(set(s[0] for s in first_sentences))
                response_parts = [s[1] for s in first_sentences[:3]]
                citations = ", ".join(f"[Source {s}]" for s in sources_used)
                return f"Based on the provided documents: {' '.join(response_parts)}. {citations}"

            return (
                "I found the provided context but couldn't extract specific information "
                f"relevant to your question: {query}"
            )

        relevant_sentences = relevant_sentences[:5]
        sources_used = sorted(set(s[0] for s in relevant_sentences))

        response_parts = []
        for source_num, sentence in relevant_sentences[:3]:
            response_parts.append(f"{sentence} [Source {source_num}]")

        return "Based on the provided context: " + " ".join(response_parts)

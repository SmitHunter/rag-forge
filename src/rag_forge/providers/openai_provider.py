"""OpenAI LLM and embedding providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_forge.providers.base import EmbeddingProviderBase, LLMProviderBase, LLMResponse

if TYPE_CHECKING:
    from rag_forge.config import Settings


class OpenAIEmbeddingProvider(EmbeddingProviderBase):
    """Embedding provider using OpenAI API."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key required for OpenAI embedding provider")

        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model_name = settings.embedding_model or "text-embedding-3-small"
        self._dimension = settings.embedding_dimension or 1536

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed texts using OpenAI API."""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)

        texts = [t.replace("\n", " ").strip() for t in texts]

        response = self._client.embeddings.create(
            input=texts,
            model=self._model_name,
        )

        embeddings = np.array([e.embedding for e in response.data], dtype=np.float32)
        return embeddings


class OpenAILLMProvider(LLMProviderBase):
    """LLM provider using OpenAI API."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key required for OpenAI LLM provider")

        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model_name = settings.llm_model or "gpt-4o-mini"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate response using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

    def generate_with_context(
        self,
        query: str,
        context_chunks: list[str],
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate response with context using OpenAI API."""
        context = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )

        default_system = """You are a helpful assistant that answers questions based on the provided context.
Always cite your sources by referring to [Source N] when using information from the context.
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise but thorough in your answers."""

        prompt = f"""Context:
{context}

Question: {query}

Please answer the question based on the context above. Cite your sources using [Source N] notation."""

        return self.generate(prompt, system_prompt or default_system)

"""Anthropic LLM provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tenacity import retry, stop_after_attempt, wait_exponential

from rag_forge.providers.base import LLMProviderBase, LLMResponse

if TYPE_CHECKING:
    from rag_forge.config import Settings


class AnthropicLLMProvider(LLMProviderBase):
    """LLM provider using Anthropic API."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key required for Anthropic LLM provider")

        from anthropic import Anthropic

        self._client = Anthropic(api_key=settings.anthropic_api_key)
        self._model_name = settings.llm_model or "claude-3-haiku-20240307"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate response using Anthropic API."""
        kwargs: dict[str, object] = {
            "model": self._model_name,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)  # type: ignore[arg-type]

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

    def generate_with_context(
        self,
        query: str,
        context_chunks: list[str],
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate response with context using Anthropic API."""
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

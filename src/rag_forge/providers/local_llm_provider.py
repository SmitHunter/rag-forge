"""Local LLM provider using Ollama for CPU/GPU inference.

Ollama provides an easy way to run open-weight models locally.
Install: https://ollama.ai/download
Then: ollama pull llama3.2:1b (or any model you prefer)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import httpx
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from rag_forge.providers.base import LLMProviderBase, LLMResponse

if TYPE_CHECKING:
    from rag_forge.config import Settings

# Settings.llm_model defaults to a cloud model; remap those when using Ollama.
_CLOUD_MODEL_PREFIXES = ("gpt-", "claude-", "o1-", "o3-")


class OllamaNotAvailableError(Exception):
    """Raised when Ollama server is not running or reachable."""

    pass


class LocalLLMProvider(LLMProviderBase):
    """LLM provider using Ollama for local model inference.

    Ollama runs models locally via a simple HTTP API.
    Default model: llama3.2:1b (small, fast, good for demos)

    Setup:
        1. Install Ollama: https://ollama.ai/download
        2. Start server: ollama serve (or it runs automatically)
        3. Pull a model: ollama pull llama3.2:1b
        4. Use this provider: --llm-provider local

    Environment variables:
        RAG_FORGE_LLM_MODEL: Model name (default: llama3.2:1b)
        RAG_FORGE_OLLAMA_HOST / OLLAMA_HOST: Ollama server URL
    """

    DEFAULT_MODEL = "llama3.2:1b"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model_name = self._resolve_model(settings.llm_model)
        self._host = os.environ.get("OLLAMA_HOST") or settings.ollama_host or self.DEFAULT_HOST
        self._client = httpx.Client(timeout=120.0)
        self._verified = False

    def _resolve_model(self, model: str | None) -> str:
        """Use the explicit local model, or fall back when Settings still has a cloud default."""
        if not model or model.startswith(_CLOUD_MODEL_PREFIXES):
            return self.DEFAULT_MODEL
        return model

    def _verify_connection(self) -> None:
        """Verify Ollama is running and model is available."""
        if self._verified:
            return

        try:
            response = self._client.get(f"{self._host}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            if not any(self._model_name in name for name in model_names):
                available = ", ".join(model_names[:5]) if model_names else "none"
                raise OllamaNotAvailableError(
                    f"Model '{self._model_name}' not found. "
                    f"Available: {available}. "
                    f"Run: ollama pull {self._model_name}"
                )
            self._verified = True

        except httpx.ConnectError as e:
            raise OllamaNotAvailableError(
                "Cannot connect to Ollama. Is it running? "
                "Install from https://ollama.ai and run 'ollama serve'"
            ) from e
        except httpx.HTTPStatusError as e:
            raise OllamaNotAvailableError(
                f"Ollama health check failed ({e.response.status_code}). "
                "Is the server running at the configured host?"
            ) from e

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_not_exception_type(OllamaNotAvailableError),
    )
    def _call_ollama(self, prompt: str, system: str | None = None) -> dict:
        """Make a request to Ollama API."""
        self._verify_connection()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.post(
            f"{self._host}/api/chat",
            json={
                "model": self._model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": max(0.1, self._temperature),
                    "num_predict": self._max_tokens,
                },
            },
        )
        response.raise_for_status()
        return response.json()

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate response using Ollama."""
        try:
            result = self._call_ollama(prompt, system_prompt)
        except httpx.HTTPStatusError as e:
            raise OllamaNotAvailableError(
                f"Ollama generate failed ({e.response.status_code}). "
                "Check that the model is pulled and the server is healthy."
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaNotAvailableError(
                f"Ollama timed out talking to {self._host}. "
                "The model may still be loading; retry in a few seconds."
            ) from e

        message = result.get("message", {})
        content = message.get("content", "")

        prompt_tokens = result.get("prompt_eval_count", len(prompt.split()))
        completion_tokens = result.get("eval_count", len(content.split()))

        return LLMResponse(
            content=content.strip(),
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
        """Generate response with retrieved context."""
        context = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )

        default_system = """You are a helpful assistant that answers questions based on the provided context.
Always cite your sources using [Source N] notation when using information from the context.
If the context doesn't contain enough information to answer, say "I cannot answer this based on the provided context."
Be concise and accurate."""

        prompt = f"""Context:
{context}

Question: {query}

Answer the question based on the context above. Cite sources with [Source N]."""

        return self.generate(prompt, system_prompt or default_system)

    def __del__(self) -> None:
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()


def check_ollama_available(host: str = "http://localhost:11434") -> tuple[bool, str]:
    """Check if Ollama is available and return status message.

    Returns:
        Tuple of (is_available, message)
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{host}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            if models:
                names = [m.get("name", "?") for m in models[:3]]
                return True, f"Ollama running with models: {', '.join(names)}"
            return True, "Ollama running but no models installed. Run: ollama pull llama3.2:1b"
    except httpx.ConnectError:
        return False, "Ollama not running. Install from https://ollama.ai and run 'ollama serve'"
    except Exception as e:
        return False, f"Ollama error: {e}"

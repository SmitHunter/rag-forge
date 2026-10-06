"""Tests for local LLM provider (Ollama)."""

from unittest.mock import MagicMock, patch

import pytest

from rag_forge.config import LLMProvider, Settings
from rag_forge.providers.local_llm_provider import (
    LocalLLMProvider,
    OllamaNotAvailableError,
    check_ollama_available,
)


class TestLocalLLMProvider:
    """Tests for LocalLLMProvider with mocked Ollama."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        return Settings(llm_provider=LLMProvider.LOCAL, llm_model="llama3.2:1b")

    @pytest.fixture
    def mock_ollama_response(self) -> dict:
        return {
            "model": "llama3.2:1b",
            "message": {
                "role": "assistant",
                "content": "The answer is 42. [Source 1]",
            },
            "prompt_eval_count": 50,
            "eval_count": 10,
        }

    @pytest.fixture
    def mock_tags_response(self) -> dict:
        return {
            "models": [
                {"name": "llama3.2:1b", "size": 1000000},
                {"name": "mistral:7b", "size": 7000000},
            ]
        }

    def test_cloud_default_model_is_remapped(self) -> None:
        """Settings default (gpt-4o-mini) must not be sent to Ollama."""
        settings = Settings(llm_provider=LLMProvider.LOCAL, llm_model="gpt-4o-mini")
        provider = LocalLLMProvider(settings)
        assert provider._model_name == LocalLLMProvider.DEFAULT_MODEL

    def test_explicit_local_model_is_kept(self) -> None:
        settings = Settings(llm_provider=LLMProvider.LOCAL, llm_model="mistral:7b")
        provider = LocalLLMProvider(settings)
        assert provider._model_name == "mistral:7b"

    def test_ollama_host_from_settings(self) -> None:
        settings = Settings(
            llm_provider=LLMProvider.LOCAL,
            ollama_host="http://127.0.0.1:11435",
        )
        provider = LocalLLMProvider(settings)
        assert provider._host == "http://127.0.0.1:11435"

    def test_generate_success(
        self, mock_settings: Settings, mock_ollama_response: dict, mock_tags_response: dict
    ) -> None:
        """Test successful generation with mocked Ollama."""
        provider = LocalLLMProvider(mock_settings)

        with patch.object(provider, "_client") as mock_client:
            mock_get = MagicMock()
            mock_get.json.return_value = mock_tags_response
            mock_post = MagicMock()
            mock_post.json.return_value = mock_ollama_response
            mock_client.get.return_value = mock_get
            mock_client.post.return_value = mock_post

            response = provider.generate("What is the answer?")

            assert response.content == "The answer is 42. [Source 1]"
            assert response.model == "llama3.2:1b"
            assert response.prompt_tokens == 50
            assert response.completion_tokens == 10

    def test_generate_with_context(
        self, mock_settings: Settings, mock_ollama_response: dict, mock_tags_response: dict
    ) -> None:
        """Test generation with context chunks."""
        provider = LocalLLMProvider(mock_settings)

        with patch.object(provider, "_client") as mock_client:
            mock_get = MagicMock()
            mock_get.json.return_value = mock_tags_response
            mock_post = MagicMock()
            mock_post.json.return_value = mock_ollama_response
            mock_client.get.return_value = mock_get
            mock_client.post.return_value = mock_post

            response = provider.generate_with_context(
                "What is the answer?",
                ["The answer to everything is 42.", "Douglas Adams wrote this."],
            )

            assert "[Source 1]" in response.content
            call_args = mock_client.post.call_args
            assert "Context:" in call_args[1]["json"]["messages"][1]["content"]

    def test_connection_error_raises(self, mock_settings: Settings) -> None:
        """Test that connection errors raise OllamaNotAvailableError."""
        import httpx

        provider = LocalLLMProvider(mock_settings)

        with patch.object(provider, "_client") as mock_client:
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(OllamaNotAvailableError, match="Cannot connect"):
                provider.generate("test")

    def test_model_not_found_raises(
        self, mock_settings: Settings, mock_tags_response: dict
    ) -> None:
        """Test that missing model raises OllamaNotAvailableError."""
        mock_settings_wrong_model = Settings(
            llm_provider=LLMProvider.LOCAL, llm_model="nonexistent:model"
        )
        provider = LocalLLMProvider(mock_settings_wrong_model)

        with patch.object(provider, "_client") as mock_client:
            mock_get = MagicMock()
            mock_get.json.return_value = mock_tags_response
            mock_client.get.return_value = mock_get

            with pytest.raises(OllamaNotAvailableError, match="not found"):
                provider.generate("test")


class TestCheckOllamaAvailable:
    """Tests for the check_ollama_available helper."""

    def test_ollama_available_with_models(self) -> None:
        """Test detection when Ollama is running with models."""
        with patch("rag_forge.providers.local_llm_provider.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "models": [{"name": "llama3.2:1b"}, {"name": "mistral:7b"}]
            }
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__enter__.return_value = mock_client

            available, message = check_ollama_available()

            assert available is True
            assert "llama3.2:1b" in message

    def test_ollama_not_running(self) -> None:
        """Test detection when Ollama is not running."""
        import httpx

        with patch("rag_forge.providers.local_llm_provider.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_class.return_value.__enter__.return_value = mock_client

            available, message = check_ollama_available()

            assert available is False
            assert "not running" in message

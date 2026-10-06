"""Configuration management for RAG Forge."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OFFLINE = "offline"


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    OFFLINE = "offline"


class ChunkingStrategy(str, Enum):
    """Document chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


class RetrievalStrategy(str, Enum):
    """Retrieval strategies."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_FORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"), description="Directory for data files")
    cache_dir: Path = Field(default=Path(".cache"), description="Directory for caches")

    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.SENTENCE_TRANSFORMERS,
        description="Embedding provider to use",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model name",
    )
    embedding_dimension: int = Field(default=384, description="Embedding vector dimension")

    llm_provider: LLMProvider = Field(
        default=LLMProvider.OFFLINE,
        description="LLM provider to use",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model name",
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="LLM temperature")
    llm_max_tokens: int = Field(default=1024, gt=0, description="Maximum tokens in response")

    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL for the local LLM provider",
    )

    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.FIXED_SIZE,
        description="Document chunking strategy",
    )
    chunk_size: int = Field(default=512, gt=0, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap between chunks in tokens")

    retrieval_strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.HYBRID,
        description="Retrieval strategy",
    )
    retrieval_top_k: int = Field(default=5, gt=0, description="Number of chunks to retrieve")
    bm25_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for BM25 in hybrid retrieval"
    )
    dense_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Weight for dense retrieval in hybrid"
    )

    use_reranking: bool = Field(default=False, description="Whether to use reranking")
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking",
    )
    rerank_top_k: int = Field(default=3, gt=0, description="Number of chunks after reranking")

    api_host: str = Field(default="127.0.0.1", description="API server host")
    api_port: int = Field(default=8000, gt=0, lt=65536, description="API server port")

    @field_validator("bm25_weight", "dense_weight")
    @classmethod
    def validate_weights(cls, v: float, info: Any) -> float:
        """Validate retrieval weights."""
        return max(0.0, min(1.0, v))

    def get_index_path(self) -> Path:
        """Get path to the FAISS index file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / "faiss_index.bin"

    def get_bm25_path(self) -> Path:
        """Get path to the BM25 index file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / "bm25_index.pkl"

    def get_chunks_path(self) -> Path:
        """Get path to the chunks metadata file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / "chunks.json"

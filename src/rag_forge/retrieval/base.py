"""Base classes for retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_forge.chunking import Chunk


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    chunk: Chunk
    score: float
    retrieval_method: str

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "retrieval_method": self.retrieval_method,
        }


class RetrieverBase(ABC):
    """Base class for retrieval methods."""

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """Index a list of chunks for retrieval.

        Args:
            chunks: List of document chunks to index.
        """
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult objects, sorted by relevance.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the index to disk.

        Args:
            path: Path to save the index.
        """
        ...

    @abstractmethod
    def load(self, path: str, chunks: list[Chunk]) -> None:
        """Load the index from disk.

        Args:
            path: Path to load the index from.
            chunks: Chunks that were indexed (needed for metadata).
        """
        ...

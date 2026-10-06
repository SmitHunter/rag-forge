"""Document chunking strategies for RAG."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

import tiktoken

if TYPE_CHECKING:
    from rag_forge.config import ChunkingStrategy, Settings


@dataclass
class Chunk:
    """A document chunk with metadata."""

    id: str
    text: str
    document_id: str
    document_title: str
    start_char: int
    end_char: int
    token_count: int
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str | int | float | dict[str, str | int | float]]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | int | float | dict[str, str | int | float]]) -> Chunk:
        """Create from dictionary."""
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            document_id=str(data["document_id"]),
            document_title=str(data["document_title"]),
            start_char=int(data["start_char"]),
            end_char=int(data["end_char"]),
            token_count=int(data["token_count"]),
            metadata=dict(data.get("metadata", {})),
        )


class ChunkerBase(ABC):
    """Base class for document chunking strategies."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._tokenizer.encode(text))

    @abstractmethod
    def chunk(self, text: str, document_id: str, document_title: str) -> list[Chunk]:
        """Chunk a document into smaller pieces."""
        ...

    def _create_chunk(
        self,
        text: str,
        document_id: str,
        document_title: str,
        start_char: int,
        end_char: int,
    ) -> Chunk:
        """Create a chunk with metadata."""
        return Chunk(
            id=str(uuid4()),
            text=text.strip(),
            document_id=document_id,
            document_title=document_title,
            start_char=start_char,
            end_char=end_char,
            token_count=self.count_tokens(text),
        )


class FixedSizeChunker(ChunkerBase):
    """Fixed-size token-based chunking with overlap."""

    def chunk(self, text: str, document_id: str, document_title: str) -> list[Chunk]:
        """Chunk text into fixed-size token windows."""
        tokens = self._tokenizer.encode(text)
        chunks: list[Chunk] = []

        if len(tokens) == 0:
            return chunks

        start_idx = 0
        char_offset = 0

        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self._tokenizer.decode(chunk_tokens)

            start_char = text.find(chunk_text[:50], char_offset)
            if start_char == -1:
                start_char = char_offset
            end_char = start_char + len(chunk_text)

            chunks.append(
                self._create_chunk(chunk_text, document_id, document_title, start_char, end_char)
            )

            char_offset = start_char + len(chunk_text) - self.chunk_overlap * 4
            start_idx = end_idx - self.chunk_overlap

            if end_idx >= len(tokens):
                break

        return chunks


class SentenceChunker(ChunkerBase):
    """Sentence-based chunking that respects sentence boundaries."""

    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def chunk(self, text: str, document_id: str, document_title: str) -> list[Chunk]:
        """Chunk text by sentences, grouping to target size."""
        sentences = self.SENTENCE_PATTERN.split(text)
        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        current_start = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                end_char = current_start + len(chunk_text)
                chunks.append(
                    self._create_chunk(
                        chunk_text, document_id, document_title, current_start, end_char
                    )
                )

                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = self.count_tokens(s)
                    if overlap_tokens + s_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens
                current_start = end_char - len(" ".join(overlap_sentences))

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            end_char = current_start + len(chunk_text)
            chunks.append(
                self._create_chunk(chunk_text, document_id, document_title, current_start, end_char)
            )

        return chunks


class ParagraphChunker(ChunkerBase):
    """Paragraph-based chunking that respects paragraph boundaries."""

    def chunk(self, text: str, document_id: str, document_title: str) -> list[Chunk]:
        """Chunk text by paragraphs, merging small ones."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[Chunk] = []
        current_paragraphs: list[str] = []
        current_tokens = 0
        current_start = 0
        char_position = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                char_position += 2
                continue

            para_tokens = self.count_tokens(para)
            para_start = text.find(para, char_position)
            if para_start == -1:
                para_start = char_position

            if para_tokens > self.chunk_size:
                if current_paragraphs:
                    chunk_text = "\n\n".join(current_paragraphs)
                    end_char = current_start + len(chunk_text)
                    chunks.append(
                        self._create_chunk(
                            chunk_text, document_id, document_title, current_start, end_char
                        )
                    )
                    current_paragraphs = []
                    current_tokens = 0

                sub_chunker = FixedSizeChunker(self.chunk_size, self.chunk_overlap)
                sub_chunks = sub_chunker.chunk(para, document_id, document_title)
                for sc in sub_chunks:
                    sc.start_char += para_start
                    sc.end_char += para_start
                chunks.extend(sub_chunks)
                current_start = para_start + len(para)

            elif current_tokens + para_tokens > self.chunk_size and current_paragraphs:
                chunk_text = "\n\n".join(current_paragraphs)
                end_char = current_start + len(chunk_text)
                chunks.append(
                    self._create_chunk(
                        chunk_text, document_id, document_title, current_start, end_char
                    )
                )
                current_paragraphs = [para]
                current_tokens = para_tokens
                current_start = para_start

            else:
                if not current_paragraphs:
                    current_start = para_start
                current_paragraphs.append(para)
                current_tokens += para_tokens

            char_position = para_start + len(para)

        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            end_char = current_start + len(chunk_text)
            chunks.append(
                self._create_chunk(chunk_text, document_id, document_title, current_start, end_char)
            )

        return chunks


class SemanticChunker(ChunkerBase):
    """Semantic chunking using sentence similarity (requires embeddings)."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.5,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self._model: object | None = None

    def _get_model(self) -> object:
        """Lazy load sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def chunk(self, text: str, document_id: str, document_title: str) -> list[Chunk]:
        """Chunk text based on semantic similarity between sentences."""
        import numpy as np

        sentence_pattern = re.compile(r"(?<=[.!?])\s+")
        sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]

        if len(sentences) <= 1:
            if text.strip():
                return [self._create_chunk(text.strip(), document_id, document_title, 0, len(text))]
            return []

        model = self._get_model()
        embeddings = model.encode(sentences)  # type: ignore[union-attr]

        chunks: list[Chunk] = []
        current_sentences: list[str] = [sentences[0]]
        current_tokens = self.count_tokens(sentences[0])
        current_start = 0

        for i in range(1, len(sentences)):
            sim = float(
                np.dot(embeddings[i - 1], embeddings[i])
                / (np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]))
            )

            sentence_tokens = self.count_tokens(sentences[i])

            should_split = (
                sim < self.similarity_threshold
                or current_tokens + sentence_tokens > self.chunk_size
            )

            if should_split and current_sentences:
                chunk_text = " ".join(current_sentences)
                end_char = text.find(chunk_text, current_start)
                if end_char == -1:
                    end_char = current_start
                end_char += len(chunk_text)

                chunks.append(
                    self._create_chunk(
                        chunk_text, document_id, document_title, current_start, end_char
                    )
                )

                current_sentences = [sentences[i]]
                current_tokens = sentence_tokens
                current_start = text.find(sentences[i], end_char - 100)
                if current_start == -1:
                    current_start = end_char
            else:
                current_sentences.append(sentences[i])
                current_tokens += sentence_tokens

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            end_char = current_start + len(chunk_text)
            chunks.append(
                self._create_chunk(chunk_text, document_id, document_title, current_start, end_char)
            )

        return chunks


def get_chunker(settings: Settings) -> ChunkerBase:
    """Factory function to get the appropriate chunker."""
    from rag_forge.config import ChunkingStrategy

    chunker_map: dict[ChunkingStrategy, type[ChunkerBase]] = {
        ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
        ChunkingStrategy.SENTENCE: SentenceChunker,
        ChunkingStrategy.PARAGRAPH: ParagraphChunker,
        ChunkingStrategy.SEMANTIC: SemanticChunker,
    }

    chunker_class = chunker_map.get(settings.chunking_strategy, FixedSizeChunker)
    return chunker_class(settings.chunk_size, settings.chunk_overlap)

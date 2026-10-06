"""Tests for document chunking strategies."""

from rag_forge.chunking import (
    Chunk,
    FixedSizeChunker,
    ParagraphChunker,
    SentenceChunker,
    get_chunker,
)
from rag_forge.config import ChunkingStrategy, Settings


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_to_dict(self) -> None:
        chunk = Chunk(
            id="test-id",
            text="Test content",
            document_id="doc1",
            document_title="Test Doc",
            start_char=0,
            end_char=12,
            token_count=2,
            metadata={"key": "value"},
        )

        d = chunk.to_dict()

        assert d["id"] == "test-id"
        assert d["text"] == "Test content"
        assert d["document_id"] == "doc1"
        assert d["metadata"] == {"key": "value"}

    def test_chunk_from_dict(self) -> None:
        data = {
            "id": "test-id",
            "text": "Test content",
            "document_id": "doc1",
            "document_title": "Test Doc",
            "start_char": 0,
            "end_char": 12,
            "token_count": 2,
            "metadata": {"key": "value"},
        }

        chunk = Chunk.from_dict(data)

        assert chunk.id == "test-id"
        assert chunk.text == "Test content"
        assert chunk.document_id == "doc1"


class TestFixedSizeChunker:
    """Tests for fixed-size chunking."""

    def test_basic_chunking(self) -> None:
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test document. " * 20

        chunks = chunker.chunk(text, "doc1", "Test Document")

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.document_id == "doc1"
            assert chunk.document_title == "Test Document"
            assert len(chunk.text) > 0

    def test_empty_text(self) -> None:
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)

        chunks = chunker.chunk("", "doc1", "Test")

        assert len(chunks) == 0

    def test_small_text(self) -> None:
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)

        chunks = chunker.chunk("Small text.", "doc1", "Test")

        assert len(chunks) == 1
        assert chunks[0].text == "Small text."


class TestSentenceChunker:
    """Tests for sentence-based chunking."""

    def test_respects_sentence_boundaries(self) -> None:
        chunker = SentenceChunker(chunk_size=50, chunk_overlap=10)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = chunker.chunk(text, "doc1", "Test")

        for chunk in chunks:
            if len(chunk.text) > 20:
                assert not chunk.text.startswith(" ")

    def test_groups_small_sentences(self) -> None:
        chunker = SentenceChunker(chunk_size=200, chunk_overlap=20)
        text = "Short. Very short. Also short. Still short."

        chunks = chunker.chunk(text, "doc1", "Test")

        assert len(chunks) <= 2


class TestParagraphChunker:
    """Tests for paragraph-based chunking."""

    def test_respects_paragraph_boundaries(self) -> None:
        chunker = ParagraphChunker(chunk_size=100, chunk_overlap=10)
        text = (
            "First paragraph content here.\n\nSecond paragraph with more text.\n\nThird paragraph."
        )

        chunks = chunker.chunk(text, "doc1", "Test")

        assert len(chunks) >= 1

    def test_handles_single_paragraph(self) -> None:
        chunker = ParagraphChunker(chunk_size=200, chunk_overlap=20)
        text = "Just one paragraph without any breaks or newlines in the content."

        chunks = chunker.chunk(text, "doc1", "Test")

        assert len(chunks) == 1


class TestChunkerFactory:
    """Tests for the chunker factory function."""

    def test_get_fixed_size_chunker(self) -> None:
        settings = Settings(chunking_strategy=ChunkingStrategy.FIXED_SIZE)

        chunker = get_chunker(settings)

        assert isinstance(chunker, FixedSizeChunker)

    def test_get_sentence_chunker(self) -> None:
        settings = Settings(chunking_strategy=ChunkingStrategy.SENTENCE)

        chunker = get_chunker(settings)

        assert isinstance(chunker, SentenceChunker)

    def test_get_paragraph_chunker(self) -> None:
        settings = Settings(chunking_strategy=ChunkingStrategy.PARAGRAPH)

        chunker = get_chunker(settings)

        assert isinstance(chunker, ParagraphChunker)

    def test_respects_chunk_size_settings(self) -> None:
        settings = Settings(
            chunking_strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=256,
            chunk_overlap=32,
        )

        chunker = get_chunker(settings)

        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 32

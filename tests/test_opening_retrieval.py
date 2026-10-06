"""The 'opening line of X' query must hit the first chapter, not front matter."""

from pathlib import Path

from rag_forge.config import EmbeddingProvider, LLMProvider, RetrievalStrategy, Settings
from rag_forge.pipeline.rag import RAGPipeline

PAP_DOC = {
    "id": "pride_and_prejudice",
    "title": "Pride and Prejudice by Jane Austen",
    "text": """
*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***

with a Preface by George Saintsbury

Pride and Prejudice is a novel whose opening is often quoted in essays
about pride. The critic discusses the author's first rank at length.

Chapter I.                                                  1    “He came down”
Heading to Chapter LVI.                                     431

[Illustration: ·PRIDE AND PREJUDICE·

Chapter I.]

It is a truth universally acknowledged, that a single man in possession
of a good fortune must be in want of a wife.

However little known the feelings or views of such a man may be on his
first entering a neighbourhood, this truth is so well fixed in the minds
of the surrounding families, that he is considered as the rightful
property of some one or other of their daughters.

“My dear Mr. Bennet,” said his lady to him one day, “have you heard that
Netherfield Park is let at last?”

Later, Wickham said that almost all Darcy’s actions may be traced to
pride, and pride has often been his best friend.

*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***
""",
}


def test_opening_line_query_retrieves_famous_sentence(tmp_path: Path) -> None:
    settings = Settings(
        cache_dir=tmp_path / "cache",
        embedding_provider=EmbeddingProvider.OFFLINE,
        llm_provider=LLMProvider.OFFLINE,
        retrieval_strategy=RetrievalStrategy.BM25,
    )
    pipeline = RAGPipeline(settings)
    pipeline.ingest_documents([PAP_DOC], show_progress=False)
    pipeline.build_index(show_progress=False)

    results = pipeline.retrieve("What is the opening line of Pride and Prejudice?", top_k=3)

    assert results
    assert any("truth universally acknowledged" in r.chunk.text.lower() for r in results)
    assert "Saintsbury" not in results[0].chunk.text
    assert results[0].chunk.text.startswith("Opening of Pride and Prejudice")

#!/usr/bin/env python3
"""Download public domain documents from Project Gutenberg for RAG evaluation.

This script downloads a curated set of freely licensed texts from Project Gutenberg,
suitable for demonstrating and evaluating the RAG system.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

GUTENBERG_MIRROR = "https://www.gutenberg.org/cache/epub"

BOOKS = [
    {
        "id": 1342,
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
    },
    {
        "id": 84,
        "title": "Frankenstein",
        "author": "Mary Shelley",
    },
    {
        "id": 1661,
        "title": "The Adventures of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
    },
    {
        "id": 2701,
        "title": "Moby Dick",
        "author": "Herman Melville",
    },
    {
        "id": 11,
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
    },
]


def clean_gutenberg_text(text: str) -> str:
    """Remove Gutenberg wrappers, illustrated front matter, and captions."""
    from rag_forge.gutenberg import clean_gutenberg_text as _clean

    return _clean(text)


def download_book(book_id: int, title: str, output_dir: Path) -> str | None:
    """Download a book from Project Gutenberg."""
    urls = [
        f"{GUTENBERG_MIRROR}/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]

    for url in urls:
        try:
            print(f"  Trying: {url}")
            response = httpx.get(url, follow_redirects=True, timeout=30.0)
            if response.status_code == 200:
                text = response.text
                text = clean_gutenberg_text(text)

                if len(text) < 1000:
                    print(f"  Text too short ({len(text)} chars), trying next URL...")
                    continue

                return text
        except Exception as e:
            print(f"  Error: {e}")
            continue

    return None


def preserve_qa_dataset(qa_path: Path) -> None:
    """Keep the curated QA set. Never overwrite data/qa_dataset.json."""
    if qa_path.exists():
        print(f"\nKeeping existing QA dataset at {qa_path} (not overwriting).")
        return
    print(
        f"\nNo QA dataset at {qa_path}. Restore the curated 76-question set with:\n"
        "  git checkout -- data/qa_dataset.json"
    )


def main() -> None:
    """Download all books and create the dataset."""
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = []

    print("Downloading books from Project Gutenberg...")
    print("(This may take a minute due to rate limiting)\n")

    for book in BOOKS:
        print(f"Downloading: {book['title']} by {book['author']}")

        text = download_book(book["id"], book["title"], output_dir)

        if text:
            doc_id = book["title"].lower().replace(" ", "_").replace("'", "")
            documents.append(
                {
                    "id": doc_id,
                    "title": f"{book['title']} by {book['author']}",
                    "text": text,
                    "metadata": {
                        "author": book["author"],
                        "gutenberg_id": book["id"],
                        "source": "Project Gutenberg",
                        "license": "Public Domain",
                    },
                }
            )
            print(f"  Downloaded: {len(text):,} characters\n")
        else:
            print("  Failed to download\n")

        time.sleep(1)

    docs_path = output_dir / "documents.json"
    with open(docs_path, "w") as f:
        json.dump(documents, f, indent=2)

    print(f"\nSaved {len(documents)} documents to {docs_path}")

    total_chars = sum(len(d["text"]) for d in documents)
    print(f"Total corpus size: {total_chars:,} characters")

    preserve_qa_dataset(output_dir / "qa_dataset.json")


if __name__ == "__main__":
    main()

"""Tests for the FastAPI interface."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_forge import api
from rag_forge.config import EmbeddingProvider, LLMProvider, RetrievalStrategy, Settings
from rag_forge.pipeline.rag import RAGPipeline


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("RAG_FORGE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RAG_FORGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_FORGE_LLM_PROVIDER", "offline")
    monkeypatch.setenv("RAG_FORGE_EMBEDDING_PROVIDER", "offline")
    api.pipeline = None
    yield TestClient(api.app)
    api.pipeline = None


def test_health_without_index(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "no_index"
    assert data["indexed_chunks"] == 0


def test_query_without_index_returns_400(client: TestClient) -> None:
    response = client.post("/query", json={"question": "What is Python?"})
    assert response.status_code == 400
    assert "indexed" in response.json()["detail"].lower()


def test_query_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_rejects_huge_top_k(client: TestClient) -> None:
    response = client.post("/query", json={"question": "What is Python?", "top_k": 999})
    assert response.status_code == 422


def test_root_serves_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "RAG Forge" in response.text
    assert "escapeHtml" in response.text


def test_query_with_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
        embedding_provider=EmbeddingProvider.OFFLINE,
        llm_provider=LLMProvider.OFFLINE,
        retrieval_strategy=RetrievalStrategy.BM25,
    )
    pipeline = RAGPipeline(settings)
    pipeline.ingest_documents(
        [
            {
                "id": "doc1",
                "title": "Python",
                "text": "Python is a high-level programming language used for many applications.",
            }
        ],
        show_progress=False,
    )
    pipeline.build_index(show_progress=False)
    pipeline.save_index()

    monkeypatch.setenv("RAG_FORGE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RAG_FORGE_LLM_PROVIDER", "offline")
    monkeypatch.setenv("RAG_FORGE_EMBEDDING_PROVIDER", "offline")
    monkeypatch.setenv("RAG_FORGE_RETRIEVAL_STRATEGY", "bm25")
    api.pipeline = None
    client = TestClient(api.app)

    health = client.get("/health")
    assert health.json()["indexed_chunks"] > 0

    response = client.post("/query", json={"question": "What is Python?", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"]
    assert data["citations"]

    api.pipeline = None

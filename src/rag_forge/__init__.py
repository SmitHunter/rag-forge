"""RAG Forge: retrieval eval forge with BM25, dense, hybrid, and rerank baselines."""

from rag_forge.config import Settings
from rag_forge.pipeline.rag import RAGPipeline

__version__ = "0.1.0"
__all__ = ["RAGPipeline", "Settings", "__version__"]

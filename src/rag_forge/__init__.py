"""RAG Forge: Production-grade RAG document QA with rigorous evaluation."""

from rag_forge.config import Settings
from rag_forge.pipeline.rag import RAGPipeline

__version__ = "0.1.0"
__all__ = ["RAGPipeline", "Settings", "__version__"]

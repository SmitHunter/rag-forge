"""Sentence Transformers embedding provider (local, no API key needed)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from rag_forge.providers.base import EmbeddingProviderBase

if TYPE_CHECKING:
    from rag_forge.config import Settings


class SentenceTransformersProvider(EmbeddingProviderBase):
    """Embedding provider using Sentence Transformers (runs locally)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model: object | None = None

    def _get_model(self) -> object:
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            dim_fn = getattr(
                self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension
            )
            actual_dim = dim_fn()
            if actual_dim:
                self._dimension = actual_dim
        return self._model

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed texts using Sentence Transformers."""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)

        model = self._get_model()
        embeddings = model.encode(  # type: ignore[union-attr]
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)  # type: ignore[union-attr]

"""SentenceTransformers-based embedding service with lazy loading and caching."""
from __future__ import annotations

import logging
import re
from typing import List

import numpy as np

logger = logging.getLogger("hallucination_guard.embedding_service")


class EmbeddingService:
    """Loads and reuses a sentence-transformers encoder for document embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - environment fallback
                logger.warning("SentenceTransformers is unavailable, using a lightweight fallback embedding path: %s", exc)
                self._model = False
                return None

            try:
                logger.info("Loading embedding model %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # pragma: no cover - environment fallback
                logger.warning("Unable to load embedding model %s, using a lightweight fallback embedding path: %s", self.model_name, exc)
                self._model = False
                return None
        if self._model is False:
            return None
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return one embedding vector per input text (always a list-of-lists,
        never a single flattened matrix wrapped in an outer list)."""
        if not texts:
            return []
        model = self.model
        if model is not None:
            try:
                embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            except Exception as exc:  # pragma: no cover - environment fallback
                logger.warning("Batch embedding failed, retrying per-text encoding: %s", exc)
                embeddings = np.array(
                    [
                        model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                        for text in texts
                    ]
                )
            embeddings = np.asarray(embeddings)
            if embeddings.ndim == 1:
                # A single text was encoded into a single flat vector.
                embeddings = embeddings.reshape(1, -1)
            return [row.tolist() for row in embeddings]
        return [self._fallback_embed(text) for text in texts]

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def _fallback_embed(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        vector = np.zeros(32, dtype=np.float32)
        for token in set(tokens):
            vector[abs(hash(token)) % 32] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector.tolist()


def _default_model_name() -> str:
    try:
        from config import settings

        return getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    except Exception:  # pragma: no cover - config should always import
        return "all-MiniLM-L6-v2"


embedding_service = EmbeddingService(model_name=_default_model_name())

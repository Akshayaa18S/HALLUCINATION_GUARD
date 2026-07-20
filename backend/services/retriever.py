"""FAISS-backed retriever for evidence documents from multiple knowledge sources."""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple

import faiss
import numpy as np

from services.embedding_service import embedding_service

logger = logging.getLogger("hallucination_guard.retriever")


class RetrievalDocument:
    """A single document in the retrieval index."""

    def __init__(self, text: str, source: str, metadata: Optional[dict] = None) -> None:
        self.text = text
        self.source = source
        self.metadata = metadata or {}


class Retriever:
    """Builds and queries a FAISS index over evidence documents."""

    def __init__(self, documents: Optional[Sequence[RetrievalDocument]] = None) -> None:
        self.documents: List[RetrievalDocument] = list(documents or [])
        self.index: Optional[faiss.Index] = None
        self._embeddings: Optional[np.ndarray] = None
        self._is_built = False

    def build_index(self, documents: Optional[Sequence[RetrievalDocument]] = None) -> None:
        docs = list(documents or self.documents)
        if not docs:
            self.index = None
            self._embeddings = None
            self._is_built = False
            return

        self.documents = docs
        embeddings = np.array(embedding_service.embed_texts([doc.text for doc in docs]), dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self._embeddings = embeddings
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self._is_built = True

    @property
    def is_built(self) -> bool:
        return self._is_built and self.index is not None

    def load_index(self, path: Optional[str] = None) -> None:
        if path is None:
            return
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            self._is_built = True
            logger.info("Loaded FAISS index from %s", path)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[RetrievalDocument, float]]:
        if not self._is_built or self.index is None or not self.documents:
            self.build_index(self.documents)
            if self.index is None or not self.documents:
                return []

        query_embedding = np.array([embedding_service.embed_text(query)], dtype="float32")
        scores, indices = self.index.search(query_embedding, min(k, len(self.documents)))
        results: List[Tuple[RetrievalDocument, float]] = []
        for score, index in zip(scores[0], indices[0]):
            if int(index) < 0 or int(index) >= len(self.documents):
                continue
            results.append((self.documents[int(index)], float(score)))
        return results

    def retrieve_top_k(self, query: str, k: int = 5) -> List[Tuple[RetrievalDocument, float]]:
        return self.retrieve(query, k=k)


retriever = Retriever()

"""Real evidence retrieval for Stage 6 (Fact Verification).

Replaces the old two-topic hardcoded document list with:

  1. A **static** corpus (FEVER + HaluEval seed facts, see
     ``data/knowledge/*.jsonl``) that is embedded and indexed with FAISS
     exactly once, then persisted to ``data/index/static.faiss`` and reused
     on every subsequent call/process start (``load()`` is idempotent).
  2. **Live Wikipedia retrieval** via the ``wikipedia`` package, so queries
     about topics that aren't in the static seed set still get real,
     on-topic evidence instead of silently falling back to unrelated facts.

``KnowledgeBase.retrieve(query, k)`` merges both sources, re-scores every
candidate against the query with the shared embedding model, deduplicates,
and returns the top ``k`` ``(RetrievalDocument, score)`` pairs — the same
contract ``Retriever.retrieve_top_k`` already exposes, so callers (e.g.
``verification_service``) don't need to know which source a document came
from.

Nothing here fabricates facts or sources: if neither the static corpus nor
Wikipedia can produce evidence for a query, ``retrieve`` returns an empty
list and callers must say so explicitly (see
``verification_service.VerificationService.verify``).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from services.embedding_service import embedding_service
from services.retriever import RetrievalDocument, Retriever

logger = logging.getLogger("hallucination_guard.knowledge_base")

_BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _BACKEND_DIR / "data" / "knowledge"
INDEX_DIR = _BACKEND_DIR / "data" / "index"
STATIC_INDEX_PATH = INDEX_DIR / "static.faiss"

try:  # optional dependency, see requirements-ml.txt
    import wikipedia
except ImportError:  # pragma: no cover - optional dependency until installed
    wikipedia = None


def _load_jsonl(path: Path, default_source: str) -> List[RetrievalDocument]:
    """Load one knowledge file. Skips malformed lines instead of crashing
    startup on a bad edit."""
    documents: List[RetrievalDocument] = []
    if not path.exists():
        return documents
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %s in %s", line_number, path.name)
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            documents.append(
                RetrievalDocument(
                    text=text,
                    source=row.get("source") or default_source,
                    metadata=row.get("metadata") or {},
                )
            )
    return documents


def _load_static_documents() -> List[RetrievalDocument]:
    documents: List[RetrievalDocument] = []
    if not DATA_DIR.exists():
        logger.warning("Knowledge data directory %s does not exist; static corpus is empty", DATA_DIR)
        return documents
    for jsonl_file in sorted(DATA_DIR.glob("*.jsonl")):
        default_source = jsonl_file.stem.replace("_", " ").title()
        documents.extend(_load_jsonl(jsonl_file, default_source=default_source))
    return documents


class KnowledgeBase:
    """Static FAISS corpus + live Wikipedia retrieval, combined."""

    def __init__(self, wikipedia_results: int = 3) -> None:
        self._static_retriever = Retriever()
        self._loaded = False
        self.wikipedia_results = wikipedia_results

    def load(self, force_rebuild: bool = False) -> None:
        """Load (or build+persist) the static FAISS index exactly once."""
        if self._loaded and not force_rebuild:
            return

        documents = _load_static_documents()
        if not documents:
            logger.warning("No static knowledge documents found under %s", DATA_DIR)
            self._loaded = True
            return

        self._static_retriever.documents = documents
        if not force_rebuild and STATIC_INDEX_PATH.exists():
            try:
                self._static_retriever.load_index(str(STATIC_INDEX_PATH))
                logger.info("Loaded cached static knowledge index with %d documents", len(documents))
                self._loaded = True
                return
            except Exception as exc:  # pragma: no cover - corrupted/incompatible index file
                logger.warning("Cached FAISS index could not be loaded, rebuilding: %s", exc)

        self._static_retriever.build_index(documents)
        self._persist_index()
        self._loaded = True
        logger.info("Built static knowledge index with %d documents", len(documents))

    def _persist_index(self) -> None:
        try:
            import faiss  # local import: keep this module importable without faiss installed

            if self._static_retriever.index is None:
                return
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._static_retriever.index, str(STATIC_INDEX_PATH))
        except Exception as exc:  # pragma: no cover - best-effort caching
            logger.warning("Could not persist static FAISS index (will rebuild next time): %s", exc)

    def _live_wikipedia(self, query: str) -> List[RetrievalDocument]:
        """Best-effort live Wikipedia lookup. Never raises — returns []
        on any failure (offline, rate-limited, disambiguation, etc.)."""
        if wikipedia is None:
            return []
        documents: List[RetrievalDocument] = []
        try:
            titles = wikipedia.search(query, results=self.wikipedia_results)
        except Exception as exc:
            logger.info("Wikipedia search unavailable for %r: %s", query, exc)
            return []

        for title in titles:
            try:
                summary = wikipedia.summary(title, sentences=4, auto_suggest=False)
            except Exception as exc:
                logger.info("Skipping Wikipedia page %r: %s", title, exc)
                continue
            documents.append(
                RetrievalDocument(text=summary, source="Wikipedia", metadata={"title": title})
            )
        return documents

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[RetrievalDocument, float]]:
        """Return the top-k evidence documents for ``query`` across every
        configured source, ranked by embedding similarity."""
        self.load()

        candidates: List[RetrievalDocument] = []
        if self._static_retriever.is_built:
            static_hits = self._static_retriever.retrieve_top_k(query, k=max(k, 5))
            candidates.extend(doc for doc, _score in static_hits)
        candidates.extend(self._live_wikipedia(query))

        if not candidates:
            return []

        deduped: List[RetrievalDocument] = []
        seen_text = set()
        for doc in candidates:
            text = (doc.text or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen_text:
                continue
            seen_text.add(key)
            deduped.append(doc)

        query_vector = np.array(embedding_service.embed_text(query), dtype="float32")
        doc_vectors = np.array(
            embedding_service.embed_texts([doc.text for doc in deduped]), dtype="float32"
        )
        # embedding_service normalizes vectors, so a dot product is cosine similarity.
        scores = doc_vectors @ query_vector
        ranked = sorted(zip(deduped, scores.tolist()), key=lambda pair: pair[1], reverse=True)
        return ranked[:k]


def _default_wikipedia_results() -> int:
    try:
        from config import settings

        return int(getattr(settings, "WIKIPEDIA_RESULTS_PER_QUERY", 3))
    except Exception:  # pragma: no cover - config should always import
        return 3


knowledge_base = KnowledgeBase(wikipedia_results=_default_wikipedia_results())

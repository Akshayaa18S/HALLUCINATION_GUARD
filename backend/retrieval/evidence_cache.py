"""
Local Evidence Disk Cache for Hallucination Guard.

Provides deterministic, persistent disk caching for retrieved evidence snippets
in `backend/data/evidence_cache.json`. Eliminates live Wikipedia HTTP 429 rate limits,
network failures, and non-deterministic page changes during evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

backend_dir = Path(__file__).resolve().parent.parent
CACHE_FILE_PATH = backend_dir / "data" / "evidence_cache.json"

logger = logging.getLogger("hallucination_guard.evidence_cache")


class LocalEvidenceCache:
    """Manages persistent JSON disk cache mapping prompt/query to evidence snippet dicts."""

    def __init__(self, cache_file: str | Path | None = None):
        self.cache_file = Path(cache_file) if cache_file else CACHE_FILE_PATH
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self.load()

    def _normalize_key(self, query: str) -> str:
        return query.strip().lower()

    def load(self) -> None:
        """Loads cached evidence snippets from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info("Loaded %d cached queries from evidence cache '%s'", len(self._cache), self.cache_file)
            except Exception as exc:
                logger.warning("Failed to read evidence cache file (%s); starting empty.", exc)
                self._cache = {}
        else:
            self._cache = {}

    def save(self) -> None:
        """Persists evidence cache to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            logger.info("Saved %d queries to evidence cache '%s'", len(self._cache), self.cache_file)
        except Exception as exc:
            logger.error("Failed to save evidence cache to disk: %s", exc)

    def get(self, query: str) -> list[dict[str, Any]] | None:
        """Returns cached evidence snippets for query if available."""
        key = self._normalize_key(query)
        return self._cache.get(key)

    def put(self, query: str, snippets: list[dict[str, Any]]) -> None:
        """Stores evidence snippets for query in cache."""
        key = self._normalize_key(query)
        self._cache[key] = snippets

    def get_or_fetch(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Returns cached evidence snippets if available; otherwise fetches with rate-limit retries."""
        key = self._normalize_key(query)
        if key in self._cache:
            return self._cache[key]

        # Fetch from WikipediaRetriever with exponential backoff retries
        snippets: list[dict[str, Any]] = []
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from retrieval.wikipedia_retriever import WikipediaRetriever
                retriever = WikipediaRetriever()
                import asyncio
                snippets = asyncio.run(retriever.retrieve(query, top_k=top_k))
                if snippets:
                    break
            except Exception as exc:
                logger.warning("Attempt %d/%d failed for query '%s': %s", attempt + 1, max_retries, query[:40], exc)
                time.sleep(1.0 * (2 ** attempt))

        # Store in cache even if empty to prevent repeated failing requests
        self._cache[key] = snippets
        return snippets


_GLOBAL_CACHE: LocalEvidenceCache | None = None


def get_evidence_cache() -> LocalEvidenceCache:
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = LocalEvidenceCache()
    return _GLOBAL_CACHE

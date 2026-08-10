"""
Retrieval Layer - Pluggable Evidence Provider Interface.

Provides a unified EvidenceProvider interface abstraction supporting Wikipedia, FEVER,
Local RAG, and custom Knowledge Bases.
"""

from __future__ import annotations

import abc
import asyncio
import concurrent.futures
import logging
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


class BaseEvidenceProvider(abc.ABC):
    """Abstract Base Class for Evidence Providers."""

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'wikipedia', 'fever', 'local_rag')."""
        pass

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Synchronously or asynchronously retrieves top_k evidence snippets for a query."""
        pass


class WikipediaProvider(BaseEvidenceProvider):
    """Wikipedia Live & Cached API Evidence Provider."""

    @property
    def provider_name(self) -> str:
        return "wikipedia"

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        from retrieval.wikipedia_retriever import WikipediaRetriever
        retriever = WikipediaRetriever()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                snippets = pool.submit(lambda: asyncio.run(retriever.retrieve(query, top_k=top_k))).result(timeout=10)
        else:
            snippets = asyncio.run(retriever.retrieve(query, top_k=top_k))

        for item in snippets:
            item["provider"] = self.provider_name
        return snippets


class FEVERProvider(BaseEvidenceProvider):
    """FEVER Dataset Evidence Provider."""

    @property
    def provider_name(self) -> str:
        return "fever"

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not getattr(settings, "fever_dataset_path", None):
            return []
        try:
            from retrieval.fever_retriever import FEVERRetriever
            retriever = FEVERRetriever()
            snippets = retriever.retrieve(query, top_k=top_k)
            for item in snippets:
                item["provider"] = self.provider_name
            return snippets
        except Exception as exc:
            logger.debug("FEVER retrieval fallback: %s", exc)
            return []


class LocalRAGProvider(BaseEvidenceProvider):
    """Local Knowledge Base / RAG Evidence Provider Fallback."""

    @property
    def provider_name(self) -> str:
        return "local_rag"

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return []


class EvidenceProviderRegistry:
    """Registry coordinating query retrieval across multiple pluggable evidence providers."""

    def __init__(self, providers: list[BaseEvidenceProvider] | None = None):
        if providers is None:
            self.providers = [WikipediaProvider(), FEVERProvider(), LocalRAGProvider()]
        else:
            self.providers = providers

    def retrieve_per_claim(self, claim: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Queries providers in order of priority to fetch evidence for a single claim."""
        all_results: list[dict[str, Any]] = []

        for provider in self.providers:
            try:
                res = provider.retrieve(claim, top_k=top_k)
                if res:
                    all_results.extend(res)
                    if len(all_results) >= top_k:
                        break
            except Exception as exc:
                logger.warning("Provider '%s' failed for claim '%s': %s", provider.provider_name, claim, exc)

        return all_results[:top_k]

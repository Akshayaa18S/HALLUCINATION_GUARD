"""
In-memory SHA256 key-value cache for intermediate LLM calls (claim extraction, verification).
Ensures 100% deterministic re-execution for identical inputs without extra LLM latency.
"""

import hashlib
import json
from typing import Any

from config.settings import settings


class IntermediateCache:
    def __init__(self):
        self._cache: dict[str, Any] = {}

    def _make_key(self, namespace: str, prompt: str, system: str | None = None) -> str:
        raw = f"{namespace}:{prompt}:{system or ''}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, namespace: str, prompt: str, system: str | None = None) -> Any | None:
        if not getattr(settings, "enable_intermediate_cache", True):
            return None
        key = self._make_key(namespace, prompt, system)
        return self._cache.get(key)

    def set(self, namespace: str, prompt: str, system: str | None, value: Any) -> None:
        if not getattr(settings, "enable_intermediate_cache", True):
            return
        key = self._make_key(namespace, prompt, system)
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()


intermediate_cache = IntermediateCache()

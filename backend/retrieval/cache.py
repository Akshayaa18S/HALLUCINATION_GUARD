"""
Lightweight disk cache for retrieval results.

Not using diskcache/redis to keep the dependency list small - this is a
JSON-file-per-key cache under settings.cache_dir, good enough for a
single-instance backend. Swap for Redis later without touching callers,
since they only see get()/set().
"""

import hashlib
import json
import logging
import time
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


class DiskCache:
    def __init__(self, namespace: str, ttl_seconds: int | None = None):
        self.namespace = namespace
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds
        self.dir = settings.cache_dir_path / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.dir / f"{digest}.json"

    def get(self, key: str):
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        if time.time() - payload["cached_at"] > self.ttl_seconds:
            path.unlink(missing_ok=True)
            return None
        return payload["value"]

    def set(self, key: str, value) -> None:
        path = self._key_path(key)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump({"cached_at": time.time(), "value": value}, f)
        except OSError as exc:
            logger.warning("Cache write failed for %s: %s", key, exc)

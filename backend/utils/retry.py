"""Small async retry decorator with exponential backoff.

Avoids pulling in tenacity as a dependency for something this simple.
Used by Phase 3 (LLM), Phase 7 (retrieval) and anywhere else doing
network I/O that can transiently fail.
"""

import asyncio
import functools
import logging

logger = logging.getLogger(__name__)


def async_retry(max_attempts: int = 3, base_delay: float = 0.5, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                        func.__name__, attempt, max_attempts, exc, delay,
                    )
                    await asyncio.sleep(delay)
            raise last_exc

        return wrapper

    return decorator

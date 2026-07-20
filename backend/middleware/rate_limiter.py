"""
Simple in-memory rate limiting middleware.
Not suitable for multi-process deployments; replace with Redis-based limiter in production.
"""
import time
import asyncio
from typing import Dict, Tuple
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from config import settings

# store: ip -> (count, reset_ts)
_store: Dict[str, Tuple[int, float]] = {}
_lock = asyncio.Lock()

class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests_per_minute: int = None):
        super().__init__(app)
        self.max_requests = max_requests_per_minute or settings.RATE_LIMIT_RPM
        self.window_seconds = 60

    async def dispatch(self, request: Request, call_next):
        # Only apply to API paths
        path = request.url.path
        if not path.startswith("/api"):
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.time()

        async with _lock:
            count, reset = _store.get(client, (0, now + self.window_seconds))
            if now >= reset:
                # reset window
                count = 0
                reset = now + self.window_seconds

            if count >= self.max_requests:
                retry_after = int(reset - now)
                headers = {
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                }
                return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429, headers=headers)

            # increment
            count += 1
            _store[client] = (count, reset)
            remaining = max(0, self.max_requests - count)

        response = await call_next(request)
        # attach headers
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

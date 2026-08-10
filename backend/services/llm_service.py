"""
Phase 3 - LLM Service.

Only responsible for: Input -> Generated Response. No verification, no
hallucination logic here by design (that's Phases 8/9) - keeps this
service swappable (e.g. point at a different Ollama model, or a hosted
API) without touching the rest of the pipeline.
"""

import logging

import httpx

from config.settings import settings
from utils.retry import async_retry

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    pass


class LLMService:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model

    @async_retry(max_attempts=3, base_delay=1.0, exceptions=(httpx.HTTPError,))
    async def generate(self, prompt: str, system: str | None = None, timeout: float = 60.0) -> str:
        """Send a prompt to Ollama and return the raw generated text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": getattr(settings, "llm_temperature", 0.0),
                "seed": getattr(settings, "llm_seed", 42),
                "top_k": getattr(settings, "llm_top_k", 1),
                "top_p": getattr(settings, "llm_top_p", 1.0),
            },
        }
        if system:
            payload["system"] = system


        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.error("Ollama request failed: %s", exc)
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMServiceError(f"Unexpected LLM error: {exc}") from exc

        text = data.get("response", "")
        if not text:
            raise LLMServiceError("Ollama returned an empty response")
        return text.strip()

    async def is_available(self) -> bool:
        """Quick health check - used by /health and to decide whether
        LLM-based extraction (Phase 5) should fall back to rule-based."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

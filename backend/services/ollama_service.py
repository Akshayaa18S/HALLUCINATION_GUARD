"""Reusable Ollama client for local LLM generation and structured prompting."""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from config import settings

logger = logging.getLogger("hallucination_guard.ollama_service")


class OllamaServiceError(Exception):
    """Raised when Ollama is unavailable or returns unusable output."""


class OllamaService:
    """Thin wrapper around the local Ollama REST API with retry and timeout handling."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, timeout: Optional[int] = None, retries: int = 2) -> None:
        self.base_url = (base_url or getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or getattr(settings, "OLLAMA_MODEL", "llama3.2:3b")
        self.timeout = timeout or getattr(settings, "LLM_REQUEST_TIMEOUT_SECONDS", 60)
        self.retries = max(1, retries)

    def _build_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def generate(self, prompt: str, *, system_prompt: Optional[str] = None, format_json: bool = False) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if format_json:
            payload["format"] = "json"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                return self._post(payload)
            except OllamaServiceError as exc:
                last_error = exc
                if attempt < self.retries:
                    logger.warning("Ollama request attempt %s/%s failed: %s", attempt, self.retries, exc)
                    time.sleep(0.5 * attempt)
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise OllamaServiceError("Ollama request failed without a captured error.")

    def generate_json(self, prompt: str, *, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        raw_text = self.generate(prompt, system_prompt=system_prompt, format_json=True)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise OllamaServiceError(f"Ollama returned invalid JSON: {exc}") from exc

    def _post(self, payload: Dict[str, Any]) -> str:
        request = urllib.request.Request(
            self._build_url("/api/generate"),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise OllamaServiceError(f"Unable to reach Ollama at {self.base_url}: {exc}") from exc
        except TimeoutError as exc:
            raise OllamaServiceError(f"Ollama request timed out after {self.timeout}s") from exc
        except Exception as exc:  # pragma: no cover - network/SDK errors
            raise OllamaServiceError(f"Ollama request failed: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaServiceError(f"Ollama returned invalid JSON: {exc}") from exc

        text = (data.get("response") or "").strip()
        if not text:
            raise OllamaServiceError("Ollama returned an empty response.")
        return text


ollama_service = OllamaService()

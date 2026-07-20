"""
LLM-backed response generation and hallucination/fact verification.

The backend now prefers a local Ollama endpoint for development and demos,
while still keeping Anthropic as a fallback when configured.

Configure via environment variables (see backend/.env):
    OLLAMA_BASE_URL     - defaults to http://localhost:11434
    OLLAMA_MODEL        - defaults to llama3.2:3b
    ANTHROPIC_API_KEY   - optional fallback for live model output
    ANTHROPIC_MODEL     - defaults to claude-3-5-sonnet-20241022
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger("hallucination_guard.llm_service")

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency until installed
    anthropic = None


class LLMServiceError(Exception):
    """Raised when the LLM backend is unavailable or returns something unusable."""


_client: Optional["anthropic.Anthropic"] = None


def _get_client() -> "anthropic.Anthropic":
    global _client
    if anthropic is None:
        raise LLMServiceError(
            "The 'anthropic' package is not installed. Run "
            "`pip install anthropic` (see backend/requirements.txt)."
        )
    if not settings.ANTHROPIC_API_KEY:
        raise LLMServiceError(
            "ANTHROPIC_API_KEY is not set. Add it to backend/.env to enable "
            "live model output instead of hardcoded responses."
        )
    if _client is None:
        _client = anthropic.Anthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )
    return _client


def _extract_text(message: Any) -> str:
    return "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    ).strip()


def _build_ollama_url(path: str) -> str:
    base_url = (settings.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
    return f"{base_url}{path}"


def _call_ollama(prompt: str, *, system_prompt: Optional[str] = None, json_mode: bool = False) -> str:
    payload: Dict[str, Any] = {
        "model": settings.OLLAMA_MODEL or "llama3.2:3b",
        "prompt": prompt,
        "stream": False,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if json_mode:
        payload["format"] = "json"

    request = urllib.request.Request(
        _build_ollama_url("/api/generate"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except Exception as exc:  # pragma: no cover - network/SDK errors
        raise LLMServiceError(f"Ollama request failed: {exc}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMServiceError(f"Ollama returned invalid JSON: {exc}") from exc

    text = (data.get("response") or "").strip()
    if not text:
        raise LLMServiceError("Ollama returned an empty response.")
    return text


def _generate_response_with_anthropic(prompt: str) -> str:
    client = _get_client()
    try:
        message = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # pragma: no cover - network/SDK errors
        raise LLMServiceError(f"Response generation failed: {exc}") from exc

    text = _extract_text(message)
    if not text:
        raise LLMServiceError("Model returned an empty response.")
    return text


def generate_response(query: str, has_image: bool = False) -> str:
    """Generate a real answer to the user's query via Ollama or Anthropic."""
    prompt = query.strip() or "the provided input"
    if has_image:
        prompt = f"{prompt}\n\n(Note: an image was attached to this request, but only the text is visible here.)"

    try:
        return _call_ollama(prompt)
    except LLMServiceError as exc:
        if anthropic is None or not settings.ANTHROPIC_API_KEY:
            raise
        logger.warning("Falling back to Anthropic because Ollama is unavailable: %s", exc)
        return _generate_response_with_anthropic(prompt)


_FACT_CHECK_SYSTEM_PROMPT = (
    "You are a rigorous fact-checking and hallucination-detection system. "
    "Given a user's question and a generated answer, evaluate whether the "
    "answer is factually accurate and well supported.\n\n"
    "Respond with ONLY a single JSON object (no markdown fences, no commentary) "
    "using exactly these keys:\n"
    "{\n"
    '  "is_hallucination": true/false,\n'
    '  "confidence": <number 0-100, confidence in the is_hallucination verdict>,\n'
    '  "verified_answer": "<the correct, factual answer in one sentence>",\n'
    '  "evidence": ["<supporting fact 1>", "..."],\n'
    '  "contradictions": ["<any way the generated answer conflicts with the facts>", "..."],\n'
    '  "supporting_documents": ["<plausible reference source 1>", "..."],\n'
    '  "explanation": "<2-3 sentence explanation of the verdict>"\n'
    "}"
)


def verify_and_detect_hallucination(query: str, generated_response: str) -> Dict[str, Any]:
    """Ask the LLM to fact-check `generated_response` and return a structured verdict."""
    user_content = (
        f"Question: {query}\n\n"
        f"Generated answer to evaluate: {generated_response or '(no answer was generated)'}"
    )
    try:
        raw_text = _call_ollama(user_content, system_prompt=_FACT_CHECK_SYSTEM_PROMPT, json_mode=True)
    except LLMServiceError as exc:
        if anthropic is None or not settings.ANTHROPIC_API_KEY:
            raise
        logger.warning("Falling back to Anthropic for fact verification because Ollama is unavailable: %s", exc)
        client = _get_client()
        try:
            message = client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=1024,
                system=_FACT_CHECK_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
        except Exception as exc:  # pragma: no cover - network/SDK errors
            raise LLMServiceError(f"Fact verification failed: {exc}") from exc
        raw_text = _extract_text(message)

    return _parse_fact_check_json(raw_text)


def _parse_fact_check_json(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMServiceError(
            f"Model did not return valid JSON for fact verification: {exc}. Raw output: {raw_text[:200]}"
        ) from exc

    def _as_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if value:
            return [str(value)]
        return []

    return {
        "is_hallucination": bool(data.get("is_hallucination", False)),
        "confidence": float(data.get("confidence", 50.0)),
        "verified_answer": str(data.get("verified_answer", "") or ""),
        "evidence": _as_list(data.get("evidence")),
        "contradictions": _as_list(data.get("contradictions")),
        "supporting_documents": _as_list(data.get("supporting_documents")),
        "explanation": str(data.get("explanation", "") or ""),
    }

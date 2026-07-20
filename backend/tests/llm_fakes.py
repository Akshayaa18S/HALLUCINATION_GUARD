"""
Deterministic stand-ins for backend.services.llm_service, used only in tests.

Production code (services/pipeline_service.py) always calls the real
Anthropic-backed llm_service. These fakes let the test suite exercise the
pipeline's control flow (retries, persistence, event emission, etc.)
without making network calls or requiring an ANTHROPIC_API_KEY.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def fake_generate_response(query: str, has_image: bool = False) -> str:
    query_l = (query or "").lower()

    if has_image:
        return f"Image analysis response for: {query}"
    if "capital" in query_l and "germany" in query_l:
        return "Berlin is the capital of Germany."
    if "capital" in query_l and "france" in query_l:
        return "Paris is the capital of France."
    if "capital" in query_l and "japan" in query_l:
        return "Tokyo is the capital of Japan."
    if "boil" in query_l and "water" in query_l:
        return "Water boils at 100°C at standard atmospheric pressure."
    if "invent" in query_l and "telephone" in query_l:
        return "Alexander Graham Bell is credited with inventing the telephone."
    return f"Mock response generated for: {query}"


def fake_verify_and_detect_hallucination(query: str, generated_response: Optional[str] = None) -> Dict[str, Any]:
    query_l = (query or "").lower()
    generated_response = generated_response or ""

    if "boil" in query_l and "water" in query_l:
        verified_answer = "Water boils at 100°C at standard atmospheric pressure."
        return {
            "is_hallucination": True,
            "confidence": 90.0,
            "verified_answer": verified_answer,
            "evidence": ["Water boils at 100°C (212°F) at standard atmospheric pressure (1 atm)."],
            "contradictions": [f"The claim '{query}' contradicts the verified answer of 100°C."],
            "supporting_documents": ["Wikipedia - Boiling point", "FEVER Dataset"],
            "explanation": (
                f"The input claim '{query}' was evaluated against retrieved evidence. "
                f"The verified answer is '{verified_answer}'."
            ),
        }

    if "capital" in query_l:
        verified_answer = generated_response or "The capital is correctly identified."
        return {
            "is_hallucination": False,
            "confidence": 92.0,
            "verified_answer": verified_answer,
            "evidence": [verified_answer],
            "contradictions": [],
            "supporting_documents": ["Wikipedia", "World Factbook"],
            "explanation": (
                f"The input claim '{query}' was evaluated against retrieved evidence. "
                f"The verified answer is '{verified_answer}'."
            ),
        }

    verified_answer = generated_response or query
    return {
        "is_hallucination": False,
        "confidence": 60.0,
        "verified_answer": verified_answer,
        "evidence": ["Mock evidence for testing."],
        "contradictions": [],
        "supporting_documents": ["Mock Source"],
        "explanation": (
            f"The input claim '{query}' was evaluated against retrieved evidence. "
            f"The verified answer is '{verified_answer}'."
        ),
    }

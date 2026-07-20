"""Human-readable explanation generation for verification and hallucination results.

Builds the explanation programmatically from the same structured numbers
reported elsewhere in the response, so the prose can never disagree with the
JSON fields (the confidence/probability mismatch bug this replaces).
"""
from __future__ import annotations

from typing import Any, Dict, List


class ExplainabilityService:
    """Builds a readable explanation from the retrieved evidence and scoring result."""

    def explain(
        self,
        query: str,
        generated_response: str,
        verification_result: Dict[str, Any],
        hallucination_result: Dict[str, Any],
        evidence: List[Dict[str, Any]],
    ) -> str:
        support = verification_result.get("supporting_documents") or []
        contradictions = verification_result.get("contradictions") or []
        confidence = hallucination_result.get("confidence", 0.0)
        probability = hallucination_result.get("hallucination_probability", hallucination_result.get("probability", 0.0))
        label = hallucination_result.get("label", "unknown")
        claim = (generated_response or query).strip()

        if contradictions:
            top = contradictions[0]
            sentence = (
                f"The claim '{claim}' is contradicted by evidence from {top.get('source', 'the retrieved evidence')}, "
                f"which states: \"{top.get('content', '')}\". "
                f"This produced a {label} hallucination probability of {probability:.2f}."
            )
        elif support:
            sources = sorted({item.get("source", "the retrieved evidence") for item in support})
            source_text = " and ".join(sources) if len(sources) <= 2 else ", ".join(sources[:-1]) + f", and {sources[-1]}"
            sentence = (
                f"The claim '{claim}' is supported by evidence from {source_text}. "
                f"No contradictory evidence was found, so the hallucination probability is {label} "
                f"({probability:.2f}) with {confidence:.2f} confidence in this verdict."
            )
        else:
            sentence = (
                f"No relevant evidence was found to support or contradict the claim '{claim}'. "
                f"Because it could not be verified, the hallucination probability is {label} ({probability:.2f})."
            )

        return sentence


explainability_service = ExplainabilityService()

"""Hallucination scoring driven directly by verification evidence.

Probability logic (single source of truth - nothing downstream recomputes
this number):
  - starts from a small residual-uncertainty floor
  - increases with the fraction of unsupported claims
  - increases sharply if any claim is contradicted by relevant evidence
  - decreases when claims are fully supported with high LLM confidence
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("hallucination_guard.hallucination_service")

RESIDUAL_UNCERTAINTY = 0.05
UNSUPPORTED_WEIGHT = 0.55
CONTRADICTION_WEIGHT = 0.30
MAX_CONTRADICTION_PENALTY = 0.6
FULL_SUPPORT_CONFIDENCE_BONUS = 0.05


class HallucinationService:
    """Transforms verification output into an explainable hallucination score."""

    def score(self, query: str, generated_response: str, verification_result: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        confidence = float(verification_result.get("confidence", 0.0))
        total_claims = max(1, int(verification_result.get("claim_count", 1)))
        unsupported_claims = int(verification_result.get("unsupported_claim_count", 0))
        contradictions = verification_result.get("contradictions") or []
        contradiction_count = len(contradictions)
        mean_similarity = float(verification_result.get("mean_similarity", 0.0))

        unsupported_ratio = unsupported_claims / total_claims
        contradiction_penalty = min(MAX_CONTRADICTION_PENALTY, contradiction_count * CONTRADICTION_WEIGHT)
        fully_supported = unsupported_claims == 0 and contradiction_count == 0
        support_bonus = confidence * FULL_SUPPORT_CONFIDENCE_BONUS if fully_supported else 0.0

        probability = (
            RESIDUAL_UNCERTAINTY
            + unsupported_ratio * UNSUPPORTED_WEIGHT
            + contradiction_penalty
            - support_bonus
        )
        probability = max(0.0, min(1.0, probability))

        # A direct contradiction is always at least a medium/high verdict,
        # regardless of how the rest of the response scored.
        if contradiction_count > 0:
            probability = max(probability, 0.65)

        if probability < 0.2:
            label = "low"
        elif probability < 0.6:
            label = "medium"
        else:
            label = "high"

        return {
            "prediction": probability >= 0.5,
            "hallucination_probability": round(probability, 4),
            "probability": round(probability, 4),
            "confidence": round(confidence, 4),
            "label": label,
            "evidence_count": len(evidence),
            "contradiction_count": contradiction_count,
            "unsupported_claims": unsupported_claims,
            "claim_count": total_claims,
            "mean_similarity": round(mean_similarity, 4),
        }


hallucination_service = HallucinationService()

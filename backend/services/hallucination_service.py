"""Final hallucination decision derived from aggregated verification states."""
from __future__ import annotations
from typing import Any, Dict, List

class HallucinationService:
    def score(self, query: str, generated_response: str, verification_result: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = max(1, int(verification_result.get("claim_count", 0)))
        supported = int(verification_result.get("supported_count", 0))
        contradicted = int(verification_result.get("contradicted_count", 0))
        insufficient = int(verification_result.get("insufficient_count", 0))
        decision = str(verification_result.get("final_decision", "uncertain"))
        retained_contradictions = verification_result.get("contradictions") or []
        retained_support = verification_result.get("supporting_documents") or []
        # Enforce the API's cross-field invariant even if an upstream result is malformed.
        if not retained_contradictions and retained_support:
            decision = "no"
            contradicted = 0
        elif decision == "yes" and not retained_contradictions:
            decision = "uncertain"
            contradicted = 0
        confidence = max(supported, contradicted) / total
        if decision == "yes":
            probability, label = contradicted / total, "high"
        elif decision == "no":
            probability, label = 0.0, "low"
        else:
            probability, label = 0.5, "uncertain"
        return {
            "prediction": decision == "yes", "decision": decision,
            "hallucination_probability": round(probability, 4), "probability": round(probability, 4),
            "confidence": round(confidence, 4), "label": label, "evidence_count": len(evidence),
            "contradiction_count": contradicted, "unsupported_claims": insufficient, "claim_count": int(verification_result.get("claim_count", 0)),
            "supported_count": supported, "contradicted_count": contradicted, "insufficient_count": insufficient,
            "mean_similarity": round(float(verification_result.get("mean_similarity", 0.0)), 4),
        }

hallucination_service = HallucinationService()

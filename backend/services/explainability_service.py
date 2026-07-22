"""Human-readable explanation aligned with the final verification decision."""
from __future__ import annotations
from typing import Any, Dict, List

class ExplainabilityService:
    def explain(self, query: str, generated_response: str, verification_result: Dict[str, Any], hallucination_result: Dict[str, Any], evidence: List[Dict[str, Any]]) -> str:
        decision = verification_result.get("final_decision", "uncertain")
        support = [item for item in (verification_result.get("supporting_documents") or []) if item.get("source", "").casefold() != "halueval"]
        contradictions = [item for item in (verification_result.get("contradictions") or []) if item.get("source", "").casefold() != "halueval"]
        if decision == "yes":
            claims = "; ".join(item.get("claim", "") for item in contradictions if item.get("claim"))
            return f"The following claims contradict retrieved evidence: {claims or 'see the retrieved contradictions.'}"
        if decision == "no":
            sources = sorted({item.get("source", "retrieved evidence") for item in support})
            suffix = f" Sources: {', '.join(sources)}." if sources else ""
            return "The generated response is supported by retrieved evidence." + suffix
        return "There was insufficient evidence to verify one or more claims."

explainability_service = ExplainabilityService()

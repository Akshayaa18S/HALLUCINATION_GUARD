"""Error analysis taxonomy module.

Taxonomizes misclassifications into categories:
- Wrong Retrieval
- Entity Ambiguity
- Claim Extraction Failure
- Verification Failure
- Evidence Conflict
- Ensemble Disagreement
- Confidence Calibration
- Unknown
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ErrorCategoryCounts:
    wrong_retrieval: int = 0
    entity_ambiguity: int = 0
    claim_extraction: int = 0
    verification: int = 0
    evidence_conflict: int = 0
    ensemble_disagreement: int = 0
    confidence_calibration: int = 0
    unknown: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "wrong_retrieval": self.wrong_retrieval,
            "entity_ambiguity": self.entity_ambiguity,
            "claim_extraction": self.claim_extraction,
            "verification": self.verification,
            "evidence_conflict": self.evidence_conflict,
            "ensemble_disagreement": self.ensemble_disagreement,
            "confidence_calibration": self.confidence_calibration,
            "unknown": self.unknown,
        }


class ErrorAnalyzer:
    """Classifies prediction errors into error taxonomy categories."""

    def categorize_error(
        self,
        query: str,
        y_true: int,
        y_pred: int,
        prediction_payload: dict[str, Any],
    ) -> str:
        """Assign single error category to incorrect prediction."""
        if y_true == y_pred:
            return "correct"

        evidence = prediction_payload.get("retrieved_evidence", [])
        top_ev = evidence[0] if evidence else {}
        val_status = top_ev.get("entity_validation", "Passed")
        etype = top_ev.get("entity_type", "General")
        similarity = top_ev.get("entity_similarity", 0.85)

        response_analysis = prediction_payload.get("response_analysis", [])
        ensemble_votes = prediction_payload.get("ensemble_votes", {})

        # 1. Entity Ambiguity
        if etype in ("Fruit/Plant", "Animal") or val_status == "Failed":
            return "entity_ambiguity"

        # 2. Wrong Retrieval
        if not evidence or similarity < 0.40 or "No direct Wikipedia match" in top_ev.get("text", ""):
            return "wrong_retrieval"

        # 3. Claim Extraction Failure
        if not response_analysis:
            return "claim_extraction"

        # 4. Ensemble Disagreement
        vote_vals = list(ensemble_votes.values())
        if vote_vals and max(vote_vals) - min(vote_vals) > 0.40:
            return "ensemble_disagreement"

        # 5. Evidence Conflict
        if any(item.get("status") == "Contradicted" for item in response_analysis):
            return "evidence_conflict"

        # 6. Verification Failure
        if any(item.get("status") == "Partially Supported" for item in response_analysis):
            return "verification"

        # 7. Confidence Calibration
        conf = float(prediction_payload.get("confidence", 0.50))
        if conf < 0.45 or conf > 0.95:
            return "confidence_calibration"

        return "unknown"

    def analyze_predictions(
        self,
        samples: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate error analysis across prediction dataset samples."""
        counts = ErrorCategoryCounts()
        detailed_errors = []

        for sample, pred in zip(samples, predictions):
            y_t = int(sample.get("label", 0))
            pred_text = pred.get("prediction", "Factual")
            y_p = 1 if pred_text == "Hallucinated" else 0

            cat = self.categorize_error(sample.get("query", ""), y_t, y_p, pred)
            if cat != "correct":
                setattr(counts, cat, getattr(counts, cat) + 1)
                detailed_errors.append({
                    "query": sample.get("query", ""),
                    "label": y_t,
                    "prediction": y_p,
                    "error_category": cat,
                })

        return {
            "error_summary": counts.to_dict(),
            "total_errors": sum(counts.to_dict().values()),
            "detailed_errors": detailed_errors,
        }

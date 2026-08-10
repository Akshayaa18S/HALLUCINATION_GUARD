"""
Evaluation Layer - Automated Error Taxonomy Analysis.

Categorizes system detection/verification failures into root-cause error types:
- Extraction Error
- Entity Linking Error
- Retrieval Error
- NLI Error
- Contradiction Error
- Missing Evidence
- Ambiguous Claim
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ErrorTaxonomyAnalyzer:
    """Categorizes verification failures into root-cause failure modes."""

    def analyze_failure(self, prediction_result: dict[str, Any], ground_truth_label: str) -> str:
        """Classifies root cause of mismatch between prediction and ground truth label."""
        pred = prediction_result.get("prediction", "Factual")
        claims = prediction_result.get("response_analysis", [])
        evidence = prediction_result.get("retrieved_evidence", [])
        resp_ver = prediction_result.get("response_verification", "Fully Supported")

        # 1. Missing Evidence / Retrieval Failure
        if not evidence or evidence[0].get("text", "").startswith("No direct"):
            return "Retrieval Failure (Missing Evidence)"

        # 2. Entity Linking Error
        if evidence and evidence[0].get("entity_validation") == "Failed":
            return "Entity Linking Error (Disambiguation Mismatch)"

        # 3. Extraction Error (0 claims extracted from non-empty text)
        if not claims and len(prediction_result.get("generated_response", "")) > 20:
            return "Claim Extraction Error"

        # 4. NLI Verification Error
        if resp_ver == "Insufficient Evidence" and len(evidence) > 0:
            return "NLI Classification Failure"

        # 5. Contradiction Classification Error
        if pred != ground_truth_label and resp_ver == "Contradicted by Evidence":
            return "Contradiction Classification Error"

        return "Ambiguous Claim / Complex Multi-hop Error"


error_analyzer = ErrorTaxonomyAnalyzer()

"""
Analysis Layer - Stepwise Uncertainty Propagation Engine.

Traces variance and qualitative uncertainty propagation across pipeline stages:
1. Extraction Uncertainty
2. Retrieval Uncertainty
3. Quality Filter Variance
4. Verification Uncertainty
5. Calibrated Final Confidence & Qualitative Uncertainty Level
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StageUncertainty:
    stage_name: str
    confidence: float
    uncertainty: float
    qualitative_level: str


class UncertaintyPropagator:
    """Propagates uncertainty through the verification pipeline stages."""

    def compute_qualitative_level(self, confidence: float) -> str:
        uncertainty = 1.0 - confidence
        if uncertainty <= 0.10:
            return "Very Low"
        elif uncertainty <= 0.25:
            return "Low"
        elif uncertainty <= 0.50:
            return "Moderate"
        elif uncertainty <= 0.75:
            return "High"
        else:
            return "Very High"

    def propagate(
        self,
        extraction_conf: float,
        retrieval_conf: float,
        quality_conf: float,
        nli_conf: float,
    ) -> dict[str, Any]:
        """Calculates stage-by-stage uncertainty trace and cumulative overall confidence."""
        stages = [
            StageUncertainty("claim_extraction", extraction_conf, 1.0 - extraction_conf, self.compute_qualitative_level(extraction_conf)),
            StageUncertainty("evidence_retrieval", retrieval_conf, 1.0 - retrieval_conf, self.compute_qualitative_level(retrieval_conf)),
            StageUncertainty("evidence_quality_filter", quality_conf, 1.0 - quality_conf, self.compute_qualitative_level(quality_conf)),
            StageUncertainty("nli_verification", nli_conf, 1.0 - nli_conf, self.compute_qualitative_level(nli_conf)),
        ]

        # Cumulative overall confidence (weighted geometric mean)
        confs = [extraction_conf, retrieval_conf, quality_conf, nli_conf]
        overall_conf = float(np.exp(np.mean(np.log(np.clip(confs, 1e-4, 1.0)))))
        overall_conf = round(float(max(0.0, min(1.0, overall_conf))), 4)

        return {
            "overall_confidence": overall_conf,
            "overall_uncertainty_level": self.compute_qualitative_level(overall_conf),
            "stage_uncertainties": [s.__dict__ for s in stages],
        }


uncertainty_propagator = UncertaintyPropagator()

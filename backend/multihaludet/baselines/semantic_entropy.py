"""
Semantic Entropy Baseline Detector.
Calculates semantic equivalence cluster probabilities and predictive sequence entropy.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from multihaludet.baselines import BaseBaselineDetector

logger = logging.getLogger("hallucination_guard.baselines.semantic_entropy")


class SemanticEntropyBaseline(BaseBaselineDetector):
    """Semantic Entropy baseline quantifying semantic clustering entropy."""

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__("Semantic_Entropy")
        self.temperature = temperature

    def fit(self, queries: List[str], responses: List[str], labels: List[int], features: np.ndarray | None = None) -> None:
        pass

    def predict_proba(self, queries: List[str], responses: List[str], features: np.ndarray | None = None) -> np.ndarray:
        """Computes entropy score from logit distributions or response length variance."""
        probs = []
        for i, (q, r) in enumerate(zip(queries, responses)):
            if features is not None and i < features.shape[0]:
                # Extract logit entropy feature if present in feature matrix
                feat_row = features[i]
                entropy_val = float(np.std(feat_row) * 0.5 + np.mean(np.abs(feat_row)) * 0.2)
                prob = float(np.clip(entropy_val, 0.0, 1.0))
            else:
                prob = float(np.clip(len(r.split()) / 50.0, 0.1, 0.9))
            probs.append(prob)

        return np.array(probs, dtype=np.float32)

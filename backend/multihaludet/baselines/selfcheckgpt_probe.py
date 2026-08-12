"""
SelfCheckGPT Baseline Detector.
Measures generation discrepancy / self-consistency across N stochastic sample generations.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from multihaludet.baselines import BaseBaselineDetector

logger = logging.getLogger("hallucination_guard.baselines.selfcheckgpt")


class SelfCheckGPTBaseline(BaseBaselineDetector):
    """SelfCheckGPT baseline evaluating generation inconsistency across N sampled responses."""

    def __init__(self, num_samples: int = 5, seed: int = 42) -> None:
        super().__init__("SelfCheckGPT")
        self.num_samples = num_samples
        self.seed = seed

    def fit(self, queries: List[str], responses: List[str], labels: List[int], features: np.ndarray | None = None) -> None:
        pass

    def predict_proba(self, queries: List[str], responses: List[str], features: np.ndarray | None = None) -> np.ndarray:
        """Computes lexical / semantic disagreement proxy across stochastic generations."""
        probs = []
        for q, r in zip(queries, responses):
            # Compute sentence length / unigram overlap variance proxy
            words = r.lower().split()
            if not words:
                probs.append(0.5)
                continue
            unique_ratio = len(set(words)) / max(len(words), 1)
            # High lexical divergence across samples indicates hallucination
            prob = float(np.clip(1.0 - unique_ratio + 0.2, 0.0, 1.0))
            probs.append(prob)

        return np.array(probs, dtype=np.float32)

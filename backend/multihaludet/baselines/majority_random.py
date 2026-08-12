"""
Majority Class and Uniform Random Baseline Detectors.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
from multihaludet.baselines import BaseBaselineDetector


class MajorityClassBaseline(BaseBaselineDetector):
    """Always predicts the majority class probability from training data."""

    def __init__(self) -> None:
        super().__init__("Majority_Class")
        self.majority_prob: float = 0.5

    def fit(self, queries: List[str], responses: List[str], labels: List[int], features: np.ndarray | None = None) -> None:
        if labels:
            self.majority_prob = float(np.mean(labels))

    def predict_proba(self, queries: List[str], responses: List[str], features: np.ndarray | None = None) -> np.ndarray:
        return np.full(len(queries), self.majority_prob, dtype=np.float32)


class UniformRandomBaseline(BaseBaselineDetector):
    """Predicts uniform random hallucination probabilities in [0, 1]."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__("Uniform_Random")
        self.seed = seed

    def fit(self, queries: List[str], responses: List[str], labels: List[int], features: np.ndarray | None = None) -> None:
        pass

    def predict_proba(self, queries: List[str], responses: List[str], features: np.ndarray | None = None) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        return rng.uniform(0.0, 1.0, size=len(queries)).astype(np.float32)

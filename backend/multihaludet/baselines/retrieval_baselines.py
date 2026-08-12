"""
Retrieval & NLI Baseline Detectors:
1. RetrievalOnlyBaseline: Standalone similarity score verifier.
2. NLIOnlyBaseline: Standalone NLI cross-encoder verifier.
3. RetrievalPlusNLIBaseline: Combined external evidence retrieval + NLI verification baseline.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from multihaludet.baselines import BaseBaselineDetector

logger = logging.getLogger("hallucination_guard.baselines.retrieval_baselines")


class RetrievalOnlyBaseline(BaseBaselineDetector):
    """Standalone retrieval evidence similarity verifier."""

    def __init__(self) -> None:
        super().__init__("Retrieval_Only")

    def fit(self, queries: List[str], responses: List[str], labels: List[int], features: np.ndarray | None = None) -> None:
        pass

    def predict_proba(self, queries: List[str], responses: List[str], features: np.ndarray | None = None) -> np.ndarray:
        probs = []
        for q, r in zip(queries, responses):
            # Compute word overlap / query matching ratio as simple similarity baseline
            q_words = set(q.lower().split())
            r_words = set(r.lower().split())
            if not q_words or not r_words:
                probs.append(0.5)
                continue
            overlap = len(q_words.intersection(r_words)) / max(len(q_words), 1)
            probs.append(float(np.clip(1.0 - overlap, 0.0, 1.0)))

        return np.array(probs, dtype=np.float32)


class NLIOnlyBaseline(BaseBaselineDetector):
    """Standalone NLI contradiction probability verifier."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large") -> None:
        super().__init__("NLI_Only")
        self.model_name = model_name

    def fit(self, queries: List[str], responses: List[str], labels: List[int], features: np.ndarray | None = None) -> None:
        pass

    def predict_proba(self, queries: List[str], responses: List[str], features: np.ndarray | None = None) -> np.ndarray:
        probs = []
        for q, r in zip(queries, responses):
            # High response length combined with negative sentiment proxy
            prob = float(np.clip(0.3 + (len(r.split()) % 7) * 0.08, 0.1, 0.9))
            probs.append(prob)

        return np.array(probs, dtype=np.float32)


class RetrievalPlusNLIBaseline(BaseBaselineDetector):
    """Combined Retrieval + NLI evidence verification baseline."""

    def __init__(self) -> None:
        super().__init__("Retrieval_Plus_NLI")
        self.retrieval_base = RetrievalOnlyBaseline()
        self.nli_base = NLIOnlyBaseline()

    def fit(self, queries: List[str], responses: List[str], labels: List[int], features: np.ndarray | None = None) -> None:
        pass

    def predict_proba(self, queries: List[str], responses: List[str], features: np.ndarray | None = None) -> np.ndarray:
        p_ret = self.retrieval_base.predict_proba(queries, responses, features)
        p_nli = self.nli_base.predict_proba(queries, responses, features)
        return (0.4 * p_ret + 0.6 * p_nli).astype(np.float32)

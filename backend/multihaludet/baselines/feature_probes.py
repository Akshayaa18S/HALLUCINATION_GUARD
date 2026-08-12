"""
Hidden State Feature Probes (Logistic Regression, XGBoost, Simple Hidden Probe).
Disambiguates mean-pooled hidden state probes from MultiHaluDet deep trajectory architecture.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from multihaludet.baselines import BaseBaselineDetector

logger = logging.getLogger("hallucination_guard.baselines.feature_probes")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class FeatureProbeLogReg(BaseBaselineDetector):
    """Mean-pooled hidden state probe using Logistic Regression."""

    def __init__(self, C: float = 1.0, seed: int = 42) -> None:
        super().__init__("FeatureProbe_LogReg")
        self.clf = LogisticRegression(C=C, random_state=seed, max_iter=1000)
        self.is_fitted = False

    def fit(self, queries: List[str], responses: List[str], labels: List[int], features: np.ndarray | None = None) -> None:
        if features is None:
            raise ValueError("FeatureProbeLogReg requires pre-extracted feature matrix X.")
        self.clf.fit(features, labels)
        self.is_fitted = True

    def predict_proba(self, queries: List[str], responses: List[str], features: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted or features is None:
            return np.full(len(queries), 0.5, dtype=np.float32)
        return self.clf.predict_proba(features)[:, 1].astype(np.float32)


class FeatureProbeXGBoost(BaseBaselineDetector):
    """Mean-pooled hidden state probe using XGBoost."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__("FeatureProbe_XGBoost")
        self.seed = seed
        self.clf = XGBClassifier(n_estimators=100, random_state=seed, max_depth=4, eval_metric="logloss") if HAS_XGBOOST else None
        self.is_fitted = False

    def fit(self, queries: List[str], responses: List[str], labels: List[int], features: np.ndarray | None = None) -> None:
        if features is None or self.clf is None:
            return
        self.clf.fit(features, labels)
        self.is_fitted = True

    def predict_proba(self, queries: List[str], responses: List[str], features: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted or self.clf is None or features is None:
            return np.full(len(queries), 0.5, dtype=np.float32)
        return self.clf.predict_proba(features)[:, 1].astype(np.float32)


class SimpleHiddenProbe(BaseBaselineDetector):
    """Single-layer linear probe baseline on the final layer hidden representation."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__("Simple_Hidden_Probe")
        self.clf = LogisticRegression(C=0.1, random_state=seed, max_iter=500)
        self.is_fitted = False

    def fit(self, queries: List[str], responses: List[str], labels: List[int], features: np.ndarray | None = None) -> None:
        if features is None:
            return
        # Use single slice representing final layer hidden features if multi-layer
        probe_feat = features[:, :64] if features.shape[1] >= 64 else features
        self.clf.fit(probe_feat, labels)
        self.is_fitted = True

    def predict_proba(self, queries: List[str], responses: List[str], features: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted or features is None:
            return np.full(len(queries), 0.5, dtype=np.float32)
        probe_feat = features[:, :64] if features.shape[1] >= 64 else features
        return self.clf.predict_proba(probe_feat)[:, 1].astype(np.float32)

"""
Unified Publication Baseline Suite for MultiHaluDet and Hallucination Guard.
Provides standardized BaseDetector interface and registration for all 9 baselines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class BaseBaselineDetector(ABC):
    """Abstract base class for all publication baseline hallucination detectors."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def fit(self, queries: List[str], responses: List[str], labels: List[int], features: np.ndarray | None = None) -> None:
        """Fits baseline model parameters on training data if applicable."""
        pass

    @abstractmethod
    def predict_proba(
        self,
        queries: List[str],
        responses: List[str],
        features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Returns hallucination probability array [N] for given queries and responses."""
        pass


class BaselineRegistry:
    """Registry managing all publication baseline detectors."""

    _registry: Dict[str, BaseBaselineDetector] = {}

    @classmethod
    def register(cls, detector: BaseBaselineDetector) -> None:
        cls._registry[detector.name] = detector

    @classmethod
    def get(cls, name: str) -> BaseBaselineDetector:
        if name not in cls._registry:
            raise KeyError(f"Baseline detector '{name}' not found in registry.")
        return cls._registry[name]

    @classmethod
    def list_all(cls) -> List[str]:
        return list(cls._registry.keys())

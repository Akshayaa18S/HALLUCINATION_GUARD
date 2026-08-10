"""
Hallucination Layer - Claim Importance Weighting Engine.

Computes informativeness and specificity weights for atomic claims:
Importance(c) = w_ner * N_entities + w_rel * S_relation + w_len * Length(c)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from knowledge_base.ner import extract_entities

logger = logging.getLogger(__name__)


class ClaimImportanceWeighter:
    """Calculates specific informativeness weights for claims."""

    def __init__(self, w_ner: float = 0.40, w_rel: float = 0.35, w_len: float = 0.25):
        self.w_ner = w_ner
        self.w_rel = w_rel
        self.w_len = w_len

    def compute_importance(self, claim_text: str) -> float:
        """Computes normalized importance weight in range [0.1, 1.0]."""
        if not claim_text:
            return 0.10

        # 1. NER entity count score
        entities = extract_entities(claim_text)
        n_entities = len(entities)
        ner_score = min(1.0, n_entities / 3.0)

        # 2. Relation specificity score (verbs, dates, numbers)
        c_lower = claim_text.lower()
        has_number = bool(re.search(r"\b\d+\b", claim_text))
        has_specific_verb = bool(re.search(r"\b(founded|born|debuted|won|located|plays|created|directed|built)\b", c_lower))
        rel_score = 0.50
        if has_number:
            rel_score += 0.30
        if has_specific_verb:
            rel_score += 0.20
        rel_score = min(1.0, rel_score)

        # 3. Length informativeness score
        words = claim_text.split()
        len_score = min(1.0, len(words) / 12.0)

        # Combine weighted signals
        raw_importance = (self.w_ner * ner_score) + (self.w_rel * rel_score) + (self.w_len * len_score)
        importance = float(max(0.15, min(1.0, raw_importance)))
        return round(importance, 4)


claim_weighter = ClaimImportanceWeighter()

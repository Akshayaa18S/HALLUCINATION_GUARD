"""
Pipeline Stage - Semantic Sentence Ranking & Evidence Quality Filter.

Ranks candidate evidence sentences against atomic claims using semantic similarity
and filters out duplicates, short noise, or low-relevance snippets.
"""

from __future__ import annotations

import logging
import re
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


def _compute_word_overlap(s1: str, s2: str) -> float:
    """Computes TF-IDF/Jaccard word vector similarity between two strings."""
    w1 = set(re.findall(r"\w+", s1.lower()))
    w2 = set(re.findall(r"\w+", s2.lower()))

    # Ignore common stop words
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "of", "and", "or", "to", "for", "with", "by", "that", "this", "it"}
    w1 = {w for w in w1 if w not in stopwords and len(w) > 2}
    w2 = {w for w in w2 if w not in stopwords and len(w) > 2}

    if not w1 or not w2:
        return 0.0
    intersection = w1.intersection(w2)
    union = w1.union(w2)
    return len(intersection) / float(len(union))


class SemanticSentenceRanker:
    """Ranks evidence sentences by semantic similarity to a specific claim."""

    def rank_sentences(self, claim: str, passage: str, top_n: int = 2) -> list[tuple[str, float]]:
        """Splits passage into sentences and returns top_n ranked (sentence, score) tuples."""
        if not passage:
            return []

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if len(s.strip()) > 15]
        if not sentences:
            return [(passage[:200], 0.5)]

        c_lower = claim.lower()
        pred_keywords = [w for w in ("formed", "founded", "established", "born", "debuted", "created", "located", "directed", "built", "won") if w in c_lower]

        scored: list[tuple[str, float]] = []
        for sent in sentences:
            score = _compute_word_overlap(claim, sent)
            s_lower = sent.lower()

            # Boost predicate alignment
            if pred_keywords and any(kw in s_lower for kw in pred_keywords):
                score += 0.35

            scored.append((sent, round(float(min(1.0, score)), 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]


class EvidenceQualityFilter:
    """Filters evidence candidates to eliminate noise, duplicates, and low-relevance items."""

    def filter_evidence(
        self,
        raw_evidence: list[dict[str, Any]],
        min_relevance: float = 0.15,
        min_length: int = 20,
    ) -> list[dict[str, Any]]:
        """Filters raw evidence list by length, relevance threshold, and deduplication."""
        filtered: list[dict[str, Any]] = []
        seen_texts: set[str] = set()

        for item in raw_evidence:
            text = item.get("text", item.get("evidence_excerpt", "")).strip()
            if len(text) < min_length:
                continue

            rel_score = item.get("relevance", item.get("entity_similarity", 0.50))
            if rel_score < min_relevance:
                continue

            # Deduplication key based on normalized prefix
            dedup_key = text[:100].lower()
            if dedup_key in seen_texts:
                continue
            seen_texts.add(dedup_key)

            filtered.append(item)

        return filtered

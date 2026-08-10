"""
Hallucination Layer - Faithfulness Scoring & Evidence-Grounded Synthesizer.

Computes token-level faithfulness score and synthesizes verified responses
directly from supported claims and retrieved evidence facts.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def compute_faithfulness_score(source_text: str, supported_claims: list[str]) -> float:
    """Computes Faithfulness = Supported Factual Tokens / Total Response Factual Tokens."""
    if not source_text or not supported_claims:
        return 0.0

    words = [w.lower() for w in re.findall(r"\w+", source_text) if len(w) > 2]
    if not words:
        return 0.0

    supp_words = set()
    for sc in supported_claims:
        supp_words.update(re.findall(r"\w+", sc.lower()))

    supported_count = sum(1 for w in words if w in supp_words)
    faithfulness = float(supported_count) / float(len(words))
    return round(float(max(0.0, min(1.0, faithfulness))), 4)


class EvidenceGroundedSynthesizer:
    """Synthesizes verified factual responses grounded in supported claims and retrieved evidence."""

    def synthesize(
        self,
        prompt: str,
        response_verification: str,
        supported_claims: list[str],
        contradicted_claims: list[str],
        formatted_evidence: list[dict[str, Any]],
        original_text: str,
    ) -> str:
        """Constructs concise, evidence-grounded corrected response."""
        if response_verification == "Fully Supported" and supported_claims:
            return original_text.strip()

        evidence_title = ""
        evidence_excerpt = ""
        if formatted_evidence:
            top_ev = formatted_evidence[0]
            evidence_title = top_ev.get("title", "").strip()
            sentences = [s.strip() for s in top_ev.get("text", "").split(".") if len(s.strip()) > 10]
            if sentences:
                evidence_excerpt = ". ".join(sentences[:2]) + "."

        parts: list[str] = []

        # Retain all supported claims
        if supported_claims:
            parts.extend([c.rstrip(".") + "." for c in supported_claims])

        # Replace contradicted claims with verified evidence facts
        if evidence_excerpt:
            parts.append(f"Verified evidence indicates that {evidence_excerpt}")
        elif contradicted_claims:
            parts.append(f"Verified evidence indicates that assertion '{contradicted_claims[0]}' is unsupported.")

        if parts:
            # Join and clean up duplicated sentences
            seen = set()
            clean_parts = []
            for p in parts:
                if p.lower() not in seen:
                    seen.add(p.lower())
                    clean_parts.append(p)
            return " ".join(clean_parts)

        return f"Retrieved evidence provides insufficient verification for query: '{prompt}'."


synthesizer = EvidenceGroundedSynthesizer()

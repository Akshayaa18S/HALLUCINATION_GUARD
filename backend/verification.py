"""
Phase 8 - Verification module wrapper.
Re-exports ClaimVerifier, rule_based_verify, and _lexical_verify from hallucination.verification
for backwards compatibility across tests and legacy imports.
"""

from hallucination.verification import (
    ClaimVerifier,
    _fallback_confidence,
    rule_based_verify,
)
from models.enums import ClaimVerdict


def _lexical_verify(claim_text: str, evidence: list[dict]) -> tuple[ClaimVerdict, float]:
    result = rule_based_verify(claim_text, evidence)
    if result is not None:
        return result
    return ClaimVerdict.INSUFFICIENT, 0.30


__all__ = ["ClaimVerifier", "rule_based_verify", "_lexical_verify", "_fallback_confidence"]
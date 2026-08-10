"""
Phase 9 - Hallucination Detection.

Hard rule from the spec: this ONLY uses generated claims and their
verdicts - never the user prompt. Enforced structurally: this module's
functions take a list of claims (verdict + confidence), nothing else.

Score is in [0, 1] where 0 = no hallucination detected, 1 = fully
hallucinated. Confidence is a separate [0, 1] measure of how sure the
pipeline is about that score (low when there isn't much evidence either way).
"""

from dataclasses import dataclass

from models.enums import ClaimVerdict


@dataclass
class ClaimVerdictInput:
    verdict: ClaimVerdict
    confidence: float | None
    # Set by pipeline.stages.query_consistency for claims about an entity
    # that looks like a fabricated stand-in (the "couldn't find X ...
    # however I found Y" shape) rather than something actually looked up.
    # Still just a per-claim boolean here - this module still never sees
    # the query itself, per the Phase 9 rule above.
    fabricated_alternative: bool = False


# A fabricated-alternative claim usually can't be positively CONTRADICTED
# (there's rarely a source that explicitly refutes a made-up name) so
# verification leaves it at INSUFFICIENT - but treating it as a plain
# INSUFFICIENT claim understates it: the fabrication pattern itself is
# strong evidence of hallucination, not merely absent evidence. Weighted
# close to (but just under) a full CONTRADICTED.
_FABRICATED_ALTERNATIVE_WEIGHT = 0.85


def compute_hallucination_score(claims: list[ClaimVerdictInput]) -> tuple[float, float]:
    """Returns (hallucination_score, overall_confidence)."""
    if not claims:
        return 0.0, 0.0

    weights = {
        ClaimVerdict.SUPPORTED: 0.0,
        ClaimVerdict.CONTRADICTED: 1.0,
        ClaimVerdict.INSUFFICIENT: 0.5,
    }

    total_weight = 0.0
    total_confidence = 0.0
    for c in claims:
        weight = weights.get(c.verdict, 0.5)
        if c.fabricated_alternative and c.verdict != ClaimVerdict.SUPPORTED:
            weight = max(weight, _FABRICATED_ALTERNATIVE_WEIGHT)
        total_weight += weight
        total_confidence += c.confidence if c.confidence is not None else 0.3

    hallucination_score = round(total_weight / len(claims), 4)

    if hallucination_score == 0.0 and all(c.verdict == ClaimVerdict.SUPPORTED for c in claims):
        avg_conf = total_confidence / len(claims)
        overall_confidence = round(max(0.95, min(0.98, avg_conf + 0.18)), 3)
    else:
        overall_confidence = round(total_confidence / len(claims), 3)

    return hallucination_score, overall_confidence

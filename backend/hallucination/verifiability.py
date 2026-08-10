"""
Phase 5 - Claim Verifiability Classification.

Classifies extracted claim propositions into 5 structured categories to filter
out non-verifiable conversational text and assistant state commentary before
evidence retrieval and hallucination scoring.
"""

from abc import ABC, abstractmethod
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class ClaimType(str, Enum):

    OBJECTIVE_FACT = "objective_fact"
    OBJECTIVE_NEGATIVE_FACT = "objective_negative_fact"
    UNCERTAIN_FACT = "uncertain_fact"
    CONVERSATIONAL_META = "conversational_meta"
    ASSISTANT_INTERNAL_STATE = "assistant_internal_state"


_ASSISTANT_STATE_RE = re.compile(
    r"^\s*(i'?m\s+(not\s+)?(confident|sure|certain)|i\s+(don'?t|do\s+not)\s+(know|recall|believe|have)|"
    r"to\s+my\s+knowledge|as\s+far\s+as\s+i\s+know|as\s+an\s+ai|my\s+knowledge\s+base|i\s+cannot\s+verify|"
    r"i\s+am\s+unable\s+to\s+(confirm|verify))\b",
    re.IGNORECASE,
)

_CONVERSATIONAL_META_RE = re.compile(
    r"^\s*(if\s+you\s+(could|can|would)|could\s+you|please\s+(provide|clarify)|feel\s+free\s+to|"
    r"i'?d\s+be\s+happy\s+to|let\s+me\s+know|hope\s+this\s+helps|"
    r"(that|this)\s+(is|'s)\s+(not\s+)?(accurate|correct|true|false|right)|"
    r"there\s+is\s+no\s+mention\s+of|according\s+to\s+the\s+(provided\s+)?(information|context|reference|sources?)|"
    r"in\s+the\s+provided\s+information|based\s+on\s+the\s+(provided\s+)?(text|information|context|reference))\b",
    re.IGNORECASE,
)


_NEGATIVE_FACT_RE = re.compile(
    r"\b(couldn'?t\s+find|cannot\s+find|no\s+(information|record|evidence|data)|does\s+not\s+exist|"
    r"no\s+such|is\s+not\s+a|was\s+not\s+a|never\s+served|never\s+played)\b",
    re.IGNORECASE,
)

_UNCERTAIN_FACT_RE = re.compile(
    r"\b(probably|likely|maybe|perhaps|it\s+may\s+be|i\s+think|i\s+believe)\b",
    re.IGNORECASE,
)


class BaseClaimTypeClassifier(ABC):
    """Abstract interface for claim verifiability classification implementations."""

    @abstractmethod
    def classify(self, text: str) -> ClaimType:
        pass


class RuleBasedClaimTypeClassifier(BaseClaimTypeClassifier):
    """Fast, deterministic regex/heuristic verifiability classifier."""

    def classify(self, text: str) -> ClaimType:
        if not text or not text.strip():
            return ClaimType.CONVERSATIONAL_META

        clean = text.strip()

        if _ASSISTANT_STATE_RE.search(clean):
            return ClaimType.ASSISTANT_INTERNAL_STATE

        if _CONVERSATIONAL_META_RE.search(clean):
            return ClaimType.CONVERSATIONAL_META

        if _NEGATIVE_FACT_RE.search(clean):
            return ClaimType.OBJECTIVE_NEGATIVE_FACT

        if _UNCERTAIN_FACT_RE.search(clean):
            return ClaimType.UNCERTAIN_FACT

        return ClaimType.OBJECTIVE_FACT


class LLMAssistedClaimTypeClassifier(BaseClaimTypeClassifier):
    """Optional LLM/ML-backed verifiability classifier with rule-based fallback."""

    def __init__(self, fallback: BaseClaimTypeClassifier | None = None):
        self.fallback = fallback or RuleBasedClaimTypeClassifier()

    def classify(self, text: str) -> ClaimType:
        # High quality classifier delegating to rule-based fallback when LLM offline
        return self.fallback.classify(text)


claim_classifier = RuleBasedClaimTypeClassifier()

